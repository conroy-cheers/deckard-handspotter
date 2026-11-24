#!/usr/bin/env python3
"""Render synthetic hands through a calibrated stereo or four-camera rig with three.js."""
import argparse
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from soma_wardrobe import GARMENTS, fit_garment
from garmentcode_fit import garment_library
from soma_protocol import Worker, packet
from scipy.spatial.transform import Rotation

from capture_optics import LensDecentering, sample_motion, sample_optical_blur
from capture_variation import (crossing_targets, lens_parameters, reach_behind_forearm, sample_appearance,
                               scene_debris, shape_body, shape_hand)
from foundation_scenes import PLACEMENTS, validate_scenes
from frame_camera import digest
from handspotter_cameras import (BlendedHandspotterCamera, Fisheye624Camera, QUAD_STREAMS, SIDES,
                                load_quad_rig, load_stereo_rig)
from head_pose import in_reference_scene, sample_head_pose, transform_hands, transform_points, turn_baked_head
from handspotter_data import save_json
from humanoid_geometry import hand_pose_plan
from render import Cubemap, read_assets, topology, unit
from render_job import job_fields, parse_render_args
from replay_surface import vertex_normals, wrist_boundary
from synthetic_arm import Body, arm_surface, hemisphere_arm, pose_hand, sample_arm, wide_reach_target
from synthetic_torso import SHOULDER_INSET, torso_surface


ENVIRONMENTS = ("desk", "workbench", "kitchen")
FOUNDATION = tuple("foundation-"+name for name in PLACEMENTS)
ASSETS = (
    "skin-color.jpg", "skin-roughness.jpg", "skin-height.png",
    "wood-color.jpg", "wood-normal.jpg", "wood-roughness.jpg",
    "concrete-color.jpg", "concrete-normal.jpg", "concrete-roughness.jpg",
    "fabric-color.jpg", "fabric-normal.jpg", "fabric-roughness.jpg",
    "desk.hdr", "workbench.hdr",
)


def pbr_assets(path, limit):
    if path.suffix!=".npz":
        return read_assets(path,limit)
    with np.load(path,allow_pickle=False) as bank:
        return [dict(id=str(name),vertices=vertices,joints=joints)
                for name,vertices,joints in zip(bank["ids"][:limit],bank["vertices"][:limit],bank["joints"][:limit],strict=True)]


def pbr_topology(path, assets):
    if path.suffix!=".npz":
        return topology(path)
    with np.load(path,allow_pickle=False) as bank:
        faces = bank["faces"]
    vertex_count = len(assets[0]["vertices"])
    if (faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer)
            or not len(faces) or faces.min()<0 or faces.max()>=vertex_count
            or any(a["vertices"].shape!=(vertex_count,3) or a["joints"].shape!=(21,3) for a in assets)):
        raise ValueError("Pose bank and hand topology are incompatible")
    return faces.astype(np.int32)


def smoothstep(start, end, x):
    t = np.clip((x-start)/(end-start), 0, 1)
    return t*t*(3-2*t)


def skin_regions(vertices, normals, joints):
    """Measure dorsal knuckles and distal nail patches in the rest surface."""
    # AnyHand's right-hand frame points from the little finger toward the
    # index finger on X, and from the wrist toward the fingers on Y.
    dorsal = smoothstep(.0, .65, -normals[:, 2])
    palm = smoothstep(.05, .65, normals[:, 2])
    knuckles = np.zeros(len(vertices))
    nails = np.zeros(len(vertices))
    for first in (1, 5, 9, 13, 17):
        points = joints[first:first+4]
        for joint in points[:3]:
            distance = np.linalg.norm(vertices-joint, axis=1)
            knuckles = np.maximum(knuckles, 1-smoothstep(.007, .013, distance))
        axis = unit(points[-1]-points[-2])
        outward = np.array([0., 0., -1.])+axis*axis[2]
        if np.linalg.norm(outward) < .15:
            continue
        tangent = unit(np.cross(axis, outward))
        length = np.linalg.norm(points[-1]-points[-2])
        center = points[-1] - .53*length*axis
        relative = vertices-center
        distance = np.hypot((relative@axis)/(.64*length), (relative@tangent)/(.40*length))
        patch = (1-smoothstep(.60, 1.16, distance))*smoothstep(.05, .55, normals@unit(outward))
        nails = np.maximum(nails, patch)
    hand = smoothstep(-.008, .024, vertices[:, 1])
    return np.column_stack((palm*hand, knuckles*dorsal, nails))


def palm_lines(joints):
    knuckles = joints[[5, 9, 13, 17], :2]
    center = knuckles.mean(axis=0)
    breadth = np.linalg.norm(knuckles[0]-knuckles[-1])
    length = np.linalg.norm(center)
    curves = np.array([
        [[-.49,.77],[-.27,.84],[.01,.81],[.34,.72]],
        [[-.44,.58],[-.16,.62],[.12,.67],[.44,.75]],
        [[.42,.75],[.23,.55],[.17,.30],[.29,.08]],
    ])
    curves *= [breadth, length]
    curves[..., 0] += center[0]
    return np.concatenate((curves[:, :-1], curves[:, 1:]), axis=2).reshape(-1, 4).tolist()


def mesh_record(asset, triangles, boundary, arm, side, scale, skin, cuff, appearance, *, target_only=False):
    vertices, joints = pose_hand(asset, arm, side, scale)
    hand = dict(asset=asset["id"], keypoints_pair_m=joints.tolist(), arm=arm.record())
    if target_only and triangles is None:
        return dict(position=np.vstack((arm.elbow, arm.wrist, joints))), hand
    faces = triangles[:, ::-1] if side == 0 else triangles
    indices, edges = boundary
    world, rest, faces = arm_surface(
        vertices, asset["vertices"], faces, (indices, edges[:, ::-1] if side == 0 else edges), arm, appearance,
        shoulder_inset=SHOULDER_INSET,
    )
    if target_only:
        return dict(position=world), hand
    normal = vertex_normals(world, faces)
    rest_faces = faces[:, ::-1] if side == 0 else faces
    rest_normal = vertex_normals(rest, rest_faces)
    regions = skin_regions(rest, rest_normal, asset["joints"])
    mesh = dict(
        part=side+1, hand_triangles=len(triangles), skin=skin, cuff=cuff,
        position=world.ravel().tolist(), normal=normal.ravel().tolist(),
        rest=rest.ravel().tolist(), rest_normal=rest_normal.ravel().tolist(),
        regions=regions.ravel().tolist(), palm_lines=palm_lines(asset["joints"]),
        faces=faces.ravel().tolist(),
    )
    if appearance:
        mesh["appearance"] = appearance
    return mesh, hand


def body_mesh(body, arms, boundary, appearance):
    rim, edges = boundary
    count = len(rim)
    openings = {arm["part"]-1: (np.asarray(arm["position"]).reshape(-1,3)[-count:],
                                edges[:, ::-1] if arm["part"]==1 else edges) for arm in arms}
    chest_scale = 1+.22*(appearance["arm_girth"][1]-1) if appearance else 1.
    positions, faces, seams = torso_surface(body, openings, chest_scale)
    normals = vertex_normals(positions, faces)
    for arm in arms:
        seam = seams[arm["part"]-1]
        ring_normals = np.asarray(arm["normal"][-3*count:]).reshape(-1,3)
        shared = ring_normals+normals[seam]
        shared /= np.linalg.norm(shared,axis=1,keepdims=True).clip(1e-9)
        normals[seam] = shared
        arm["normal"][-3*count:] = shared.ravel().tolist()
    record = dict(shoulder_centers_pair_m=[[-body.half_shoulder_width,body.shoulder_y,body.shoulder_z],
                                         [body.half_shoulder_width,body.shoulder_y,body.shoulder_z]],
                  chest_depth_scale=chest_scale)
    return dict(position=positions.ravel().tolist(), normal=normals.ravel().tolist(),
                faces=faces.ravel().tolist(), appearance=appearance), record


def detector_optics(camera, lens):
    """Small uniform blocks for the GPU's angular projection and its derivative."""
    def endpoint(model):
        if isinstance(model, Fisheye624Camera):
            center = np.asarray(model.camera.c)
            coefficients = model.coefficients
        else:
            center = (model.camera.c + .5) * model.scale - .5
            coefficients = np.pad(model.camera.distortion[:4], (0, 8))
            theta = np.linspace(0, model.angular_limit, 32769)
            radial = theta * np.polynomial.polynomial.polyval(theta**2, np.r_[1., coefficients[:4]])
            if np.any(np.diff(radial) <= 0):
                raise ValueError("Nonmonotonic KB4 domain")
        return dict(f=model.focal.tolist(), c=center.tolist(), coefficients=coefficients.tolist())

    blend = isinstance(camera, BlendedHandspotterCamera)
    if blend:
        camera.radial_domain()
    source = endpoint(camera.source if blend else camera)
    return dict(source=source, frame=endpoint(camera.frame) if blend else source,
                fraction=camera.fraction if blend else 0., limit=camera.limit if blend else camera.angular_limit,
                rotation=camera.transform[:3, :3].tolist(),
                decenter=lens.record() if lens else None)


def project(camera, lens, points):
    xy, depth = camera.project(points)
    return (lens.project(xy, camera.width) if lens else xy), depth


ROOM_PIVOT = np.array([0., -.29, -.56])


def upright_layout(layout):
    """Level older recorded rooms around the same work-surface anchor."""
    if "rotation" not in layout:
        return layout
    rotation = Rotation.from_quat(layout["rotation"])
    if np.allclose(rotation.apply([0, 1, 0]), [0, 1, 0]):
        return layout
    forward = rotation.apply([0, 0, 1])
    level = Rotation.from_euler("Y", np.arctan2(forward[0], forward[2]))
    origin = layout["origin"] + rotation.apply(ROOM_PIVOT) - level.apply(ROOM_PIVOT)
    return dict(layout, rotation=level.as_quat().tolist(), origin=origin.tolist())


def scene_setup(seed, index, environment, meshes, vary_position=False):
    render_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    yaw = Rotation.from_euler("Y", render_rng.uniform(-.29, .29))
    render_rng.random(2)  # Preserve seeded lighting and furniture after removing pitch and roll.
    matrix = yaw.as_matrix()
    origin = ROOM_PIVOT - yaw.apply(ROOM_PIVOT)
    if vary_position:
        position_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 31]))
        origin += [position_rng.uniform(-.18, .18), 0., position_rng.uniform(-.08, .08)]
    height = -.29
    for mesh in meshes:
        points = (np.asarray(mesh["position"]).reshape(-1, 3)-origin)@matrix
        above_top = points[(np.abs(points[:, 0])<.78)&(points[:, 2]<-.39)]
        if len(above_top):
            height = min(height, float(above_top[:, 1].min())-.033)
    lighting = dict(yaw=float(render_rng.uniform(-1.1, 1.1)),
                    exposure=float(render_rng.uniform(.64, 1.27)),
                    elevation=float(render_rng.uniform(.52, .83)),
                    key=float(render_rng.uniform(1.4, 2.6)),
                    fill=float(render_rng.uniform(.55, 1.0)))
    light_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 4]))
    lighting["fixture"] = light_rng.choice(("window", "ceiling", "task")).item()
    layout = dict(name=environment, height=height, seed=int(render_rng.integers(2**24)),
                  rotation=yaw.as_quat().tolist(), origin=origin.tolist())
    if environment in FOUNDATION:
        model = environment.removeprefix("foundation-")
        layout["foundation"] = dict(model=model, **PLACEMENTS[model])
        if layout["foundation"]["light"] == "courtyard":
            lighting["fixture"] = "sun"
    return layout, lighting


@dataclass(frozen=True)
class SampleSetup:
    index: int
    environment: str
    environment_source: str
    added_garment: str | None
    cloth_mode: str | None
    empty_room: bool = False

    def scene(self, seed, meshes=(), record=None, vary_position=False):
        if self.environment_source == "recorded" and "lighting" in record:
            return upright_layout(record["environment"]), record["lighting"]
        layout, lighting = scene_setup(seed, self.index, self.environment, meshes, vary_position)
        if self.environment_source == "recorded":
            layout = upright_layout(record["environment"])
        if self.environment_source != "override" and record is not None:
            lighting = record.get("lighting", lighting)
        return layout, lighting


def resolve_sample(index, seed, names, environment, wardrobe, cloth_mode, library, record=None):
    if environment is None and record is not None and "environment" in record:
        room = record["environment"]["name"]
        source = "recorded"
    else:
        slot = index % len(names)
        if record is None or environment == "random":
            order = np.random.SeedSequence([seed, index - slot, 3])
            room = str(np.random.default_rng(order).permutation(names)[slot])
        else:
            room = names[slot]
        source = "override" if environment is not None else "generated"
    garment = wardrobe
    if wardrobe == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 8122]))
        pool = tuple(library) if library and rng.random() < .5 else tuple(GARMENTS)
        garment = str(rng.choice(pool))
    elif wardrobe in ("balanced", "garmentcode-balanced"):
        garments = tuple(GARMENTS if wardrobe == "balanced" else library)
        garment = garments[(seed + index) % len(garments)]
    return SampleSetup(index, room, source, garment, cloth_mode if garment else None)


def prepare_scenes(args, assets, triangles, directory, setups, streams):
    boundary = wrist_boundary(triangles) if triangles is not None else None
    entries = []
    targets_only = args.soma_worker is not None
    for setup in setups:
        index = setup.index
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        negative = args.negative_every and (index + (args.seed if args.randomize_poses else 0)) % args.negative_every == 0
        hemisphere = args.hemisphere_reach or (args.randomize_poses and args.wide_reach and rng.random() < .5)
        meshes, hands = [], {}
        appearance = None
        capture = None
        target = foreground = None
        if args.capture_noise:
            capture_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 6]))
            severity = float(capture_rng.uniform(.10,.59)) if capture_rng.random() < .80 else 0.
            capture = dict(lens=[lens_parameters(capture_rng, severity) for _ in streams])
            if not negative and capture_rng.random() < .39 and not hemisphere:
                foreground, target = crossing_targets(capture_rng)
        body = Body.sample(rng)
        skin = float(rng.uniform(.18, .85))
        cuff = float(rng.choice([-.055, -.095, -1.]))
        if args.appearance_variation:
            appearance_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 5]))
            appearance = sample_appearance(appearance_rng)
            cuff = appearance["cuff"]
            body = shape_body(body, appearance)
        if not negative:
            subjects = {}
            for side, presence in enumerate((.82, .84)):
                if rng.random() >= presence:
                    continue
                asset = assets[rng.integers(len(assets))]
                if appearance:
                    asset = shape_hand(asset, appearance)
                reach = target if side==foreground else None
                varied = args.wide_reach and reach is None
                if varied:
                    reach_rng = np.random.default_rng(np.random.SeedSequence([args.seed,index,7,side]))
                    reach = wide_reach_target(reach_rng,side)
                if hemisphere:
                    reach_rng = np.random.default_rng(np.random.SeedSequence([args.seed,index,19,side]))
                    other = subjects[0][1] if 0 in subjects else None
                    arm = hemisphere_arm(reach_rng, body, side, other)
                else:
                    arm = sample_arm(rng, body, side, target=reach, wide=varied)
                scale = rng.uniform(.93, 1.07)*np.sqrt(body.forearm_length/.255)
                subjects[side] = asset, arm, scale
            if foreground is not None and len(subjects)==2:
                background = 1-foreground
                asset, _, scale = subjects[background]
                arm = reach_behind_forearm(capture_rng, body, background, subjects[foreground][1], asset, scale)
                subjects[background] = asset, arm, scale
            for side, (asset, arm, scale) in subjects.items():
                mesh, hand = mesh_record(asset, triangles, boundary, arm, side, scale, skin, cuff, appearance,
                                         target_only=targets_only)
                meshes.append(mesh)
                hands[SIDES[side]] = hand
        layout, lighting = setup.scene(args.seed, meshes, vary_position=args.randomize_poses)
        torso, upper_body = (None, {}) if targets_only else body_mesh(body, meshes, boundary, appearance)
        entry = dict(id=f"{index:05d}", environment=layout, lighting=lighting, upper_body=upper_body,
                     hands=hands, views={})
        if hemisphere:
            entry["reach_profile"] = "hemisphere"
        job = dict(meshes=meshes, torso=torso, lighting=lighting, layout=layout)
        if appearance:
            entry["appearance"] = appearance
        if capture is not None:
            capture["crossing_foreground"] = SIDES[foreground] if foreground is not None and len(hands)==2 else None
            capture["debris"] = scene_debris(capture_rng, hands)
            entry["capture"] = capture
            job["debris"] = capture["debris"]
        if not targets_only:
            save_json(directory/f"{index}.json", job)
        entries.append(entry)
    return entries


def prepare_body_job(args, directory, setup, entry, bank):
    """Stage a resolved body pose for the PBR worker."""
    index = setup.index
    surfaces = []
    arrays = {}
    garment = None
    skin_contact = None
    for prefix in ("skin","cloth"):
        if prefix=="cloth" and "cloth_faces" not in bank:
            continue
        v,f,labels = (bank[f"{prefix}_{name}"] for name in ("vertices","faces","labels"))
        if (v.ndim!=2 or v.shape[1]!=3 or f.shape!=(len(labels),3)
                or not len(f) or f.min()<0 or f.max()>=len(v)
                or not np.isfinite(v).all() or not np.isin(labels,[1,2,3]).all()):
            raise ValueError(f"Invalid {prefix} surface in body {index}")
        if prefix=="skin" and "head_pose" in entry and not entry["head_pose"].get("native_soma"):
            v = turn_baked_head(v,f,labels,entry["head_pose"])
        if prefix == "skin" and setup.added_garment:
            arrays.update(contact_position=v.astype("<f4",copy=False).ravel(),
                          contact_faces=f.astype("<u4",copy=False).ravel())
            skin_contact = dict(material="skin",position="contact_position",faces="contact_faces")
        normals = vertex_normals(v,f)
        for part in np.unique(labels):
            faces = f[labels==part]
            used,indices = np.unique(faces,return_inverse=True)
            geometry = dict(position=v[used].astype("<f4",copy=False).ravel(),
                            normal=normals[used].astype("<f4",copy=False).ravel(),
                            faces=indices.astype("<u4",copy=False).ravel())
            references = {name:f"{prefix}_{part}_{name}" for name in geometry}
            arrays.update({references[name]:values for name,values in geometry.items()})
            surfaces.append(dict(material=prefix,part=int(part),**references))
    if not setup.added_garment and "cloth_faces" not in bank:
        entry.pop("wardrobe", None)
    if setup.added_garment:
        required = {"rest_vertices","rest_cloth_joints","rest_arm_joints"}
        if not required.issubset(bank) or "cloth_faces" in bank:
            raise ValueError("SOMA-X wardrobe requires a bare body bank baked with --cloth-reference")
        name = setup.added_garment
        rng = np.random.default_rng(np.random.SeedSequence([args.seed,index,8123]))
        garment = fit_garment(bank,entry,name,rng,args.wardrobe_library)
        entry["appearance"] = dict(entry.get("appearance") or {},garment=name,
            cloth_roughness=garment.pop("roughness"),pattern=garment.pop("pattern"))
        entry["wardrobe"] = dict(garment=name,mode=setup.cloth_mode,
            extension_mm=garment.pop("coverage_mm"))
        if "pattern_source" in garment:
            entry["wardrobe"]["pattern_source"] = garment["pattern_source"]
    capture = entry.get("capture") or {}
    job = dict(meshes=[],surfaces=surfaces,appearance=entry.get("appearance"),layout=entry["environment"],
               lighting=entry["lighting"],debris=capture.get("debris",[]))
    if garment:
        job["garment"] = garment
        job["skin_contact"] = skin_contact
    (directory/f"{index}.bin").write_bytes(packet(job, arrays))

def prepare_baked_scenes(args, directory, records, setups):
    for setup,entry in zip(setups, records, strict=True):
        with np.load(args.body_bank/f"{setup.index:05d}.npz",allow_pickle=False) as bank:
            prepare_body_job(args,directory,setup,entry,bank)


def sensor_settings(args, entry, index, eye):
    seed = np.random.SeedSequence([args.seed, index, eye, 2])
    haze = (entry.get("capture") or {}).get("lens", ())
    return dict(noise_seed=int(seed.generate_state(1)[0]),
                filter_sigma=float(np.random.default_rng(seed).uniform(.38, .67)),
                optical_sigma=sample_optical_blur(args.seed, index, eye) if args.slight_blur else 0.,
                haze=haze[eye] if args.capture_noise and eye < len(haze) else None)


def read_sensor(path, camera, native):
    # The GPU writes one bounding rectangle per hand, then their pixel counts,
    # the 160x128 Handspotter input bytes, and optionally native RGBA.
    data = path.read_bytes()
    end = 64 + 128 * 160
    if len(data) != end + (camera.height * camera.width * 4 if native else 0):
        raise ValueError(f"Incomplete GPU sensor output: {path}")
    bounds, counts = np.frombuffer(data, "<f4", count=16).reshape(2, 2, 4)
    visible = {}
    for label, side in enumerate(SIDES):
        if counts[label, 0] == 0:
            continue
        visible[side] = dict(box_xyxy=bounds[label].astype(int).tolist(), visible_pixels=int(counts[label, 0]))
    tensor = np.frombuffer(data, np.uint8, count=128*160, offset=64).reshape(128, 160)
    image = np.frombuffer(data, np.uint8, offset=end).reshape(camera.height, camera.width, 4) if native else None
    return visible, tensor, image


HAND_KEYPOINTS = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    *(f"{finger}_{joint}" for finger in ("index", "middle", "ring", "pinky")
      for joint in ("mcp", "pip", "dip", "tip")),
)
HAND_SKELETON = [
    link for base in range(2, len(HAND_KEYPOINTS)+1, 4)
    for link in ([1, base], [base, base+1], [base+1, base+2], [base+2, base+3])
]


def coco_view(image_id, file_name, mask, visible):
    height, width = mask.shape
    image = dict(id=image_id, file_name=file_name, width=width, height=height)
    annotations = []
    for category_id, side in enumerate(SIDES, 1):
        if side not in visible:
            continue
        hand = visible[side]
        x, y, right, bottom = hand["box_xyxy"]
        segmentation = coco_mask.encode(np.asfortranarray(mask == category_id, dtype=np.uint8))
        segmentation["counts"] = segmentation["counts"].decode("ascii")
        keypoints = []
        for (col, row), in_front in zip(hand["keypoints_xy"], hand["keypoints_in_front"], strict=True):
            if in_front and 0 <= col < width and 0 <= row < height:
                visibility = 2 if mask[int(row), int(col)] == category_id else 1
                keypoints.extend((col, row, visibility))
            else:
                keypoints.extend((0, 0, 0))
        annotations.append(dict(id=(image_id-1)*len(SIDES)+category_id,
                                image_id=image_id, category_id=category_id,
                                bbox=[x, y, right-x, bottom-y], area=hand["visible_pixels"],
                                segmentation=segmentation, iscrowd=0, keypoints=keypoints,
                                num_keypoints=sum(v > 0 for v in keypoints[2::3])))
    return image, annotations


def capture_outputs(args, entries, cameras, lenses, directory, streams, indices):
    if not args.tensors_only:
        for folder in ("images", "masks"):
            (args.out/folder).mkdir()
    quad = streams==QUAD_STREAMS
    filenames = streams if quad else SIDES

    def write_group(item):
        slot,(index,entry) = item
        if "head_pose" in entry:
            to_pair = np.linalg.inv(entry["head_pose"]["reference_from_pair"])
            entry["hands"] = transform_hands(entry["hands"], to_pair)
            shoulders = entry["upper_body"].get("shoulder_centers_pair_m")
            if shoulders is not None:
                entry["upper_body"]["shoulder_centers_pair_m"] = transform_points(shoulders, to_pair).tolist()
            for arm in entry.get("shutter_motion", {}).get("arms", {}).values():
                for joint in ("shoulder", "elbow", "wrist", "palm"):
                    arm[joint] = transform_points(arm[joint], to_pair).tolist()
                arm["rotation_deg"] = (to_pair[:3, :3] @ arm["rotation_deg"]).tolist()
        views = []
        coco_views = []
        for eye, (camera, lens, stream) in enumerate(zip(cameras, lenses, streams, strict=True)):
            visible, tensor, image = read_sensor(directory/f"{slot}-{eye}.bin", camera, not args.tensors_only)
            name = f"{index:05d}-{filenames[eye]}.png"
            view = {}
            if args.slight_blur:
                view["optical_blur_sigma_native_px"] = sample_optical_blur(args.seed, index, eye)
            if not args.tensors_only:
                for folder, channel in (("images", 0), ("masks", 1)):
                    if not cv2.imwrite(str(args.out/folder/name), image[:, :, channel], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise OSError(f"Could not save {folder}/{name}")
                view.update(image=f"images/{name}", mask=f"masks/{name}")
            for side, hand in visible.items():
                xy, depth = project(camera, lens, np.array(entry["hands"][side]["keypoints_pair_m"]))
                if quad and args.handspotter_render_scale!=8:
                    enlarge = 8/args.handspotter_render_scale
                    hand["box_xyxy"] = [value*enlarge for value in hand["box_xyxy"]]
                    hand["visible_pixels"] *= enlarge**2
                    xy = (xy+.5)*enlarge-.5
                hand.update(keypoints_xy=xy.tolist(), keypoints_in_front=(depth>0).tolist())
            if image is not None:
                image_id = slot*len(streams) + eye + 1
                coco_views.append(coco_view(image_id, view["image"], image[:, :, 1], visible))
            entry["views"][stream] = dict(view, visible_hands=visible)
            views.append(tensor)
        return views, coco_views

    with ThreadPoolExecutor(max_workers=args.output_workers) as pool:
        groups = list(pool.map(write_group, enumerate(zip(indices, entries, strict=True))))
    data = np.asarray([views for views, _ in groups], np.uint8)
    if not args.tensors_only:
        coco_views = [view for _, views in groups for view in views]
        save_json(args.out/"coco.json", dict(
            info=dict(description="AnyHand Studio synthetic hand instances"), licenses=[],
            categories=[dict(id=i, name=f"{side}_hand", supercategory="hand",
                             keypoints=HAND_KEYPOINTS, skeleton=HAND_SKELETON) for i, side in enumerate(SIDES, 1)],
            images=[image for image, _ in coco_views],
            annotations=[ann for _, annotations in coco_views for ann in annotations],
        ))
    return data


def run(args, session=None):
    if args.count < 1 or args.negative_every < 0 or args.asset_limit < 1 or args.output_workers < 1 or not 64 <= args.face_size <= 2048:
        raise ValueError("Invalid scene count, asset limit, negative interval, or face size")
    if args.out.exists():
        raise FileExistsError(args.out)
    live = args.soma_worker is not None
    if args.soma_template and not live:
        raise ValueError("--soma-template requires --soma-worker")
    if live and not (args.soma_assets and args.soma_correctives):
        raise ValueError("Supply --soma-assets and --soma-correctives to the native worker")
    if live and args.body_bank:
        raise ValueError("Use --soma-template or --source for live SOMA-X rendering")
    if args.soma_hand_poses and not live:
        raise ValueError("--soma-hand-poses requires the native SOMA-X worker")
    hand_poses = args.soma_hand_poses or ("mixed" if live and not args.soma_template else "prior")
    sample = args.viewport if args.viewport is not None else args.capture_sample
    if sample is not None and not 0 <= sample < args.count:
        raise ValueError("Preview sample is outside the batch")
    indices = range(args.count) if sample is None else [sample]
    if live and args.rig_json and not args.openvr_calibration:
        raise ValueError("A native Frame headset attachment requires --openvr-calibration")
    if args.soma_wardrobe and not (args.body_bank or live):
        raise ValueError("--soma-wardrobe requires a body bank or native SOMA-X worker")
    args.wardrobe_library = garment_library(args.garment_library)
    wardrobe_names = {"random", "balanced", *GARMENTS, *args.wardrobe_library}
    if args.wardrobe_library:
        wardrobe_names.add("garmentcode-balanced")
    if args.soma_wardrobe and args.soma_wardrobe not in wardrobe_names:
        raise ValueError(f"Unknown wardrobe style {args.soma_wardrobe!r}")
    if args.body_bank and args.hemisphere_reach:
        raise ValueError("Generate a --hemisphere-reach template before baking humanoid meshes")
    baked = None
    records = None
    if args.body_bank:
        baked = json.loads((args.body_bank/"manifest.json").read_text())
        records = baked["records"]
        if len(records) != args.count:
            raise ValueError("The body bank must supply one body per requested group")
    elif args.soma_template and not args.randomize_poses:
        rows = json.loads(args.soma_template.read_text())["records"][:args.count]
        records = rows
        if len(rows) != args.count:
            raise ValueError("The SOMA-X template has fewer than --count groups")

    names = (args.environment,) if args.environment not in (None, "all", "random") else ENVIRONMENTS
    if args.foundation_scenes and (args.environment in ("all", "random") or (records is None and args.environment is None)):
        rooms = [room for room in PLACEMENTS if (args.foundation_scenes/room/f"{room}.gltf").is_file()]
        if not rooms:
            raise ValueError(f"No supported Foundation scenes found at {args.foundation_scenes}")
        names = (ENVIRONMENTS if args.environment == "random" else ()) + tuple(f"foundation-{room}" for room in rooms)
    setups = [resolve_sample(i, args.seed, names, args.environment, args.soma_wardrobe, args.cloth_mode,
                             args.wardrobe_library, records[i] if records is not None else None) for i in indices]
    if args.randomize_poses and args.negative_every:
        # Alternate empty backgrounds and resting wearers among scheduled negatives.
        setups = [replace(setup, empty_room=True, added_garment=None, cloth_mode=None)
                  if (args.seed + setup.index) % (2 * args.negative_every) == 0 else setup
                  for setup in setups]
    # A room edit only changes the layout and lights. Generated mesh rooms also
    # use the clearance height measured when their geometry was staged.
    imported = sorted({setup.environment.removeprefix("foundation-") for setup in setups
                       if setup.environment in FOUNDATION})
    foundation_files = {}
    if imported:
        args.foundation_scenes = args.foundation_scenes or args.packaged_foundation
        if args.foundation_scenes is None:
            raise ValueError("Choose a Foundation asset directory with --foundation-scenes")
        if session:
            foundation_files = session.foundation_files(args.foundation_scenes, imported[0])
        else:
            foundation_files = validate_scenes(args.foundation_scenes, imported)
    if session and session.geometry_matches(args):
        setup = setups[0]
        record = records[setup.index] if records is not None else None
        layout, lighting = setup.scene(args.seed, record=record, vary_position=args.randomize_poses)
        if record is None:
            layout["height"] = session.height
        return session.publish_room(args, setup, layout, lighting, foundation_files)
    gestures = hand_pose_plan(args.count, args.soma_seed, hand_poses)
    if records is not None:
        for setup in setups:
            row = records[setup.index]
            row["environment"], row["lighting"] = setup.scene(args.seed, record=row)
    for path in [args.three/"build/three.module.js", *(args.assets/name for name in ASSETS)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.toolkit:
        sys.path.insert(0, str(args.toolkit.resolve()))
    cv2.setNumThreads(1)
    calibration = args.rig_json or args.calibration
    source, transform, cameras = (load_quad_rig(calibration,args.handspotter_render_scale) if args.rig_json else load_stereo_rig(calibration))
    streams = tuple(source)
    if args.handspotter_render_scale!=8 and (not args.rig_json or not args.tensors_only):
        raise ValueError("Reduced Handspotter rendering requires --rig-json and --tensors-only")
    if args.tensors_only and not args.rig_json:
        raise ValueError("Handspotter tensors require a four-camera --rig-json")
    lenses = [LensDecentering.sample(np.random.default_rng(np.random.SeedSequence([args.seed, 17, eye])), camera)
              if args.lens_decentering else None for eye, camera in enumerate(cameras)]
    front_left,front_right = cameras[1:3] if args.rig_json else cameras
    rig_center = (front_left.origin+front_right.origin)*.5
    cube = Cubemap(args.face_size)
    if live:
        command = [args.soma_worker,"--assets",str(args.soma_assets),"--correctives",str(args.soma_correctives),
                   "--seed",str(0 if session else args.soma_seed)]
        if args.rig_json:
            command += ["--rig",str(args.rig_json),"--openvr",str(args.openvr_calibration)]
    stage_context = nullcontext(session.directory) if session else tempfile.TemporaryDirectory(prefix="anyhand-pbr-")
    with stage_context as stage:
        directory = Path(stage)
        if baked is not None:
            entries = [baked["records"][i] for i in indices]
        elif args.randomize_poses:
            # Only the palm landmarks are needed to orient the native SOMA wrists.
            donors = json.loads(args.soma_template.read_text())["records"]
            assets = []
            for row in donors:
                for side, hand in in_reference_scene(row)["hands"].items():
                    points = np.asarray(hand["keypoints_pair_m"], dtype=float)
                    points = points - points[0]
                    if side == "left":
                        points[:, 0] *= -1
                    assets.append(dict(id=f"{row['id']}-{side}", joints=points, vertices=points))
            if not assets:
                raise ValueError("Random reach generation needs a template with hand landmarks")
            entries = prepare_scenes(args, assets, None, directory, setups, streams)
        elif args.soma_template:
            entries = [in_reference_scene(rows[i]) for i in indices]
        else:
            assets = list({asset["id"]: asset for path in args.source
                           for asset in pbr_assets(path,args.asset_limit)}.values())
            entries = prepare_scenes(args,assets,pbr_topology(args.topology,assets),directory,setups,streams)
        motion = []
        worker_context = nullcontext()
        if live:
            worker_context = session.worker_for(command) if session else Worker(command)
        with worker_context as soma:
            for position,(setup,entry) in enumerate(zip(setups,entries,strict=True)):
                index = setup.index
                entry.pop("shutter_motion", None)
                if args.head_look and "head_pose" not in entry:
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed,index,18]))
                    entry["head_pose"] = sample_head_pose(rng,entry["hands"])
                if setup.empty_room:
                    scene = dict(meshes=[], surfaces=[], layout=entry["environment"], lighting=entry["lighting"])
                    (directory/f"{index}.bin").write_bytes(packet(scene, {}))
                elif soma:
                    requested = gestures[index]
                    metadata,bank = soma.request(op="pose",index=index,row=entry,gestures=requested,seed=args.soma_seed,
                                                 angles=args.pose_angles.get(index))
                    entry = metadata["record"]
                    if requested is not None and tuple(entry["hands"][side].get("gesture") for side in SIDES) != requested:
                        raise ValueError("SOMA-X worker did not return the requested hand gestures")
                    entries[position] = entry
                    prepare_body_job(args,directory,setup,entry,bank)
                entry["render_setup"] = asdict(setup)
                settings = sample_motion(args, index, entry["hands"])
                if settings:
                    entry["shutter_motion"] = settings
                motion.append(settings)
        if baked is not None:
            prepare_baked_scenes(args,directory,entries,setups)
        has_motion = any(motion)
        head_transforms = [
            np.linalg.inv(entry["head_pose"]["reference_from_pair"]).tolist()
            if "head_pose" in entry else None for entry in entries
        ]
        eyes = []
        for camera in cameras:
            faces = []
            for basis in cube.bases:
                axes = camera.transform[:3, :3] @ basis
                faces.append(dict(position=camera.origin.tolist(), up=(-axes[:, 1]).tolist(),
                                  look=(camera.origin+axes[:, 2]).tolist()))
            eyes.append(faces)
        optics = [detector_optics(c, lens) for c, lens in zip(cameras, lenses, strict=True)]
        save_json(directory/"job.json", dict(eyes=eyes, optics=optics, size=args.face_size, count=len(entries),
                                           viewport=args.viewport, sample=sample, streams=streams,
                                           viewport_setup=entries[0]["render_setup"] if args.viewport is not None else None,
                                           sensor_sizes=[(cam.width, cam.height) for cam in cameras],
                                           sensor_settings=[[sensor_settings(args, entry, index, eye) for eye in range(len(cameras))]
                                                            for entry,index in zip(entries,indices,strict=True)],
                                           native_images=not (args.tensors_only or session),
                                           binary_surfaces=live or baked is not None,
                                           foundation_files=list(foundation_files),
                                           motion=motion if has_motion else None, rig_center=rig_center.tolist(),
                                           pair_from_reference=head_transforms if any(head_transforms) else None,
                                           wardrobe_mode="drape" if any(s.added_garment for s in setups) else None))
        if session:
            session.cameras, session.lenses, session.streams = cameras, lenses, streams
            return session.publish_geometry(args, entries[0], foundation_files)
        worker = Path(__file__).with_name("pbr_worker.mjs")
        command = [args.node, str(worker), stage, str(args.assets.resolve()), str(args.three.resolve()),
                   args.chromium]
        if imported:
            command.append(str(args.foundation_scenes.resolve()))
        subprocess.run(command, check=True)
        if args.viewport is not None:
            return
        result = json.loads((directory/"result.json").read_text())
        descriptions = result.pop("optics_validation")
        if args.soma_wardrobe:
            dressed = (entry for entry in entries if entry["render_setup"]["added_garment"])
            for entry,stats in zip(dressed,result.pop("wardrobe_cases", []),strict=True):
                entry["wardrobe"]["fit"] = stats
        args.out.mkdir(parents=True, exist_ok=False)
        data = capture_outputs(args, entries, cameras, lenses, directory, streams, indices)
        if streams == QUAD_STREAMS:
            np.save(args.out/"handspotter.npy", data)
    sensor_calibration = {}
    for stream, cam, info, lens in zip(streams, cameras, descriptions, lenses, strict=True):
        sensor_calibration[stream] = dict(source_calibration=source[stream]["calibration"],
                                         pair_from_eye=cam.transform.tolist(), validation=info)
        if lens:
            sensor_calibration[stream]["lens_decentering"] = lens.record()
    save_json(args.out/"rig.json", dict(stream_order=streams, pair_from_device=transform.tolist(),
        baseline_mm=float(np.linalg.norm(front_left.origin-front_right.origin)*1000),
        source_sha256=digest(calibration), cameras=sensor_calibration))
    save_json(args.out/"dataset.json", dict(schema=1, seed=args.seed, face_size=args.face_size,
        handspotter_render_scale=args.handspotter_render_scale if args.rig_json else None, rig="rig.json",
        wide_reach=args.wide_reach, stream_order=streams, records=entries, renderer=result,
        topology_sha256=digest(args.topology) if args.topology else None,
        source_sha256=({p.name:digest(p) for p in args.source} if args.source else
                       {args.soma_template.name:digest(args.soma_template)} if args.soma_template else
                       {args.body_bank.name:digest(args.body_bank/"manifest.json")}),
        **({"soma":dict(seed=args.soma_seed,hand_poses=hand_poses,correctives_sha256=digest(args.soma_correctives),
                        openvr_sha256=digest(args.openvr_calibration) if args.rig_json else None)} if live else {}),
        material_sha256={name: digest(args.assets/name) for name in ASSETS}, foundation_sha256=foundation_files))
    print(json.dumps(dict(pairs=len(entries), renderer=result,
        visible_hands=sum(len(view["visible_hands"]) for entry in entries for view in entry["views"].values()))))


def render_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    packaged = json.loads(Path(os.environ["ANYHAND_RENDER_DEFAULTS"]).read_text()) if os.environ.get("ANYHAND_RENDER_DEFAULTS") else {}
    camera = parser.add_mutually_exclusive_group(required=not packaged)
    camera.add_argument("--calibration", type=Path, help="Calibrated stereo pair in left-to-right order")
    camera.add_argument("--rig-json", type=Path, help="Four-camera rig in Handspotter slot order")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--viewport", type=int, help=argparse.SUPPRESS)
    preview.add_argument("--capture-sample", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--studio-resolve-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job", type=Path, help="Load a Studio render job (JSON). Relative inputs resolve beside the job.")
    parser.add_argument("--native-soma", action="store_true", help="Use the SOMA-X worker and assets bundled with Studio")
    parser.add_argument("--pose-edit", help="Override one native pose as index,yaw,pitch,roll (degrees)")
    parser.add_argument("--toolkit", type=Path, help="Override the installed FISHEYE624 camera toolkit")
    parser.add_argument("--topology",type=Path)
    for name, env in (("assets", "ANYHAND_PBR_ASSETS"), ("three", "ANYHAND_THREE_JS")):
        parser.add_argument("--"+name, type=Path, default=os.environ.get(env), required=env not in os.environ)
    source = parser.add_mutually_exclusive_group(required=not packaged)
    source.add_argument("--source",type=Path,nargs="+")
    source.add_argument("--body-bank",type=Path,help="Baked humanoid surfaces and matched scene records")
    source.add_argument("--soma-template",type=Path,help="Rendered scene targets to replay with native SOMA-X")
    parser.add_argument("--soma-worker",help="Executable for the native SOMA-X worker")
    parser.add_argument("--soma-assets",type=Path)
    parser.add_argument("--soma-correctives",type=Path)
    parser.add_argument("--openvr-calibration",type=Path)
    parser.add_argument("--randomize-poses", action="store_true", help="Generate new reaches using template hand landmarks")
    parser.add_argument("--soma-seed",type=int,help="Override the identity and gesture seed derived from the render seed")
    parser.add_argument("--soma-hand-poses", choices=("prior", "mixed", "gestures"),
                        help="Mix named gestures with the pose prior for new native scenes, or select either alone")
    parser.add_argument("--garment-library",type=Path,default=os.environ.get("ANYHAND_GARMENT_LIBRARY", packaged.get("garment_library")),
                        help="Directory containing compiled GarmentCode pattern JSON files")
    wardrobe = parser.add_argument("--soma-wardrobe",
                        help="Garment name, balanced for built-in styles, or garmentcode-balanced for compiled patterns")
    wardrobe.suggestions = ["random", "balanced", *GARMENTS]
    parser.add_argument("--cloth-mode",choices=("drape","fitted"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, required=not packaged, help="Master seed (random on each new packaged run)")
    parser.add_argument("--count", type=int, help="Sample count (1 for the packaged starter, otherwise 24)")
    parser.add_argument("--negative-every", type=int, default=9)
    parser.add_argument("--asset-limit", type=int, default=90)
    parser.add_argument("--face-size", type=int, default=720)
    parser.add_argument("--environment", choices=("random", "all", *ENVIRONMENTS, *FOUNDATION),
                        help="Override the room. Omit to replay recorded rooms or cycle generated rooms.")
    parser.add_argument("--foundation-scenes", type=Path, help="Use the downloaded Foundation-Resources scenes")
    parser.add_argument("--appearance-variation", action=argparse.BooleanOptionalAction,
                        help="Vary skin tones, hand and arm proportions, and garments")
    parser.add_argument("--capture-noise", action=argparse.BooleanOptionalAction,
                        help="Introduce lens haze, airborne debris, and crossing-arm poses")
    parser.add_argument("--slight-blur", action=argparse.BooleanOptionalAction, help="Sample mild optical softness per view")
    parser.add_argument("--lens-decentering", action=argparse.BooleanOptionalAction,
                        help="Perturb each camera's optical center and tangential distortion")
    parser.add_argument("--arm-motion-blur", action=argparse.BooleanOptionalAction, help="Blur moving arms using per-pixel motion vectors")
    parser.add_argument("--camera-motion-blur", action=argparse.BooleanOptionalAction,
                        help="Blur the scene using coherent headset motion vectors")
    parser.add_argument("--wide-reach",action=argparse.BooleanOptionalAction,help="Broaden hand reach, height and distance from the headset")
    parser.add_argument("--hemisphere-reach",action="store_true",help="Sample reaches throughout the torso's front hemisphere")
    parser.add_argument("--head-look",action=argparse.BooleanOptionalAction,help="Vary head direction relative to the torso, mostly independently of the hands")
    parser.add_argument("--tensors-only",action="store_true",
                        help="Keep the native manifest and Handspotter tensors without PNGs or COCO annotations")
    parser.add_argument("--handspotter-render-scale",type=int,choices=(2,4,8),default=8,
                        help="Native Frame rendering pixels per Handspotter input cell")
    parser.add_argument("--output-workers",type=int,default=1,help="Threads for independent pixel preparation and export")
    parser.add_argument("--node", default="node")
    parser.add_argument("--chromium", default="chromium")
    args = parse_render_args(parser, argv)
    use_starter = not (args.source or args.soma_template or args.body_bank)
    args.cloth_mode = "drape"  # Older saved jobs may request fitted garments.
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.count is None:
        args.count = packaged["count"] if use_starter else 24
    if packaged:
        if not (args.rig_json or args.calibration):
            args.rig_json = Path(packaged["rig_json"])
        if args.rig_json == Path(packaged["rig_json"]) and not args.openvr_calibration:
            args.openvr_calibration = Path(packaged["openvr_calibration"])
        if use_starter:
            args.soma_template = Path(packaged["soma_template"])
            args.randomize_poses = True
            args.environment = args.environment or packaged["environment"]
            args.soma_hand_poses = args.soma_hand_poses or packaged["soma_hand_poses"]
            args.foundation_scenes = args.foundation_scenes or Path(packaged["foundation_scenes"])
            args.soma_wardrobe = args.soma_wardrobe or packaged["soma_wardrobe"]
        if args.soma_template and not args.soma_worker:
            args.native_soma = True
    if args.randomize_poses and not args.soma_template:
        parser.error("--randomize-poses requires a SOMA template")
    automatic_soma_seed = args.soma_seed is None
    if automatic_soma_seed:
        args.soma_seed = int(np.random.SeedSequence([args.seed, 24]).generate_state(1)[0]) if args.randomize_poses else 681721
    for key in ("appearance_variation", "capture_noise", "slight_blur", "lens_decentering",
                "arm_motion_blur", "camera_motion_blur", "head_look", "wide_reach"):
        if getattr(args, key) is None:
            setattr(args, key, bool(use_starter and packaged.get(key) and not (key == "wide_reach" and args.hemisphere_reach)))
    if args.wide_reach and args.hemisphere_reach:
        parser.error("Choose one of --wide-reach and --hemisphere-reach")
    args.packaged_foundation = Path(packaged["foundation_scenes"]) if packaged else None
    if args.native_soma:
        args.soma_worker = args.soma_worker or os.environ.get("ANYHAND_SOMA_WORKER")
        args.soma_assets = args.soma_assets or os.environ.get("ANYHAND_SOMA_ASSETS", packaged.get("soma_assets"))
        args.soma_correctives = args.soma_correctives or os.environ.get("ANYHAND_SOMA_CORRECTIVES", packaged.get("soma_correctives"))
        if not args.soma_worker:
            parser.error("--native-soma needs the Studio runtime or --soma-worker")
        if args.soma_correctives:
            args.soma_correctives = Path(args.soma_correctives)
    edit = {}
    if args.pose_edit:
        try:
            index, yaw, pitch, roll = args.pose_edit.split(",")
            index = int(index)
            if not args.soma_worker or not 0 <= index < args.count:
                raise ValueError("pose index must be in the native SOMA-X batch")
            edit[index] = list(map(float, (yaw, pitch, roll)))
        except ValueError as error:
            parser.error(f"Invalid --pose-edit: {error}")
    args.pose_angles = edit
    if bool(args.source)!=bool(args.topology):
        parser.error("--source and --topology must be supplied together")
    if args.studio_resolve_job:
        resolved = {key: getattr(args, key) for key in job_fields(parser) if getattr(args, key) is not None}
        if automatic_soma_seed:
            resolved.pop("soma_seed", None)
        print(json.dumps(resolved, default=str))
        parser.exit()
    return args


class ViewportSession:
    def __init__(self, root):
        self.root = root
        self.serial = 0
        self.geometry = None
        self.worker = None
        self.server = None
        self.store_rooms = {}
        self.capture_args = None

    def worker_for(self, command):
        if self.worker is None or self.command != command or self.worker.process.poll() is not None:
            if self.worker:
                self.worker.__exit__(None, None, None)
            self.worker = Worker(command)
            self.command = command
        return nullcontext(self.worker)

    def geometry_matches(self, args):
        return self.geometry is not None and self.key == self.geometry_key(args)

    @staticmethod
    def geometry_key(args):
        return {k: v for k, v in vars(args).items() if k not in ("environment", "foundation_scenes", "out", "wardrobe_library")}

    def foundation_files(self, root, room):
        path = root.resolve()
        if not path.is_relative_to("/nix/store"):
            return validate_scenes(path, [room])
        key = (path, room)
        if key not in self.store_rooms:
            self.store_rooms[key] = validate_scenes(path, [room])
        return self.store_rooms[key]

    def publish_geometry(self, args, entry, files):
        job = json.loads((self.directory/"job.json").read_text())
        self.job = dict(job, geometry=self.serial)
        self.entry = entry
        self.height = entry["environment"]["height"]
        self.key = self.geometry_key(args)
        self.geometry = self.serial
        return self.publish(args, self.job, files)

    def publish_room(self, args, setup, layout, lighting, files):
        job = dict(self.job, room=dict(layout=layout, lighting=lighting), viewport_setup=asdict(setup))
        return self.publish(args, job, files)

    def publish(self, args, job, files):
        self.capture_args = None
        if self.detector:
            if not args.rig_json:
                raise ValueError("Detector inputs need a four-camera Frame rig")
            self.capture_args = copy(args)
            self.capture_args.tensors_only = True
            self.capture_entry = dict(self.entry, render_setup=job["viewport_setup"])
            if "room" in job:
                self.capture_entry.update(environment=job["room"]["layout"], lighting=job["room"]["lighting"])
            job = dict(job, capture=dict(token=secrets.token_hex(24)))
        save_json(self.directory/"job.json", dict(job, foundation_files=list(files),
                 foundation_key=files if files else None))
        if files:
            (self.directory/"foundation-root").write_text(str(args.foundation_scenes.resolve()))
        if self.server is None or self.server.poll() is not None:
            command = [args.node, str(Path(__file__).with_name("pbr_worker.mjs")), str(self.root),
                       str(args.assets.resolve()), str(args.three.resolve()), "--studio-session"]
            self.server = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.url = self.server.stdout.readline().strip().removeprefix("STUDIO_VIEWPORT ")
            if not self.url.startswith("http://127.0.0.1:"):
                raise RuntimeError("Viewport server did not start")
        return dict(url=self.url, version=self.serial)

    def finish_capture(self, version):
        if version != self.serial or self.capture_args is None:
            raise ValueError("Detector capture has been superseded")
        from render_preview import preview_views
        entry = deepcopy(self.capture_entry)
        data = capture_outputs(self.capture_args, [entry], self.cameras, self.lenses,
                               self.directory, self.streams, [self.capture_args.viewport])
        return dict(views=preview_views(self.streams, entry, data[0]))

    def serve(self):
        for line in sys.stdin:
            request = json.loads(line)
            if "finish" in request:
                try:
                    reply = self.finish_capture(request["finish"])
                except Exception as error:
                    reply = dict(error=str(error))
                print(json.dumps(reply), flush=True)
                continue
            self.detector = request.get("mode") == "detector"
            self.serial += 1
            self.directory = self.root/str(self.serial)
            self.directory.mkdir()
            try:
                recipe = self.root/"request.json"
                save_json(recipe, request["job"])
                diagnostics = StringIO()
                with redirect_stderr(diagnostics):
                    try:
                        args = render_args(["--job", str(recipe), "--viewport", str(request["index"]),
                                            "--out", str(self.root/"unused")])
                    except SystemExit as error:
                        raise ValueError(diagnostics.getvalue().strip() or "Invalid viewport settings") from error
                reply = run(args, self)
            except Exception as error:
                reply = dict(error=str(error))
            print(json.dumps(reply), flush=True)
            for item in self.root.iterdir():
                if item.name.isdecimal() and int(item.name) < self.serial - 1 and int(item.name) != self.geometry:
                    shutil.rmtree(item)

    def close(self):
        if self.server:
            self.server.stdin.close()
            self.server.wait()
        if self.worker:
            self.worker.__exit__(None, None, None)


if __name__ == "__main__":
    if sys.argv[1:] == ["--studio-session"]:
        with tempfile.TemporaryDirectory(prefix="viewport-session-") as root:
            session = ViewportSession(Path(root))
            try:
                session.serve()
            finally:
                session.close()
    else:
        try:
            run(render_args(sys.argv[1:]))
        except (ValueError, OSError, KeyError) as error:
            sys.exit(f"Render failed: {error}")
