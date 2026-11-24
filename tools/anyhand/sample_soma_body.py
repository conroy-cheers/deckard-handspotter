"""Retarget synthetic Frame reaches to the native SOMA-X body, fingers and correctives."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from head_pose import in_reference_scene
from humanoid_geometry import (
    SIDES, body_basis, hand_record, pair_points, retarget_arms, scene_record, turn_joint, write_bank,
)
from sample_soma_hand import absolute_poses, landmarks
from soma_arm_collision import ArmCollision
from soma_arm_skinning import SomaArmSkinning


def bake(args):
    from soma import SOMALayer

    if args.start<0 or min(args.count,args.batch,args.threads)<1:
        raise ValueError("Expected a nonnegative start and positive count, batch and thread sizes")
    torch.set_num_threads(args.threads)
    layer = SOMALayer(device="cpu", lod="mid", identity_model_type="soma",
                      data_root=args.assets, correctives_model_path=args.correctives)
    source = json.loads(args.source.read_text())["records"][args.start:args.start+args.count]
    if len(source)!=args.count:
        raise ValueError("The scene dataset does not contain the requested range")
    source = [in_reference_scene(row) for row in source]
    names = layer.public_joint_names
    parents = layer.output_joint_parent_ids.numpy()[1:]
    arms = [[names.index(side.title()+joint) for joint in ("Arm","ForeArm","Hand")]
            for side in SIDES]
    cloth_joints = [names.index(name) for name in ("Hips","Spine1","Chest","Neck1")]
    neck = [names.index(name) for name in ("LeftEye","RightEye")]
    chest = [names.index(name) for name in ("Spine2","Chest","LeftShoulder","RightShoulder")]
    points = [landmarks(names,side) for side in SIDES]
    arm_skinning = SomaArmSkinning(layer,arms,points) if args.arm_rig=="anatomical" else None
    arm_collision = ArmCollision(layer)
    shoulders = np.array([arm[0] for arm in arms])
    faces = layer.faces.numpy()
    labels = np.full(len(faces),3,np.uint8)
    rngs = [np.random.default_rng(np.random.SeedSequence([args.seed,i]))
            for i in range(args.start,args.start+args.count)]
    with np.load(args.assets/"SOMAHand.npz",allow_pickle=False) as prior:
        fingers = []
        for side in SIDES:
            labels[np.isin(faces,prior[f"{side}_vert_ids"]).all(axis=1)] = SIDES.index(side)+1
            ids = [names.index(n)-1 for n in prior[f"{side}_pose_joint_names"]]
            poses = torch.cat([absolute_poses(rng,1,prior,side)[:,1:] for rng in rngs])
            fingers.append((ids,poses.numpy()))
    if any((labels==part).sum()<100 for part in (1,2)):
        raise ValueError("SOMA hand and body surface indexing disagree")

    def evaluate(local, fk_only=True):
        return layer.pose(torch.as_tensor(local,dtype=torch.float32), pose2rot=False,
                          absolute_pose=True, fk_only=fk_only,
                          apply_correctives=not fk_only)

    def deform(local, fitted, output):
        if arm_skinning:
            return arm_skinning.deform(torch.as_tensor(local,dtype=torch.float32),output,fitted)
        return output.vertices.numpy(),output.transforms.numpy(),None

    args.out.mkdir(parents=True)
    records = []
    for start in range(0,args.count,args.batch):
        stop = min(start+args.batch,args.count)
        batch = rngs[start:stop]
        rows = source[start:stop]
        identity = np.stack([rng.normal(0,.68,layer.num_shape_components).clip(-2,2)
                             for rng in batch]).astype(np.float32)
        layer.prepare_identity(torch.from_numpy(identity))
        neutral = layer.pose(torch.zeros(len(batch),len(names)-1,3),
                             fk_only=True,apply_correctives=False).transforms.numpy()
        fitted = arm_skinning.fitted_arms() if arm_skinning else None
        local = neutral[:,parents,:3,:3].swapaxes(-2,-1)@neutral[:,1:,:3,:3]
        rest = neutral[:,:,:3,3]
        up = np.array([0.,1.,0.])
        bases = [body_basis(j[arms[0][0]], j[arms[1][0]], up) for j in rest]
        for i,rng in enumerate(batch):
            local[i,np.asarray(chest)-1] @= Rotation.from_euler(
                "xyz",rng.normal(0,[[3,4,2],[4,5,3],[3,5,5],[3,5,5]]),degrees=True).as_matrix()
        for ids,poses in fingers:
            local[:,ids] = poses[start:stop]

        world = evaluate(local).transforms.numpy()
        eyes = world[:,:,:3,3][:,neck].mean(axis=1)
        retarget_arms(local,parents,lambda rotations: evaluate(rotations).transforms.numpy(),
                      rows,arms,points,bases,eyes,bind_poses=None if arm_skinning else neutral)

        requested_roll = arm_skinning.constrain(local,evaluate(local).transforms,fitted) if arm_skinning else None
        output = evaluate(local,fk_only=False)
        vertices,posed,diagnostics = deform(local,fitted,output)
        original_shoulders = local[:,shoulders-1].copy()
        pending = list(range(len(batch)))
        # Verify the re-skinned surface after each shoulder adjustment.
        for attempt in range(4):
            moved = []
            for i in pending:
                try:
                    adjustment = arm_collision.adjustment(vertices[i],posed[i,shoulders,:3,3],bases[i])
                except ValueError as exc:
                    raise ValueError(f"Cannot separate SOMA-X arms in pose {rows[i]['id']}") from exc
                if adjustment is not None:
                    for shoulder,delta in zip(shoulders,adjustment,strict=True):
                        turn_joint(local[i],posed[i],parents,shoulder,delta)
                    moved.append(i)
            if not moved:
                break
            if attempt == 3:
                ids = [rows[i]["id"] for i in moved]
                raise ValueError(f"SOMA-X arms still intersect after reskinning poses {ids}")
            pending = moved
            vertices,posed,diagnostics = deform(local,fitted,evaluate(local,fk_only=False))
        shoulder_delta = local[:,shoulders-1] @ original_shoulders.swapaxes(-2,-1)
        separation = np.degrees(Rotation.from_matrix(shoulder_delta.reshape(-1,3,3)).magnitude()).reshape(-1,2)
        if args.cloth_reference:
            rest_vertices = layer.pose(torch.zeros(len(batch),len(names)-1,3),
                                       apply_correctives=False).vertices.numpy()
        for i,(row,basis) in enumerate(zip(rows,bases,strict=True)):
            joints = pair_points(posed[i,:,:3,3],basis,eyes[i])
            hands = {side:hand_record(row,"soma-body",side,joints[indices],joints[s],joints[e])
                     for side,(s,e,_),indices in zip(SIDES,arms,points,strict=True)}
            if arm_skinning:
                for side,roll in zip(SIDES,requested_roll[i],strict=True):
                    hands[side]["arm"].update(diagnostics[i][side])
                    hands[side]["arm"]["requested_forearm_rotation_deg"] = roll
            for side,angle in zip(SIDES,separation[i],strict=True):
                hands[side]["arm"]["shoulder_separation_deg"] = float(angle)
            records.append(scene_record(row,"soma-body",hands,pose_correctives=True,arm_rig=args.arm_rig))
            surfaces = dict(skin_vertices=pair_points(vertices[i],basis,eyes[i]).astype(np.float32),
                            skin_faces=faces,skin_labels=labels)
            if args.cloth_reference:
                surfaces.update(
                    rest_vertices=pair_points(rest_vertices[i],basis,eyes[i]).astype(np.float32),
                    rest_cloth_joints=pair_points(rest[i,cloth_joints],basis,eyes[i]).astype(np.float32),
                    rest_arm_joints=pair_points(rest[i,arms],basis,eyes[i]).astype(np.float32),
                )
            np.savez_compressed(args.out/f"{start+i:05d}.npz",**surfaces)
        print(f"Baked {stop}/{args.count} SOMA-X bodies",flush=True)
    checksums = {}
    for path in (args.correctives,*(args.assets/name for name in (
            "SOMA_neutral.npz","SOMAHand.npz","SOMA_template_rig.usda","SOMA_procedural_transforms.json"))):
        with path.open("rb") as asset:
            checksums[path.name] = hashlib.file_digest(asset,"sha256").hexdigest()
    write_bank(args.out,args.source,dict(library="SOMA-X",arm_rig=args.arm_rig,
                                        asset_sha256=checksums),records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets","correctives","source","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--count",type=int,required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--batch",type=int,default=12)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--arm-rig",choices=("anatomical","stock"),default="stock",
                   help="Native SOMA-X arm skinning, with the anatomical trial available for replay")
    p.add_argument("--cloth-reference",action="store_true",
                   help="Save same-topology reference vertices and joint positions for garment fitting")
    with torch.no_grad():
        bake(p.parse_args())
