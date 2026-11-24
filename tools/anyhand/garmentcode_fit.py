"""Pose-fit compiled, sewn GarmentCode panel meshes to a SOMA-X body."""
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.transform import Rotation

from humanoid_geometry import SIDES


def garment_library(path):
    if path is None:
        return {}
    return {file.stem: file for file in sorted(path.glob("*.json")) if file.is_file()}


def rotation_between(source, target):
    cross = np.cross(source, target)
    length = np.linalg.norm(cross)
    dot = np.dot(source, target)
    if length > 1e-8:
        return Rotation.from_rotvec(cross/length * np.arctan2(length, dot))
    if dot > 0:
        return Rotation.identity()
    axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
    return Rotation.from_rotvec(np.pi * axis/np.linalg.norm(axis))


def panels_front(vertices, faces, panel_ids, selected):
    """Identify the front panel faces from BoxMesh's front/back placement."""
    depth = vertices[faces].mean(axis=1)[:, 2]
    assignment = panel_ids[selected]
    counts = np.bincount(assignment, minlength=panel_ids.max()+1)
    depth = np.bincount(assignment, weights=depth[selected], minlength=len(counts))
    means = depth / np.maximum(counts, 1)
    limits = means[counts > 0]
    return selected & (means[panel_ids] < (limits.min()+limits.max())/2)


def panel_direction(faces, selected, front, count):
    """Use coincident sewn vertices for a continuous front-to-back phase."""
    fore = np.bincount(faces[front].ravel(), minlength=count)
    aft = np.bincount(faces[selected & ~front].ravel(), minlength=count)
    return (aft > 0).astype(float) - (fore > 0)


def neckline_vertices(vertices, faces, arm_weights, attached=None):
    triangle_edges = faces[:, (0, 1, 1, 2, 2, 0)].reshape(-1, 2)
    edges, counts = np.unique(np.sort(triangle_edges, axis=1), axis=0, return_counts=True)
    edges = edges[counts == 1]
    a, b = edges.T
    labels = connected_components(
        csr_matrix((np.ones(len(edges)), (a, b)), shape=(len(vertices), len(vertices))),
        directed=False, return_labels=True,
    )[1]
    openings = np.unique(labels[edges])
    torso_openings = [labels == label for label in openings
                      if arm_weights[labels == label].mean() < .15]
    if not torso_openings:
        raise ValueError("The upper-body pattern has no neckline opening")
    if attached is None:
        attached = np.zeros(len(vertices), dtype=bool)
    return max(torso_openings, key=lambda mask: ((mask & attached).sum(), vertices[mask, 1].mean()))


def panel_vertices(faces, selected, count):
    used = np.zeros(count, dtype=bool)
    used[faces[selected]] = True
    return used


def place_neckwear(position, vertices, used, attachment, edges, lengths, hood):
    """Fit the panels sewn onto the bodice's neckline."""
    if not attachment.any():
        raise ValueError("Collar or hood panels have no sewn attachment to the bodice")
    a, b = edges.T
    shared = used[a] & used[b]
    size = len(vertices)
    material = csr_matrix((lengths[shared], (a[shared], b[shared])), shape=(size, size))
    distance, _, nearest = dijkstra(material, directed=False, indices=np.flatnonzero(attachment),
                                    return_predecessors=True, min_only=True)
    free = used & ~attachment
    span = distance[free]
    if not np.isfinite(span).all():
        raise ValueError("Collar or hood has a component disconnected from the neckline")
    root = position[attachment].mean(axis=0)
    base = position[nearest[free]]
    if hood:
        # Rotate the assembled hood down towards the upper back. Blend from
        # the fitted neckline, leaving the seam vertices shared with the shirt.
        donor = vertices[attachment].mean(axis=0)
        up = np.maximum(vertices[free, 1]-donor[1], 0.)
        target = root + np.column_stack((
            .65*(vertices[free, 0]-donor[0]),
            -.04-.66*up,
            .10+.55*up+.22*(vertices[free, 2]-donor[2]),
        ))
        blend = np.clip(span/.14, 0., 1.)[:, None]
        position[free] = base+(target-base)*blend
    else:
        # Place the free edge of the collar by its actual distance through
        # the fabric, independent of the offsets between the BoxMesh panels.
        outward = base-root
        outward[:, 1] = 0.
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), .001)
        position[free] = base+span[:, None]*(.50*outward+[0., .85, 0.])


def wrap_torso(vertices, bank, front_back, torso_vertices, neck, width, vertical, collar, scale):
    """Wrap the pattern around torso sections, with flat side seams meeting at the sides."""
    rest, skin = (bank[name] for name in ("rest_vertices", "skin_vertices"))
    shoulder_y = bank["rest_arm_joints"][:, 0, 1].mean()
    # Select the central body in the rest pose, so bent arms never get
    # mistaken for chest or back geometry while fitting the panels.
    torso = skin[np.abs(rest[:, 0]-neck[0]) < width*.60]
    y = neck[1]+.007+(vertices[:, 1]-vertices[collar, 1].max())*vertical
    levels = np.arange(y.min()-.02, max(y.max(), shoulder_y)+.024, .024)
    bounds = []
    sections = []
    for level in levels:
        layer = torso[np.abs(torso[:, 1]-min(level, shoulder_y-.03)) < .02][:, (0, 2)]
        sections.append(layer)
        if len(layer):
            bounds.append(np.quantile(layer, (.01, .99), axis=0).ravel())
        else:
            bounds.append(np.full(4, np.nan))
    bounds = np.asarray(bounds)
    valid = np.isfinite(bounds).all(axis=1)
    limits = np.column_stack([np.interp(y, levels[valid], values[valid]) for values in bounds.T])
    center = (limits[:, :2]+limits[:, 2:])/2
    radius = (limits[:, 2:]-limits[:, :2])/2 + [.016, .024]
    # A flat front panel spans a half-turn. The same fabric width is not a
    # straight-line distance across the chest.
    panel_extent = np.quantile(np.abs(vertices[torso_vertices, 0]), .97)
    phase = np.clip(vertices[:, 0]/panel_extent, -1, 1) * np.pi/2
    radius[:, 0] = np.maximum(radius[:, 0], panel_extent*scale*.69)
    # A bounding ellipse can still cut through the ribs, shoulder blades or
    # bust in its diagonal sectors. Enclose the torso in those sectors too.
    direction = np.column_stack((np.sin(phase), front_back*np.cos(phase)))
    index = np.clip(np.searchsorted(levels, y), 0, len(levels)-1)
    for section_id, layer in enumerate(sections):
        mask = index == section_id
        if not len(layer) or not mask.any():
            continue
        layer_center = center[mask].mean(axis=0)
        unit = (layer-layer_center)/radius[mask].mean(axis=0)
        magnitude = np.linalg.norm(unit, axis=1)
        support = (direction[mask] @ (unit/np.maximum(magnitude[:, None], .001)).T) > .94
        allowance = np.max(np.where(support, magnitude[None, :]+.06, 1.), axis=1)
        radius[mask] *= np.maximum(allowance, 1.)[:, None]
    x = center[:, 0]+radius[:, 0]*np.sin(phase)
    z = center[:, 1]+radius[:, 1]*front_back*np.cos(phase)
    return np.column_stack((x, y, z))


def wrap_sleeve(vertices, faces, selected, front, seam, start, bend, end, direction, scale, arm_length):
    """Turn the two loose BoxMesh panels into a round sleeve along the arm."""
    used = np.unique(faces[selected])
    sleeve = vertices[used]
    local_front = panel_direction(faces, selected, front, len(vertices))[used]
    across = np.cross([0., 0., 1.], direction)
    center = vertices[seam].mean(axis=0)
    relative = sleeve-center
    axial = relative@direction
    transverse = relative@across

    # Estimate material width from the cross-section of the flat placement.
    # The depth of the separated BoxMesh panels is an assembly clearance.
    # It must never determine how far cloth sits from the arm.
    keys = np.linspace(axial.min(), axial.max(), max(3, int(np.ptp(axial)/.018)))
    span = max(.028, 2*(keys[-1]-keys[0])/(len(keys)-1))
    bounds = np.array([(transverse[np.abs(axial-key) <= span].min(),
                        transverse[np.abs(axial-key) <= span].max()) for key in keys])
    lo, hi = [np.interp(axial, keys, values) for values in bounds.T]
    fraction = np.clip((transverse-lo)/np.maximum(hi-lo, .005), 0, 1)
    # Both stitched longitudinal edges land on the same tube generatrix.
    longitudinal = np.zeros(len(vertices), dtype=bool)
    shared = np.intersect1d(faces[front], faces[selected & ~front])
    longitudinal[shared] = True
    sewn = longitudinal[used]
    fraction[sewn] = np.rint(fraction[sewn])
    phase = fraction*np.pi
    radius = np.maximum((hi-lo)*scale/np.pi, .046)

    upper, lower = np.linalg.norm(bend-start), np.linalg.norm(end-bend)
    upper_axis, lower_axis = (bend-start)/upper, (end-bend)/lower
    along = axial*((upper+lower)/arm_length)
    upper_rot = rotation_between(direction, upper_axis)
    elbow_rotation = rotation_between(upper_axis, lower_axis).as_rotvec()
    blend = np.clip((along-upper+.07)/.14, 0., 1.)
    bend_rot = Rotation.from_rotvec(blend[:, None] * elbow_rotation)
    axis_points = start + np.minimum(along, upper)[:, None]*upper_axis
    axis_points += np.maximum(0., along-upper)[:, None]*lower_axis
    radial = radius[:, None]*(-np.cos(phase)[:, None]*across +
                              (local_front*np.sin(phase))[:, None]*[0., 0., 1.])
    return used, axis_points+bend_rot.apply(upper_rot.apply(radial)), along


def join_armholes(torso, sleeve, attached, collar, adjacency):
    # The seams remain welded. Spread their motion into neighbouring torso
    # triangles so the set-in sleeve does not pull only on the seam vertices.
    offset = np.zeros_like(torso)
    offset[attached] = sleeve[attached]-torso[attached]
    free = ~(attached | collar)
    for _ in range(16):
        offset[free] = .90*(adjacency @ offset)[free]
    return torso+offset


def clear_initial_skin_overlap(position, bank, adjacency):
    """Resolve embedded patches before cloth settling, including pinned points."""
    import trimesh

    skin = trimesh.Trimesh(bank["skin_vertices"], bank["skin_faces"], process=False)
    active = np.arange(len(position))
    correction = np.zeros_like(position)
    for _ in range(3):
        closest, _, faces = trimesh.proximity.closest_point(skin, position[active])
        outward = skin.face_normals[faces]
        clearance = np.einsum("ij,ij->i", position[active]-closest, outward)
        correction.fill(0)
        correction[active] = np.clip(.008-clearance, 0., .06)[:, None]*outward
        if not correction.any():
            break
        movement = correction + .35*(adjacency @ correction)
        position += movement
        # A vertex whose clearance was adequate remains resolved until it moves.
        active = np.flatnonzero(movement.any(axis=1) | correction.any(axis=1))


def fit_compiled_garment(path, bank, record):
    spec = json.loads(path.read_text())
    if spec.get("version") != 1:
        raise ValueError(f"Unsupported GarmentCode library format in {path}")
    vertices = np.asarray(spec["vertices"], dtype=np.float64) * [-1, 1, -1]
    faces = np.asarray(spec["faces"], dtype=np.int32)
    regions = np.asarray(spec["arm_regions"], dtype=np.float64)
    panel_ids = np.asarray(spec["panels"], dtype=np.int32)
    edges = np.asarray(spec["edges"], dtype=np.int32)
    lengths = np.asarray(spec["rest_lengths"], dtype=np.float64)
    body = spec["measurements"]
    hips, _, _, neck = bank["rest_cloth_joints"]
    shoulder, _, _ = bank["rest_arm_joints"].transpose(1, 0, 2)
    width = float(np.linalg.norm(shoulder[1]-shoulder[0]))
    scale = width/(float(body["shoulder_w"])/100)
    vertical = np.clip((neck[1]-hips[1])/(body["waist_line"]/100+.15), .65, 1.45)
    total = regions.sum(axis=1)
    panel_names = spec["panel_names"]
    arm_panels = spec.get("arm_panels")
    if arm_panels is None:
        arm_panels = [[name for name in panel_names if "sleeve" in name.lower() and side in name.lower()]
                      for side in SIDES]
    sleeve_panels = np.array([np.isin(panel_names, names) for names in arm_panels])
    hood_faces = np.array(["hood" in name.lower() for name in panel_names])[panel_ids]
    band_faces = np.array(["collar" in name.lower() or "lapel" in name.lower()
                           for name in panel_names])[panel_ids]
    neckwear_faces = hood_faces | band_faces
    torso_faces = ~(sleeve_panels[:, panel_ids].any(axis=0) | neckwear_faces)
    torso_used = panel_vertices(faces, torso_faces, len(vertices))
    neckwear_used = panel_vertices(faces, neckwear_faces, len(vertices))
    collar = neckline_vertices(vertices, faces[~neckwear_faces], total, torso_used & neckwear_used)
    front = panels_front(vertices, faces, panel_ids, torso_faces)
    torso_direction = panel_direction(faces, torso_faces, front, len(vertices))
    torso = wrap_torso(vertices, bank, torso_direction, torso_used & (total < .01),
                      neck, width, vertical, collar, scale)
    a, b = np.concatenate((edges, edges[:, ::-1])).T
    weight = 1/np.maximum(np.linalg.norm(vertices[a]-vertices[b], axis=1), .004)
    weight /= np.bincount(a, weights=weight, minlength=len(vertices))[a]
    adjacency = csr_matrix((weight, (a, b)), shape=(len(vertices), len(vertices)))
    sleeves = torso.copy()
    sleeve_support = np.zeros(len(vertices))
    extension = 0.
    angle = np.deg2rad(body["arm_pose_angle"])
    for side, sign, weight, panels in zip(SIDES, (-1, 1), regions.T, sleeve_panels, strict=True):
        selected = panels[panel_ids]
        if not selected.any():
            continue
        seam = (weight > 0) & (weight < 1)
        direction = np.array([sign*np.cos(angle), -np.sin(angle), 0.])
        if not seam.any():
            mask = weight > 0
            onset = vertices[mask]@direction
            seam[mask] = onset < np.quantile(onset, .15)
        front = panels_front(vertices, faces, panel_ids, selected)
        arm = record["hands"][side]["arm"]
        start, bend, end = (np.asarray(arm[f"{name}_pair_m"]) for name in ("shoulder", "elbow", "wrist"))
        used, sleeve, along = wrap_sleeve(vertices, faces, selected, front, seam, start, bend, end,
                                          direction, scale, body["arm_length"]/100)
        sleeves[used] = sleeve
        upper = np.linalg.norm(bend-start)
        sleeve_support[used] = .12*np.clip(1-along/upper, 0, 1)
        extension = max(extension, along.max()-upper-np.linalg.norm(end-bend))
    positioned = join_armholes(torso, sleeves, total > 0, collar, adjacency)
    for selected, is_hood in ((hood_faces, True), (band_faces, False)):
        if selected.any():
            used = panel_vertices(faces, selected, len(vertices))
            place_neckwear(positioned, vertices, used, used & torso_used, edges, lengths, is_hood)
    collar_top = vertices[collar, 1].max()
    # Support the strips crossing the shoulders and the back of the neck.
    # The front neckline and the sleeve below the shoulder remain free to drape.
    shoulder_band = torso_used & (np.abs(vertices[:, 0]) < body["shoulder_w"]/200) & (total < .5)
    pin = np.where(shoulder_band, .72*np.clip((vertices[:, 1]-collar_top+.048)/.048, 0, 1), 0.)
    pin = np.maximum(pin, sleeve_support)
    pin[collar] = np.maximum(pin[collar], .18)
    pin[collar & (vertices[:, 1] >= collar_top-.025)] = 1.
    if not np.isfinite(positioned).all():
        raise ValueError(f"Non-finite pose fitting for {path}")
    clear_initial_skin_overlap(positioned, bank, adjacency)
    return dict(position=positioned.astype(np.float32).ravel().tolist(), faces=faces.ravel().tolist(),
                pin=pin.astype(np.float32).tolist(), edges=spec["edges"], rest_lengths=spec["rest_lengths"],
                panel_uv=spec["panel_uv"], uv_faces=spec["uv_faces"],
                pattern_source=spec["source"], stiffness=.23, roughness=.78,
                pattern=0, coverage_mm=1000*extension)
