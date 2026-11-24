"""Place the tracking rig and labels relative to a torso-fixed reference."""
import numpy as np


NECK_PIVOT = np.array([0., -.135, .135])


def front_direction(rng):
    forward = rng.uniform(0., 1.)
    azimuth = rng.uniform(-np.pi, np.pi)
    tangent = np.sqrt(1. - forward**2)
    return np.array([tangent*np.cos(azimuth), tangent*np.sin(azimuth), -forward])


def look_angles(direction):
    x, y, z = direction
    return np.degrees([np.arctan2(x, -z), np.arctan2(y, np.hypot(x, z))])


def sample_head_pose(rng, hands):
    from scipy.spatial.transform import Rotation

    if hands and rng.random() < .20:
        hand = hands[rng.choice(tuple(hands))]
        target = np.asarray(hand["keypoints_pair_m"])[[5, 9, 13, 17]].mean(axis=0)
        yaw, pitch = np.clip(look_angles(target) + rng.normal(0., [11., 9.]), [-85., -76.], [85., 70.])
        mode = "toward_hand"
    else:
        while True:
            yaw, pitch = look_angles(front_direction(rng))
            if abs(yaw) <= 85. and -76. <= pitch <= 70.:
                break
        mode = "independent"
    roll = float(np.clip(rng.normal(0., 8.), -22., 22.))
    rotation = Rotation.from_euler("YXZ", [-yaw, pitch, roll], degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = NECK_PIVOT - rotation @ NECK_PIVOT
    return dict(mode=mode, yaw_pitch_roll_deg=[float(yaw), float(pitch), roll],
                pivot_reference_m=NECK_PIVOT.tolist(), reference_from_pair=matrix.tolist())


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def transform_hands(hands, matrix):
    result = {}
    for side, hand in hands.items():
        arm = hand["arm"]
        endpoints = {key: transform_points(value, matrix).tolist() for key, value in arm.items()
                     if key in ("shoulder_pair_m", "elbow_pair_m", "wrist_pair_m")}
        result[side] = dict(hand, arm=dict(arm, **endpoints),
                            keypoints_pair_m=transform_points(hand["keypoints_pair_m"], matrix).tolist())
    return result


def in_reference_scene(entry):
    """Recover torso-relative targets from a rendered dataset for humanoid baking."""
    if "head_pose" not in entry:
        return entry
    matrix = np.asarray(entry["head_pose"]["reference_from_pair"])
    return dict(entry, hands=transform_hands(entry["hands"], matrix))


def baked_head_weights(vertices, faces, labels, pivot):
    """Identify the connected head and taper the rotation across the neck."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    weights = np.zeros(len(vertices), dtype=np.float64)
    low = pivot[1] - .055
    eligible = vertices[:, 1] > low
    eligible[faces[labels != 3].ravel()] = False
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return weights
    seed_target = np.array([0., .040, .150])
    seed = candidates[np.argmin(np.linalg.norm(vertices[candidates]-seed_target, axis=1))]
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    a, b = edges[eligible[edges].all(axis=1)].T
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(len(vertices), len(vertices))).tocsr()
    _, component = connected_components(graph, directed=False)
    head = candidates[component[candidates] == component[seed]]
    weight = np.clip((vertices[head, 1] - low) / .105, 0., 1.)
    weights[head] = weight**2 * (3. - 2.*weight)
    return weights


def turn_baked_head(vertices, faces, labels, pose):
    """Move the head with the rig and blend the turn through the connected neck."""
    from scipy.spatial.transform import Rotation

    pivot = np.asarray(pose["pivot_reference_m"])
    weights = baked_head_weights(vertices, faces, labels, pivot)
    head = weights > 0.
    angle = Rotation.from_matrix(np.asarray(pose["reference_from_pair"])[:3, :3]).as_rotvec()
    result = vertices.copy()
    if head.any():
        result[head] = Rotation.from_rotvec(weights[head, None]*angle).apply(vertices[head]-pivot)+pivot
    return result
