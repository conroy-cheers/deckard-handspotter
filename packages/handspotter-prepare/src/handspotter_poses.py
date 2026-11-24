"""Generate a small original articulated hand mesh bank for synthetic runs."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np


def mesh():
    # Rasterized palm and five connected digits. The two rows at the wrist
    # taper inward and the only open boundary is the wrist seam.
    footprint = {(x, y) for y in range(2, 9) for x in range(-4, 4)}
    footprint.update((x, y) for y in (0, 1) for x in range(-3, 3))
    footprint.update((x, y) for y in range(4, 8) for x in range(4, 6))
    for x, start, tip in ((2, 9, 16), (0, 9, 17), (-2, 9, 16), (-4, 9, 14)):
        footprint.update((i, y) for i in (x, x + 1) for y in range(start, tip))
    for x in (-2, 0, 2):
        footprint.difference_update((x, y) for y in range(10, 18))
    # Separate the digits above their webbing.
    cells = [(x, y) for x, y in sorted(footprint)]
    vertices, ids, faces, edge_counts = [], {}, [], Counter()

    def vertex(point, side):
        key = (*point, side)
        if key not in ids:
            x, y = point
            ids[key] = len(vertices)
            thick = .009 if y < 8 else .006
            vertices.append((x * .009, y * .010, side * thick))
        return ids[key]

    for x, y in cells:
        polygon = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            edge_counts[a, b] += 1
        for side in (1, -1):
            a, b, c, d = [vertex(point, side) for point in polygon]
            faces.extend(((a, b, c), (a, c, d)) if side == 1 else ((a, c, b), (a, d, c)))
    for a, b in edge_counts:
        if (b, a) in edge_counts or a[1] == b[1] == 0:
            continue
        at, bt, ab, bb = vertex(a, 1), vertex(b, 1), vertex(a, -1), vertex(b, -1)
        faces.extend(((bt, at, ab), (bt, ab, bb)))

    joints = [[0, 0, 0]]
    for x, root, tip in ((.030, .025, .080), (.027, .084, .154),
                         (.009, .086, .164), (-.009, .086, .154), (-.027, .082, .134)):
        joints.extend([x, y, 0] for y in np.linspace(root, tip, 4))
    return np.asarray(vertices), np.asarray(joints), np.asarray(faces, dtype=np.int32)


def pose_bank(directory: Path, seed: int, count: int):
    vertices, joints, faces = mesh()
    rng = np.random.default_rng(seed)
    ids, posed, keys = [], [], []
    for _ in range(count):
        proportions = rng.uniform(.86, 1.14, 3)
        curl = rng.uniform(.02, .65)

        def deform(points):
            points = points * proportions
            points[:, 2] -= curl * np.maximum(points[:, 1] - .07, 0)
            return points.astype(np.float32)

        hand = deform(vertices)
        posed.append(hand)
        keys.append(deform(joints))
        ids.append("procedural-" + hashlib.sha256(hand.astype("<f4").tobytes()).hexdigest()[:20])
    poses, topology = directory / "poses.npz", directory / "topology.npz"
    np.savez(poses, ids=ids, vertices=posed, joints=keys)
    np.savez(topology, faces=faces)
    return poses, topology
