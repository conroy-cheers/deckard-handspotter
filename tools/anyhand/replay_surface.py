"""Mesh-space material and open-wrist geometry for calibrated replay."""
import numpy as np


def vertex_normals(vertices, faces):
    points = vertices[faces]
    normal = np.zeros_like(vertices)
    area_normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    for corner in range(3):
        np.add.at(normal, faces[:, corner], area_normal)
    return normal / np.linalg.norm(normal, axis=1, keepdims=True).clip(1e-9)


def wrist_boundary(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, indices, counts = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    boundary = edges[indices[counts == 1]]
    vertices, incoming = np.unique(boundary[:, 1], return_counts=True)
    outgoing, counts = np.unique(boundary[:, 0], return_counts=True)
    if not len(boundary) or not np.array_equal(vertices, outgoing) or np.any(incoming != 1) or np.any(counts != 1):
        raise ValueError("Expected an oriented open wrist boundary")
    return vertices, np.searchsorted(vertices, boundary)


def attach_forearm(vertices, faces, material, boundary, axis):
    """Loft a short, open forearm from the actual wrist boundary."""
    indices, edges = boundary
    ring = vertices[indices]
    center = ring.mean(axis=0)
    # Raise the cross section gradually and flatten it relative to the axis.
    radial = ring - center
    radial -= (radial @ axis)[:, None] * axis
    rings = (center + .023 * axis + 1.04 * radial,
             center + .061 * axis + 1.17 * radial,
             center + .108 * axis + 1.29 * radial)
    ring_indices = np.arange(len(vertices), len(vertices) + 3 * len(indices)).reshape(3, -1)
    strips = []
    for inner, outer in zip((indices, *ring_indices[:-1]), ring_indices):
        p, q = inner[edges].T
        u, v = outer[edges].T
        # Shared edges have opposite orientation to the adjoining hand face.
        strips.extend((np.stack((q, p, u), axis=1), np.stack((q, u, v), axis=1)))
    forearm_faces = np.concatenate(strips)
    detail = material[indices]
    return (np.concatenate((vertices, *rings)), np.concatenate((faces, forearm_faces)),
            np.concatenate((material, detail * .8, detail * .5, detail * .25)))


def skin_detail(rest):
    """World-independent variation tied to the rest surface in metres."""
    coordinates = rest @ np.array([[.83, .38, .4], [.18, -.9, .24], [.4, .12, -.91]])
    broad = np.sin(coordinates[:, 0] * 441 + .8 * np.sin(coordinates[:, 1] * 313))
    fine = np.sin(coordinates[:, 1] * 1297 + np.sin(coordinates[:, 2] * 907))
    return 3.5 * broad + 2.5 * fine


def fit_light(records):
    """Fit ambient and directional terms to paired development pixels."""
    normals = np.concatenate([record[0] for record in records])
    target = np.concatenate([record[1] for record in records])
    features = np.column_stack((np.ones(len(normals)), normals))
    ridge = np.diag([0., .025, .025, .025]) * len(normals)
    weights = np.ones(len(normals))
    for _ in range(5):
        weighted = features.T * weights
        light = np.linalg.solve(weighted @ features + ridge, weighted @ target)
        error = target - features @ light
        cutoff = max(8., 2 * np.median(np.abs(error)))
        weights = cutoff / np.maximum(cutoff, np.abs(error))
    # Match the amount of contrast lost when projecting real textured skin
    # into a smooth lighting basis.
    raw = features @ light
    quantiles = [5, 25, 50, 75, 95]
    knots, index = np.unique(np.percentile(raw, quantiles), return_index=True)
    return light, np.stack((knots, np.percentile(target, quantiles)[index]))
