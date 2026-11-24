"""Test posed SOMA-X skin and propose native joint rotations for overlaps."""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from humanoid_geometry import unit


@wp.kernel
def _crossing_faces(
    points: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.int32),
    other: wp.uint64,
    count: wp.array(dtype=wp.int32),
):
    index = 3 * wp.tid()
    a = points[faces[index]]
    b = points[faces[index + 1]]
    c = points[faces[index + 2]]
    low = wp.min(a, wp.min(b, c))
    high = wp.max(a, wp.max(b, c))
    query = wp.mesh_query_aabb(other, low, high)
    face = int(0)
    while wp.mesh_query_next(query, face):
        d = wp.mesh_eval_position(other, face, 1.0, 0.0)
        e = wp.mesh_eval_position(other, face, 0.0, 1.0)
        f = wp.mesh_eval_position(other, face, 0.0, 0.0)
        if wp.intersect_tri_tri(wp.vec3d(a), wp.vec3d(b), wp.vec3d(c),
                               wp.vec3d(d), wp.vec3d(e), wp.vec3d(f)):
            wp.atomic_add(count, 0, 1)
            break


def _subtree(parents, root):
    selected = np.zeros(len(parents), dtype=bool)
    selected[root] = True
    for joint in range(root + 1, len(parents)):
        selected[joint] = selected[parents[joint]]
    return selected


class SkinPatch:
    def __init__(self, faces, mask):
        self.vertices, indices = np.unique(faces[mask[faces].all(axis=1)], return_inverse=True)
        self.faces = indices.reshape(-1, 3)
        self.indices = wp.array(indices.ravel(), dtype=wp.int32, device="cpu")

    def obstacle(self, points):
        return wp.Mesh(points=wp.array(points, dtype=wp.vec3, device="cpu"), indices=self.indices)

    def crossings(self, points, low, high, obstacle):
        if np.any(points.max(axis=0) < low) or np.any(high < points.min(axis=0)):
            return 0
        count = wp.zeros(1, dtype=wp.int32, device="cpu")
        wp.launch(_crossing_faces, dim=len(self.faces), device="cpu",
                  inputs=[wp.array(points, dtype=wp.vec3, device="cpu"), self.indices, obstacle.id],
                  outputs=[count])
        return int(count.numpy()[0])


class ArmCollision:
    """Test the skin driven by the shoulder and head subtrees."""

    def __init__(self, layer):
        names = layer.target_joint_names
        parents = layer.joint_parent_ids.numpy()
        weights = layer.skinning_weights.numpy()
        faces = layer.faces.numpy()
        self.arms = []
        self.head_arms = []
        self.lower_weights = []
        for side in ("Left", "Right"):
            arm = weights[:, _subtree(parents, names.index(side+"Arm"))].sum(axis=1)
            # The collar and its shared shoulder seam remain part of the torso.
            patch = SkinPatch(faces, arm > .5)
            self.arms.append(patch)
            if side+"ForeArm" in names:
                lower = weights[:, _subtree(parents, names.index(side+"ForeArm"))].sum(axis=1)
                distal = SkinPatch(faces, lower > .5)
                self.head_arms.append(distal)
                self.lower_weights.append(lower[distal.vertices, None])
        self.vertices = [patch.vertices for patch in self.arms]
        self.faces = [patch.faces for patch in self.arms]
        if "Head" in names:
            head = weights[:, _subtree(parents, names.index("Head"))].sum(axis=1)
            self.head = SkinPatch(faces, head > .6)

    def crossing_faces(self, left, right):
        """Count left-arm faces that intersect at least one right-arm face."""
        low, high = right.min(axis=0), right.max(axis=0)
        if np.any(left.max(axis=0) < low) or np.any(high < left.min(axis=0)):
            return 0
        return self.arms[0].crossings(left, low, high, self.arms[1].obstacle(right))

    def head_contacts(self, vertices):
        head = vertices[self.head.vertices]
        low, high = head.min(axis=0), head.max(axis=0)
        arms = [vertices[patch.vertices] for patch in self.head_arms]
        if all(np.any(points.max(axis=0) < low) or np.any(high < points.min(axis=0)) for points in arms):
            return (0, 0)
        obstacle = self.head.obstacle(head)
        return tuple(patch.crossings(points, low, high, obstacle)
                     for patch, points in zip(self.head_arms, arms, strict=True))

    def head_adjustments(self, vertices, side, joints):
        """Rank hinge and shoulder motions by approximate wrist displacement."""
        shoulder, elbow, wrist = joints
        lower = unit(wrist-elbow)
        upper = unit(elbow-shoulder)
        bend = np.cross(upper, lower)
        flex = np.degrees(np.arctan2(np.linalg.norm(bend), upper @ lower))
        candidates = []
        if np.linalg.norm(bend) > 1e-6:
            for degrees in np.arange(-50., 51., 2.):
                if degrees and 14 <= flex+degrees <= 140:
                    candidates.append((np.linalg.norm(wrist-elbow)*abs(degrees), 1, unit(bend)*degrees))
        reach = unit(wrist-shoulder)
        first = unit(np.cross(reach, np.eye(3)[np.argmin(abs(reach))]))
        second = np.cross(reach, first)
        headings = np.linspace(0., 2*np.pi, 16, endpoint=False)
        axes = np.cos(headings)[:, None]*first + np.sin(headings)[:, None]*second
        for degrees in np.arange(2., 57., 2.):
            candidates.extend((np.linalg.norm(wrist-shoulder)*degrees, 0, axis*degrees) for axis in axes)

        patch = self.head_arms[side]
        points = vertices[patch.vertices]
        head = vertices[self.head.vertices]
        low, high = head.min(axis=0), head.max(axis=0)
        obstacle = self.head.obstacle(head)
        for _, pivot, rotvec in sorted(candidates, key=lambda item: item[0]):
            delta = Rotation.from_rotvec(np.deg2rad(rotvec)).as_matrix()
            displacement = (points-joints[pivot])@(delta.T-np.eye(3))
            if pivot == 1:
                displacement *= self.lower_weights[side]
            if not patch.crossings(points+displacement, low, high, obstacle):
                yield pivot, delta

    def adjustment(self, vertices, shoulders, basis):
        """Propose a small shoulder rotation that separates the posed arm meshes."""
        left, right = [vertices[indices] for indices in self.vertices]
        if not self.crossing_faces(left, right):
            return None
        centered = [points - center for points, center in zip((left, right), shoulders, strict=True)]
        # Search both over/under orders and lateral separation. Keep the elbow,
        # forearm and finger articulation together during each trial.
        headings = np.arange(8) * np.pi / 4
        axes = np.cos(headings)[:, None] * basis[:, 0] + np.sin(headings)[:, None] * basis[:, 1]
        for effort in range(2, 51, 2):
            for left_fraction, right_fraction in ((.707, .707), (1., 0.), (0., 1.)):
                # Compare candidates at similar total squared shoulder rotation.
                angles = np.deg2rad(effort) * np.array([left_fraction, -right_fraction])
                for axis in axes:
                    matrices = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
                    moved = [points @ matrix.T + center for points, matrix, center
                             in zip(centered, matrices, shoulders, strict=True)]
                    if not self.crossing_faces(*moved):
                        return matrices
        raise ValueError("No separating shoulder rotations found")
