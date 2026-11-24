"""Sample native SOMA hand identities and finger poses into a PBR pose bank."""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from humanoid_geometry import palm_frame, unit


FINGERS = ("Index","Middle","Ring","Pinky")


def landmarks(names, side="right"):
    # The four non-thumb fingers have an extra metacarpal joint before MCP.
    names = list(names)
    prefix = side.title()+"Hand"
    labels = [prefix,*(prefix+"Thumb"+k for k in ("1","2","3","End"))]
    labels += [prefix+finger+k for finger in FINGERS for k in ("2","3","4","End")]
    return [names.index(label) for label in labels]


def absolute_poses(rng, count, prior, side="right"):
    deviations = np.sqrt(prior[f"{side}_pose_eigenvalues"])
    components = prior[f"{side}_pose_components"]
    weights = rng.normal(size=(count,len(deviations))).clip(-2.2,2.2)*.85
    offsets = (weights*deviations)@components
    poses = prior[f"{side}_pose_mean"]+offsets.reshape(count,24,3)
    # Include more open hands alongside the articulation prior, independent of
    # the wrist orientation chosen later by the arm solver.
    open_palm = rng.random(count)<.18
    poses[open_palm] *= rng.uniform(0.,.22,size=(open_palm.sum(),1,1))
    relative = np.broadcast_to(np.eye(3),(count,25,3,3)).copy()
    relative[:,1:] = Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(count,24,3,3)
    parent = prior[f"{side}_pose_reference_orient_parent"].swapaxes(-2,-1)
    orient = prior[f"{side}_pose_reference_orient"]
    return torch.from_numpy((parent@relative@orient).astype(np.float32))


@dataclass
class Digit:
    joints: np.ndarray
    start: np.ndarray
    bones: np.ndarray
    axes: np.ndarray
    pad_vertex: int
    pad_offset: np.ndarray
    pad_normal: np.ndarray
    tip_vertices: np.ndarray
    tip_offsets: np.ndarray

    @classmethod
    def from_bind(cls, chain, world, skin, palm, across, thumb):
        points = world[chain, :3, 3].astype(np.float64) * 1000.
        bones = np.diff(points, axis=0)
        directions = bones / np.linalg.norm(bones, axis=1, keepdims=True)
        flex = -across if thumb else palm
        spread = palm if thumb else across
        hinge = np.stack([unit(np.cross(direction, flex)) for direction in directions])
        swing = np.stack([unit(np.cross(direction, spread)) for direction in directions])
        axes = np.stack((hinge, swing, directions), axis=1)

        end = points[-1]
        tip_direction = directions[-1]
        pad_direction = unit(flex - (flex @ tip_direction) * tip_direction)
        center = end - 5. * tip_direction
        delta = skin - center
        candidates = np.flatnonzero(np.linalg.norm(delta, axis=1) < 13.)
        lateral = np.cross(tip_direction, pad_direction)
        offsets = delta[candidates]
        score = offsets @ pad_direction - 1.6 * np.hypot(offsets @ tip_direction, offsets @ lateral)
        vertex = int(candidates[score.argmax()])
        offsets = skin - end
        along = offsets @ tip_direction
        radial = np.linalg.norm(offsets - along[:, None] * tip_direction, axis=1)
        tip_vertices = np.flatnonzero((along >= -14.) & (along < 5.) & (radial < 12.))
        return cls(chain[:-1], points[0], bones, axes, vertex, skin[vertex] - end,
                   unit(pad_direction + .22 * tip_direction), tip_vertices, offsets[tip_vertices])

    def rotations(self, degrees):
        vectors = np.einsum("ji,jik->jk", np.deg2rad(degrees), self.axes)
        return Rotation.from_rotvec(vectors).as_matrix()

    def pad(self, degrees):
        point, turn = self.start.copy(), np.eye(3)
        for bone, rotation in zip(self.bones, self.rotations(degrees), strict=True):
            turn = turn @ rotation
            point += turn @ bone
        return point + turn @ self.pad_offset, turn @ self.pad_normal, point + self.tip_offsets @ turn.T

    def apply(self, local, world, degrees):
        bind = world[self.joints, :3, :3]
        local[self.joints-1] = local[self.joints-1] @ bind.swapaxes(-1, -2) @ self.rotations(degrees) @ bind


def fit_pinch(thumb, finger, rng):
    # The CMC provides opposition and axial roll. The finger can bend at
    # all three joints, with a small sideways movement at its base.
    thumb_controls = ([0, 0, 0, 1, 2], [0, 1, 2, 0, 0])
    finger_controls = ([0, 0, 1, 2], [0, 1, 0, 0])
    start = np.array([26., 12., 15., 10., 12., 35., 0., 40., 22.])
    lower = [-12., -32., -55., -12., -10., -8., -20., 0., 0.]
    upper = [72., 56., 75., 60., 65., 78., 20., 90., 60.]
    start += rng.normal(0., 3.5, len(start))
    gap = float(rng.uniform(.6, 2.5) if rng.random() < .75 else rng.uniform(4., 8.))

    def angles(values):
        thumb_angles = np.zeros((3, 3))
        finger_angles = np.zeros((3, 3))
        thumb_angles[thumb_controls] = values[:5]
        finger_angles[finger_controls] = values[5:]
        return thumb_angles, finger_angles

    def residual(values):
        a, b = angles(values)
        thumb_point, thumb_normal, thumb_tip = thumb.pad(a)
        finger_point, finger_normal, finger_tip = finger.pad(b)
        # Separate the curved tips along the thumb pad normal. Matching
        # only two pad vertices can leave the surrounding skin overlapping.
        offset = finger_point - thumb_point
        tangent = offset - (offset @ thumb_normal) * thumb_normal
        clearance = (finger_tip @ thumb_normal).min() - (thumb_tip @ thumb_normal).max()
        return np.concatenate((
            tangent, [clearance - gap],
            12. * (thumb_normal + finger_normal),
            .065 * (values - start),
        ))

    result = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper),
                           max_nfev=60, ftol=1.e-4)
    return (*angles(result.x), gap)


def gesture_pose(rng, gesture, side, indices, names, world, skin, tip_skin, local):
    """Pose a native hand using its own bind skeleton and fingertip skin."""
    palm_basis = palm_frame(world[indices, :3, 3])
    palm = palm_basis[:, 2] * (1 if side == "right" else -1)
    across = palm_basis[:, 0]
    prefix = side.title() + "Hand"
    digits = {}
    shapes = {}
    fan = rng.uniform(.8, 1.5) * np.array([0., 9., 3., -3., -9.])
    curl = rng.uniform(0., 1.)
    for name, spread in zip(("Thumb", *FINGERS), fan, strict=True):
        thumb = name == "Thumb"
        root = 1 if thumb else 2
        chain = np.array([names.index(prefix + name + suffix) for suffix in (str(root), str(root+1), str(root+2), "End")])
        digits[name] = Digit.from_bind(chain, world, skin[tip_skin[name]], palm, across, thumb)
        if thumb and gesture in ("fist", "point"):
            flex = rng.uniform([-12., 0., 5.], [-3., 15., 22.])
        elif thumb:
            flex = rng.uniform([-8., 0., 0.], [10., 12., 16.])
        elif gesture == "fist" or (gesture == "point" and name != "Index"):
            flex = np.array([46., 66., 24.]) + curl * np.array([17., 19., 18.]) + rng.uniform(-2., 2., 3)
        elif gesture.startswith("pinch-"):
            flex = np.array([-4., 0., 0.]) + curl * np.array([28., 40., 25.]) + rng.uniform(-2., 2., 3)
        else:
            flex = rng.uniform([-8., -4., -4.], [9., 5., 5.])
        shapes[name] = np.column_stack((flex, [spread + rng.uniform(-2., 2.), 0., 0.], np.zeros(3)))

    contact = None
    if gesture.startswith("pinch-"):
        name = gesture.removeprefix("pinch-").title()
        thumb, finger = (digits[digit] for digit in ("Thumb", name))
        shapes["Thumb"], shapes[name], gap = fit_pinch(thumb, finger, rng)
        thumb_skin, finger_skin = tip_skin["Thumb"], tip_skin[name]
        joint = thumb.joints[-1]
        normal_local = world[joint, :3, :3].T @ thumb.pad_normal
        contact = (
            [thumb_skin[thumb.pad_vertex], finger_skin[finger.pad_vertex]],
            thumb_skin[thumb.tip_vertices], finger_skin[finger.tip_vertices], joint, normal_local, gap,
        )
    for name, digit in digits.items():
        digit.apply(local, world, shapes[name])
    return contact


def local_frame(joints):
    def unit(vector):
        return vector/np.linalg.norm(vector,axis=-1,keepdims=True)
    y = unit(joints[:,9]-joints[:,0])
    x = joints[:,5]-joints[:,17]
    x = unit(x-np.sum(x*y,axis=-1,keepdims=True)*y)
    return np.stack((x,y,np.cross(x,y)),axis=-1)


def generate(args):
    from soma import SOMAHandLayer

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    layer = SOMAHandLayer(
        hand_type="right",lod="mid",device="cpu",mode="torch",data_root=args.assets,
        correctives_model_path=args.correctives,
    )
    names = layer.rig_data["joint_names"]
    indices = landmarks(names)
    with np.load(args.assets/"SOMAHand.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["right_pose_reference_joint_names"],names)
        np.testing.assert_array_equal(data["right_pose_reference_parent_ids"],layer.joint_parent_ids.numpy())
        np.testing.assert_allclose(data["right_pose_reference_orient"],
                                   layer.bind_pose_world.numpy()[:,:3,:3],atol=1e-5)
        poses = absolute_poses(rng,args.count,data)
    identity = rng.normal(0,.9,size=(args.count,layer.num_shape_components)).clip(-2.2,2.2).astype(np.float32)
    meshes,joints = [],[]
    with torch.no_grad():
        for start in range(0,args.count,args.batch):
            stop = start+args.batch
            layer.prepare_identity(torch.from_numpy(identity[start:stop]))
            output = layer.pose(poses[start:stop],pose2rot=False,absolute_pose=True,
                                apply_correctives=args.correctives is not None)
            points = output.joints.cpu().numpy()[:,indices]
            vertices = output.vertices.cpu().numpy()
            basis = local_frame(points)
            wrist = points[:,:1,:]
            meshes.append((vertices-wrist)@basis)
            joints.append((points-wrist)@basis)
            if start%512==0:
                print(f"Sampled {min(stop,args.count)}/{args.count} hand poses",flush=True)
    vertices,joints = np.concatenate(meshes),np.concatenate(joints)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("Model returned non-finite hand geometry")
    ids = [f"soma-{hashlib.sha256(a.tobytes()+p.numpy().tobytes()).hexdigest()[:20]}"
           for a,p in zip(identity,poses,strict=True)]
    if len(set(ids))!=args.count:
        raise ValueError("Duplicate hand poses")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.out,ids=ids,vertices=vertices,joints=joints,faces=layer.faces.cpu().numpy(),
             sampler_seed=args.seed,pose_correctives=args.correctives is not None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--correctives",type=Path)
    p.add_argument("--count",type=int,default=2048)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--batch",type=int,default=32)
    args = p.parse_args()
    if min(args.count,args.threads,args.batch)<1:
        p.error("Positive count, thread and batch sizes are required")
    generate(args)
