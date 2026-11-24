"""Evaluate a native SOMA-X pose and its headset attachment in one pass."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from head_pose import in_reference_scene
from humanoid_geometry import (
    EYES_PAIR_M, HAND_GESTURES, SIDES, body_basis, hand_record, pair_points, retarget_arms, scene_record, turn_joint,
)
from sample_soma_hand import FINGERS, absolute_poses, gesture_pose, landmarks
from soma_arm_collision import ArmCollision


def headset_mount(rig_path, openvr_path):
    """Fit the leveled camera-pair calibration into the neutral head frame."""
    if rig_path is None:
        return np.eye(4), {}
    rig = json.loads(Path(rig_path).read_text())
    hardware = json.loads(Path(openvr_path).read_text())["cameras"]
    mount = None
    cameras = {}
    for slot in rig["stream_order"]:
        view = rig["cameras"][slot]
        camera = view["source_calibration"]
        if camera["projection_model_type"] != "kb4":
            raise ValueError("Native Frame attachment requires the calibrated KB4 rig")
        pair_from_eye = np.asarray(view["pair_from_eye"])
        openvr_from_eye = np.asarray(hardware[camera["label"]]["parent_from_eye"])
        candidate = openvr_from_eye @ np.linalg.inv(pair_from_eye)
        if mount is None:
            mount = candidate
        else:
            np.testing.assert_allclose(mount, candidate, atol=1.e-5, rtol=0.)
        cameras[slot] = dict(sensor=camera["label"], pair_from_eye=pair_from_eye.tolist())
    # Keep the existing seating-depth assumption. Calibrated OpenVR axes
    # give the orientation. The display calibration supplies no wearer fit.
    mount[:3, 3] = 0.
    return mount, cameras


class SomaLive:
    def __init__(self, assets, correctives, seed, threads=4, rig=None, openvr=None):
        from soma import SOMALayer

        torch.set_num_threads(threads)
        self.layer = SOMALayer(device="cpu", lod="mid", identity_model_type="soma",
                               data_root=assets, correctives_model_path=correctives)
        # SOMA freezes the corrective network for inference. Fold its fixed
        # support masks once, before the per-pose forward passes.
        correctives_mlp = self.layer.correctives_model.module
        with torch.no_grad():
            for weights, mask in (("W1", "M1_prior"), ("W2", "M2_prior")):
                support = getattr(correctives_mlp, mask)
                if support is not None:
                    getattr(correctives_mlp, weights).mul_(support)
                    setattr(correctives_mlp, mask, None)
        self.seed = seed
        names = self.layer.public_joint_names
        self.names = list(names)
        self.parents = self.layer.output_joint_parent_ids.numpy()[1:]
        self.arms = np.array([[names.index(side.title()+joint) for joint in ("Arm","ForeArm","Hand")]
                              for side in SIDES])
        self.points = [landmarks(names,side) for side in SIDES]
        self.eyes = [names.index(name) for name in ("LeftEye","RightEye")]
        self.neck = [names.index(name) for name in ("Neck1","Neck2","Head")]
        self.chest = [names.index(name) for name in ("Spine2","Chest","LeftShoulder","RightShoulder")]
        self.cloth_joints = [names.index(name) for name in ("Hips","Spine1","Chest","Neck1")]
        self.faces = self.layer.faces.numpy()
        self.labels = np.full(len(self.faces),3,np.uint8)
        self.fingers = []
        self.tip_skin = {}
        with np.load(assets/"SOMAHand.npz",allow_pickle=False) as data:
            self.prior = dict(data)
        target_names = list(self.layer.target_joint_names)
        weights = self.layer.skinning_weights.numpy()
        for side,label in zip(SIDES,(1,2),strict=True):
            self.labels[np.isin(self.faces,self.prior[f"{side}_vert_ids"]).all(axis=1)] = label
            self.fingers.append([names.index(name)-1 for name in self.prior[f"{side}_pose_joint_names"]])
            digit_ids = {}
            for finger in ("Thumb", *FINGERS):
                stem = side.title() + "Hand" + finger
                distal = stem + ("3" if finger == "Thumb" else "4")
                columns = [target_names.index(name) for name in (distal, stem + "End") if name in target_names]
                digit_ids[finger] = np.flatnonzero(weights[:, columns].sum(axis=1) > .8)
            self.tip_skin[side] = digit_ids
            if (self.labels == label).sum() < 100:
                raise ValueError("SOMA hand and body surface indexing disagree")
        self.mount,self.cameras = headset_mount(rig,openvr)
        self.collision = ArmCollision(self.layer)

    def evaluate(self, local, fk_only=True):
        return self.layer.pose(torch.as_tensor(local,dtype=torch.float32),pose2rot=False,
                               absolute_pose=True,fk_only=fk_only,apply_correctives=not fk_only)

    def world(self, local):
        return self.evaluate(local).transforms.numpy()

    @torch.no_grad()
    def prepare(self, row, index, rendered, gestures=None):
        self.row = in_reference_scene(row) if rendered else row
        self.index = index
        self.annotate_gestures = gestures is not None
        self.gestures = ("prior", "prior") if gestures is None else tuple(gestures)
        if len(self.gestures) != len(SIDES) or any(name not in ("prior", *HAND_GESTURES) for name in self.gestures):
            raise ValueError("Supply one supported hand gesture per side")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,index]))
        fingers = [absolute_poses(rng,1,self.prior,side).numpy()[0,1:] for side in SIDES]
        identity = rng.normal(0,.68,(1,self.layer.num_shape_components)).clip(-2,2).astype(np.float32)
        self.layer.prepare_identity(torch.from_numpy(identity))
        neutral = self.layer.pose(torch.zeros(1,len(self.parents),3),apply_correctives=False)
        neutral_world = neutral.transforms.numpy()[0]
        rest = neutral_world[:, :3, 3]
        s0,s1 = self.arms[:,0]
        self.basis = body_basis(rest[s0],rest[s1],np.array([0.,1.,0.]))
        self.local = (neutral_world[self.parents,:3,:3].swapaxes(-2,-1)@neutral_world[1:,:3,:3])[None]
        self.local[0,np.asarray(self.chest)-1] @= Rotation.from_euler(
            "xyz",rng.normal(0,[[3,4,2],[4,5,3],[3,5,5],[3,5,5]]),degrees=True).as_matrix()
        self.pinch_targets = {}
        neutral_skin = neutral.vertices.numpy()[0] * 1000.
        for slot, (side, gesture) in enumerate(zip(SIDES, self.gestures, strict=True)):
            if gesture == "prior":
                self.local[0, self.fingers[slot]] = fingers[slot]
                continue
            gesture_rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 24, slot]))
            contact = gesture_pose(gesture_rng, gesture, side, self.points[slot], self.names,
                                   neutral_world, neutral_skin, self.tip_skin[side], self.local[0])
            if contact is not None:
                self.pinch_targets[side] = contact
        body = self.world(self.local)[0]
        self.eye_midpoint = body[self.eyes,:3,3].mean(axis=0)
        self.reference_from_world = np.eye(4)
        self.reference_from_world[:3,:3] = self.basis.T
        self.reference_from_world[:3,3] = EYES_PAIR_M - self.basis.T @ self.eye_midpoint
        self.rest = dict(rest_vertices=self.reference(neutral.vertices.numpy()[0]).astype(np.float32),
                         rest_cloth_joints=self.reference(rest[self.cloth_joints]).astype(np.float32),
                         rest_arm_joints=self.reference(rest[self.arms]).astype(np.float32))
        retarget_arms(self.local,self.parents,self.world,[self.row],self.arms,self.points,
                      [self.basis],[self.eye_midpoint],bind_poses=[neutral_world])
        shoulders = self.arms[:,0]
        for _ in range(4):
            posed = self.evaluate(self.local,fk_only=False)
            mesh,world = posed.vertices.numpy()[0],posed.transforms.numpy()[0]
            fix = self.collision.adjustment(mesh,world[shoulders,:3,3],self.basis)
            if fix is None:
                break
            for joint,delta in zip(shoulders,fix,strict=True):
                turn_joint(self.local[0],world,self.parents,joint,delta)
        else:
            raise ValueError(f"Cannot separate arms of SOMA scene {row['id']}")
        self.head_neutral = self.reference_from_world @ world[self.neck[-1]]

    def reference(self, points):
        return pair_points(points,self.basis,self.eye_midpoint)

    def resolve_head_contacts(self, local, output):
        contacts = self.collision.head_contacts(output.vertices.numpy()[0])
        adjustments = {}
        for side in np.argsort(contacts)[::-1]:
            if not contacts[side]:
                continue
            vertices = output.vertices.numpy()[0]
            world = output.transforms.numpy()[0]
            chain = self.arms[side]
            for pivot, delta in self.collision.head_adjustments(vertices, side, world[chain,:3,3]):
                candidate = local.copy()
                turn_joint(candidate[0],world,self.parents,chain[pivot],delta)
                posed = self.evaluate(candidate,fk_only=False)
                skin = posed.vertices.numpy()[0]
                remaining = self.collision.head_contacts(skin)
                if remaining[side] or remaining[1-side] > contacts[1-side]:
                    continue
                if self.collision.crossing_faces(*(skin[ids] for ids in self.collision.vertices)):
                    continue
                wrist = chain[-1]
                shift = posed.transforms.numpy()[0,wrist,:3,3]-world[wrist,:3,3]
                adjustments[SIDES[side]] = dict(
                    joint=("shoulder","elbow")[pivot],
                    degrees=float(np.degrees(Rotation.from_matrix(delta).magnitude())),
                    wrist_shift_mm=float(1000*np.linalg.norm(shift)),
                )
                local, output, contacts = candidate, posed, remaining
                break
            else:
                raise ValueError(f"Cannot separate the {SIDES[side]} arm from the head in scene {self.row['id']}")
        return output, adjustments

    @torch.no_grad()
    def pose(self, angles):
        angles = np.asarray(angles,dtype=np.float64)
        if angles.shape != (3,) or not np.isfinite(angles).all() or (
            (angles < [-85,-76,-22]).any() or (angles > [85,70,22]).any()
        ):
            raise ValueError("Yaw, pitch or roll exceeds the scene limits")
        yaw,pitch,roll = angles
        rotvec = self.basis @ Rotation.from_euler("YXZ",[-yaw,pitch,roll],degrees=True).as_rotvec()
        local = self.local.copy()
        for joint,fraction in zip(self.neck,(.23,.32,.45),strict=True):
            turn_joint(local[0],self.world(local)[0],self.parents,joint,
                       Rotation.from_rotvec(rotvec*fraction).as_matrix())
        output = self.evaluate(local,fk_only=False)
        output, adjustments = self.resolve_head_contacts(local, output)
        world = output.transforms.numpy()[0]
        joints = self.reference(world[:,:3,3])
        head = self.reference_from_world @ world[self.neck[-1]]
        attachment = head @ np.linalg.inv(self.head_neutral) @ self.mount
        hands = {side:hand_record(self.row,"soma-body",side,joints[points],joints[s],joints[e])
                 for side,(s,e,_),points in zip(SIDES,self.arms,self.points,strict=True)}
        for side, hand in hands.items():
            # The native pose is keyed by worker seed and sample index.
            # Scene IDs restart in each render shard.
            hand["asset"] = f"soma-body-{self.seed}-{self.index}-{side}"
        if self.annotate_gestures:
            for side, gesture in zip(SIDES, self.gestures, strict=True):
                hands[side]["gesture"] = gesture
        skin = output.vertices.numpy()[0]
        for side, (pads, thumb_tip, finger_tip, joint, normal, gap) in self.pinch_targets.items():
            axis = world[joint, :3, :3] @ normal
            clearance = (skin[finger_tip] @ axis).min() - (skin[thumb_tip] @ axis).max()
            a, b = skin[pads]
            hands[side]["pinch"] = dict(target_clearance_mm=gap, tip_clearance_mm=float(1000. * clearance),
                                        pad_distance_mm=float(1000. * np.linalg.norm(a-b)))
        record = scene_record(self.row,"soma-body",hands,pose_correctives=True,arm_rig="stock")
        record["head_pose"] = dict(record.get("head_pose", {}), native_soma=True,
                                   yaw_pitch_roll_deg=angles.tolist(), reference_from_pair=attachment.tolist(),
                                   reference_from_head_joint=head.tolist(),
                                   pivot_reference_m=joints[self.neck[0]].tolist(),
                                   arm_adjustments=adjustments)
        bank = dict(self.rest, skin_vertices=self.reference(skin).astype(np.float32),
                    skin_labels=self.labels, skin_faces=self.faces)
        return record,bank
