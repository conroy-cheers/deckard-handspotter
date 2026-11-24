"""Pose a two-bone forearm and a two-pivot wrist on the SOMA-X mesh."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def unit(vector):
    return vector/np.linalg.norm(vector)


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value**2*(3.-2.*value)


def transform_about(point, rotation):
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3,3] = point-rotation@point
    return transform


def transform_point(transform, point):
    return transform[:3,:3]@point+transform[:3,3]


def twist_angle(rotation, axis):
    # Project the quaternion onto the rotation subspace about the forearm axis.
    q = Rotation.from_matrix(rotation).as_quat(canonical=True)
    return 2.*np.arctan2(np.dot(q[:3],axis),q[3])


def bounded_roll(roll):
    # Reserve a soft approach to the neutral-relative roll envelope so that
    # out-of-range targets do not all concentrate on one exact limit.
    start,band = np.deg2rad([65.,20.])
    excess = max(0.,abs(roll)-start)
    return np.copysign(min(abs(roll),start)+band*excess/(band+excess),roll)


@dataclass(frozen=True)
class Forearm:
    radial_head: np.ndarray
    ulnar_head: np.ndarray
    radiocarpal: np.ndarray
    midcarpal: np.ndarray
    palm_normal: np.ndarray
    forward: np.ndarray
    radial: np.ndarray
    wrist: np.ndarray
    length: float
    width: float

    @classmethod
    def fit(cls, neutral, bind, elbow, wrist, landmarks, side):
        center = bind[wrist,:3,3]
        origin = bind[elbow,:3,3]
        forward = unit(center-origin)
        MCP = bind[np.asarray(landmarks)[[5,9,13,17]],:3,3]
        across = MCP[0]-MCP[-1]
        radial = unit(across-np.dot(across,forward)*forward)
        palm_normal = np.cross(radial,forward)*(1 if side=="right" else -1)
        relative = neutral-center
        near = (np.abs(relative@forward)<.015)&(np.linalg.norm(relative,axis=1)<.065)
        low,high = np.quantile(relative[near]@radial,[.05,.95])
        width = float(high-low)
        # Approximate the intrabony landmarks from the fitted wrist breadth.
        # These offsets describe a rendering rig, with the distal ulna on the
        # little-finger side and the proximal radius on the thumb side.
        radial_head = origin+width*(.23*forward+.39*radial)
        ulnar_head = center+width*(-.10*forward-.30*radial)
        radiocarpal = center+width*.08*radial
        midcarpal = radiocarpal+width*.27*forward
        return cls(radial_head,ulnar_head,radiocarpal,midcarpal,palm_normal,forward,
                   radial,center,float(np.linalg.norm(center-origin)),width)

    def roll_axis(self, elbow_motion):
        radial_head = transform_point(elbow_motion,self.radial_head)
        ulnar_head = transform_point(elbow_motion,self.ulnar_head)
        return radial_head,unit(ulnar_head-radial_head)

    def limit_hand_rotation(self, elbow_motion, desired_hand_rotation):
        _,axis = self.roll_axis(elbow_motion)
        rotation = desired_hand_rotation@elbow_motion[:3,:3].T
        roll = twist_angle(rotation,axis)
        correction = bounded_roll(roll)-roll
        limited = rotation@Rotation.from_rotvec(correction*axis).as_matrix()@elbow_motion[:3,:3]
        return limited,roll,correction

    def matrices(self, elbow_motion, desired_hand_rotation):
        """Return ulna, radius, proximal carpal row and distal hand motions."""
        radial_head,axis = self.roll_axis(elbow_motion)
        rotation = desired_hand_rotation@elbow_motion[:3,:3].T
        roll = twist_angle(rotation,axis)
        radius = transform_about(radial_head,Rotation.from_rotvec(roll*axis).as_matrix())@elbow_motion

        bend = Rotation.from_matrix(desired_hand_rotation@radius[:3,:3].T)
        direction = radius[:3,:3]@self.forward
        palm = radius[:3,:3]@self.palm_normal
        # Extension draws more on the midcarpal joint. Use a continuous transition
        # for neutral, sideways and combined bends.
        dorsal_bend = -np.dot(bend.apply(direction),palm)
        mid_fraction = .50+.20*smoothstep(dorsal_bend/.5)
        angles = bend.as_rotvec()
        proximal = transform_about(transform_point(radius,self.radiocarpal),
                                   Rotation.from_rotvec((1.-mid_fraction)*angles).as_matrix())@radius
        distal = transform_about(transform_point(proximal,self.midcarpal),
                                 Rotation.from_rotvec(mid_fraction*angles).as_matrix())@proximal
        return np.stack((elbow_motion,radius,proximal,distal)),roll,mid_fraction

    def weights(self, vertices):
        q = vertices-self.wrist
        longitudinal = q@self.forward/self.width
        lateral = q@self.radial/self.width
        # Feather the proximal and distal carpal influences across separate
        # parts of the wrist. Leave articulated finger weights to SOMA-X.
        proximal = smoothstep((longitudinal+.85)/1.28)
        distal = smoothstep((longitudinal+.32)/1.32)
        from_elbow = 1.+longitudinal*self.width/self.length
        # The skin and muscle envelope shares motion across both bones. Bias
        # the thumb side toward the radius without pinning either skin edge
        # to a shaft, which can carve a groove as the radius crosses the ulna.
        radius = smoothstep((from_elbow-.04)/.9)*(.67+.20*np.tanh(2.*lateral))
        return np.stack(((1.-proximal)*(1.-radius),(1.-proximal)*radius,
                         proximal-distal,distal),axis=1)


def quaternion_product(a,b):
    vector = a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+np.cross(a[...,:3],b[...,:3])
    real = a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(axis=-1,keepdims=True)
    return np.concatenate((vector,real),axis=-1)


def skin_helpers(points, weights, transforms):
    """Blend rigid helper motions with dual quaternions to retain wrist volume."""
    real = Rotation.from_matrix(transforms[:,:3,:3]).as_quat()
    translation = np.pad(transforms[:,:3,3],((0,0),(0,1)))
    dual = .5*quaternion_product(translation,real)
    # Align signs to the dominant transform at each vertex before blending.
    weighted = weights*np.where(real[weights.argmax(axis=1)]@real.T>=0.,1.,-1.)
    rotation = weighted@real
    norm = np.linalg.norm(rotation,axis=1,keepdims=True)
    rotation /= norm
    offset = 2.*quaternion_product(weighted@dual/norm,rotation*[-1,-1,-1,1])[:,:3]
    return Rotation.from_quat(rotation).apply(points)+offset


class SomaArmSkinning:
    """Replace just the stock forearm/wrist influences in one SOMA-X pose."""
    def __init__(self, layer, arms, landmarks):
        self.layer = layer
        skinner = layer.batched_skinning
        target = layer.target_joint_names
        public = layer.public_joint_names
        weight = skinner.get_bone_weights().numpy()
        bone = skinner.get_bone_indices()
        if bone is not None:
            bone = bone.numpy()
            sparse = weight
            weight = np.zeros_like(layer.skinning_weights.numpy())
            rows,cols = np.where(bone>=0)
            weight[rows,bone[rows,cols]] = sparse[rows,cols]
        self.sides = []
        for side,(_,elbow,wrist),points in zip(("left","right"),arms,landmarks,strict=True):
            prefix = side.title()
            forearm = [i for i,name in enumerate(target)
                       if name.startswith(prefix+"ForeArm") or name==prefix+"Hand"]
            fingers = [i for i,name in enumerate(target)
                       if name.startswith(prefix+"Hand") and name!=prefix+"Hand"]
            moving_joints = [i for i,name in enumerate(public) if name.startswith(prefix+"Hand")]
            affected = np.flatnonzero(weight[:,forearm].sum(axis=1)>0)
            weight_on_helpers = weight[affected][:,forearm].astype(np.float64)
            self.sides.append((side,elbow,wrist,points,forearm,affected,weight_on_helpers,
                               weight[:,fingers].sum(axis=1),moving_joints))

    def fitted_arms(self):
        """Measure each joint and bind its weights to the neutral identity."""
        skinner = self.layer.batched_skinning
        neutral = skinner.bind_shapes.numpy()
        bind = self.layer.public_bind_transforms_world().numpy()
        fitted = []
        for mesh,pose in zip(neutral,bind,strict=True):
            model = []
            for side,elbow,wrist,points,_,indices,*_ in self.sides:
                fit = Forearm.fit(mesh,pose,elbow,wrist,points,side)
                neutral_patch = mesh[indices].astype(np.float64)
                along = (neutral_patch-fit.wrist)@fit.forward/fit.width
                # Stock correctives were learned around its single wrist
                # pivot. Fade that displacement around the two carpal rows
                # while retaining the learned elbow and finger deformation.
                carpal = smoothstep((along+.9)/.8)*(1.-smoothstep((along-.6)/.65))
                model.append((fit,fit.weights(neutral_patch),neutral_patch,.75*carpal[:,None]))
            fitted.append(model)
        return fitted

    def constrain(self, local, initial, fitted):
        """Drive the SOMA-X hand and corrective model with the achievable pose."""
        bind = self.layer.public_bind_transforms_world().numpy()
        initial = initial.numpy()
        requested_roll = []
        for i,model in enumerate(fitted):
            angles = []
            for (fit,*_),(_,elbow,wrist,*_) in zip(model,self.sides,strict=True):
                pose = initial[i]
                elbow_motion = pose[elbow]@np.linalg.inv(bind[i,elbow])
                desired = pose[wrist,:3,:3]@bind[i,wrist,:3,:3].T
                limited,roll,correction = fit.limit_hand_rotation(elbow_motion,desired)
                if correction:
                    local[i,wrist-1] = pose[elbow,:3,:3].T@limited@bind[i,wrist,:3,:3]
                angles.append(float(np.degrees(roll)))
            requested_roll.append(angles)
        return requested_roll

    def deform(self, local, output, fitted):
        skinner = self.layer.batched_skinning
        public = output.transforms
        root = torch.eye(3,dtype=local.dtype).expand(len(local),1,3,3)
        target = skinner.expand_source_world_transforms(
            torch.cat((root,local),dim=1),public,
            transform_expander=self.layer.procedural_transforms.expand_world_transforms_from_source_fk,
        )
        delta = (target@skinner.inverse_bind_transform).numpy()
        bind = self.layer.public_bind_transforms_world().numpy()
        rest = skinner.bind_shapes.numpy()
        world = public.numpy().copy()
        vertices = output.vertices.numpy().copy()
        diagnostics = []
        for i,model in enumerate(fitted):
            info = {}
            for (fit,helper_weights,neutral_patch,carpal),side_data in zip(model,self.sides,strict=True):
                side,elbow,wrist,_,bones,indices,weights,finger_weight,joints = side_data
                elbow_motion = world[i,elbow]@np.linalg.inv(bind[i,elbow])
                desired_rotation = world[i,wrist,:3,:3]@bind[i,wrist,:3,:3].T
                transforms,roll,mid_fraction = fit.matrices(elbow_motion,desired_rotation)
                # The distal pivot keeps the requested palm orientation. Its
                # translation follows the two carpal centers and the radius.
                hand = transforms[-1]@bind[i,wrist]
                shift = hand[:3,3]-world[i,wrist,:3,3]
                world[i,joints,:3,3] += shift

                points = rest[i,indices].astype(np.float64)
                motion = delta[i,bones,:3].astype(np.float64)
                original = np.einsum("vk,kij,vj->vi",weights,motion[:,:,:3],points)+weights@motion[:,:,3]
                carpal_shape = points+carpal*(neutral_patch-points)
                revised = skin_helpers(carpal_shape,helper_weights,transforms)
                vertices[i,indices] += weights.sum(axis=1,keepdims=True)*revised-original
                vertices[i] += finger_weight[:,None]*shift
                info[side] = dict(forearm_rotation_deg=float(np.degrees(roll)),
                                  midcarpal_fraction=float(mid_fraction),
                                  wrist_shift_mm=float(1000.*np.linalg.norm(shift)))
            diagnostics.append(info)
        return vertices,world,diagnostics
