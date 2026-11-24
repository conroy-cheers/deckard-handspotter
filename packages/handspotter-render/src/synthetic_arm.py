"""Pose the hand and arm as a single kinematic chain in the camera-pair frame."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from head_pose import front_direction
from render import unit


@dataclass
class Body:
    half_shoulder_width: float
    shoulder_y: float
    shoulder_z: float
    upper_length: float
    forearm_length: float

    @classmethod
    def sample(cls, rng):
        scale = rng.uniform(.91, 1.09)
        return cls(.192*scale, rng.uniform(-.325, -.285), rng.uniform(.05, .115),
                   .31*scale, .255*scale)


@dataclass
class Arm:
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    hand_rotation: np.ndarray
    flexion: float
    deviation: float
    pronation: float
    handed: int

    def record(self):
        return dict(shoulder_pair_m=self.shoulder.tolist(), elbow_pair_m=self.elbow.tolist(),
                    wrist_pair_m=self.wrist.tolist(), flexion_deg=float(self.flexion),
                    deviation_deg=float(self.deviation), pronation_deg=float(self.pronation))


def solve_reach(shoulder, target, pole, upper, lower, elbow_limits=(55.,155.)):
    """Intersect the upper-arm and forearm spheres on the elbow's preferred side."""
    reach = target-shoulder
    direction = unit(reach)
    # Bound the interior elbow angle away from fully folded and fully straight.
    distances = np.sqrt(upper**2+lower**2-2*upper*lower*np.cos(np.deg2rad(elbow_limits)))
    distance = np.clip(np.linalg.norm(reach), *distances)
    wrist = shoulder+distance*direction
    along = (upper**2-lower**2+distance**2)/(2*distance)
    sideways = np.sqrt(max(0., upper**2-along**2))
    elbow_direction = unit(pole-np.dot(pole, direction)*direction)
    elbow = shoulder+along*direction+sideways*elbow_direction
    return elbow, wrist


def wide_reach_target(rng, side):
    handed = 2*side-1
    depth_range = ((.15,.27),(.26,.45),(.41,.64))[rng.integers(3)]
    return np.array([handed*rng.uniform(-.12,.34),rng.uniform(-.36,.10),-rng.uniform(*depth_range)])


def sample_arm(rng, body, side, target=None, wide=False, elbow_pole=None):
    handed = -1 if side == 0 else 1
    shoulder = np.array([handed*body.half_shoulder_width, body.shoulder_y, body.shoulder_z])
    default_target = np.array([handed*rng.uniform(.075, .235), rng.uniform(-.235, -.060),
                              rng.uniform(-.485, -.295)])
    pole = np.array([handed*rng.uniform(.32, .76), -1., rng.uniform(.02, .48)])
    if elbow_pole is not None:
        pole = elbow_pole
    reach = default_target if target is None else np.asarray(target)
    limits = (40.,166.) if wide else (55.,155.)
    elbow, wrist = solve_reach(shoulder, reach, pole, body.upper_length, body.forearm_length,limits)

    forward = unit(wrist-elbow)
    upper = shoulder-elbow
    # The elbow plane defines a neutral, thumb-up orientation.
    radial = handed*unit(upper-np.dot(upper, forward)*forward)
    basis = np.column_stack((radial, forward, np.cross(radial, forward)))
    pronation = np.clip(rng.normal(35., 36.), -68., 80.)
    twist = Rotation.from_euler("y", -handed*pronation, degrees=True).as_matrix()
    # Sample within an elliptical flexion/deviation envelope.
    bend = rng.normal(0., .43, 2)
    bend /= max(1., np.linalg.norm(bend))
    flexion, deviation = bend*[43., 17.]
    wrist_bend = Rotation.from_euler("XZ", [flexion, -handed*deviation], degrees=True).as_matrix()
    return Arm(shoulder, elbow, wrist, basis@twist@wrist_bend, flexion, deviation, pronation, handed)


def arm_clearance_samples(arm):
    # Approximate limb capsules with regularly spaced samples, including a
    # short palm extension that accounts for the wrist orientation.
    upper = np.linspace(arm.shoulder, arm.elbow, 9)[3:]
    lower = np.linspace(arm.elbow, arm.wrist, 10)[1:]
    palm = arm.wrist + np.outer([.04, .09, .14], arm.hand_rotation[:, 1])
    return np.vstack((upper,lower,palm)), np.r_[np.full(6,.054), np.linspace(.048,.033,9), .044,.044,.038]


def hemisphere_arm(rng, body, side, other=None):
    """Sample a reachable wrist across the volume in front of the torso."""
    center = np.array([0., body.shoulder_y-.07, body.shoulder_z-.14])
    handed = 2*side-1
    other_shape = arm_clearance_samples(other) if other is not None else None
    for _ in range(256):
        target = center + rng.uniform(.10,.68)*front_direction(rng)
        pole = np.array([handed*rng.uniform(.35, 1.), rng.uniform(-1., 1.), -rng.uniform(.30, 1.)])
        arm = sample_arm(rng,body,side,target=target,wide=True,elbow_pole=pole)
        if np.linalg.norm(arm.wrist-target) > 1.e-6:
            continue
        samples, radius = arm_clearance_samples(arm)
        # Keep the hand and the distal limbs out of the chest and head.
        local = samples - [0.,body.shoulder_y,body.shoulder_z+.012]
        chest = (np.abs(local[:,0]) < body.half_shoulder_width + .6*radius)
        chest &= (np.abs(local[:,2]) < .106 + .63*radius)
        chest &= (local[:,1] >= -.52) & (local[:,1] <= .045)
        head = (samples-[0.,-.004,.135]) / ([.080,.120,.125]+radius[:,None])
        if chest.any() or np.any(np.linalg.norm(head,axis=1) < 1.):
            continue
        if other_shape is not None:
            points, thickness = other_shape
            separation = np.linalg.norm(samples[:,None]-points,axis=-1)
            if np.any(separation < radius[:,None]+thickness):
                continue
        return arm
    raise ValueError("Unable to sample a clear hemispherical reach")


def pose_hand(asset, arm, side, scale):
    """Put the hand's palm axis in the wrist frame without changing finger articulation."""
    joints = asset["joints"]
    origin = joints[0]
    palm = unit(joints[[5, 9, 13, 17]].mean(axis=0)-origin)
    transverse = joints[5]-joints[17]
    radial = unit(transverse-np.dot(transverse, palm)*palm)
    frame = np.column_stack((radial, palm, np.cross(radial, palm)))
    reflection = np.diag([-1., 1., 1.]) if side == 0 else np.eye(3)
    transform = frame@reflection@arm.hand_rotation.T*scale
    return (asset["vertices"]-origin)@transform+arm.wrist, (joints-origin)@transform+arm.wrist


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)


# Perry et al. (2021), doi:10.3389/frobt.2021.576783, table 3.
# Cubics for the major and minor semiaxes per unit flexed forearm circumference.
_SUPINATION = np.array([[.1713, -.1990, -.0535, .1922],
                       [.3329, -.5798, .2068, .1156]])
_PRONATION = np.array([[.2536, -.3264, -.0004, .1808],
                       [.2882, -.5210, .1894, .1225]])


def arm_sections(station, lower, upper, pronation, girth):
    # ANSUR II median flexed forearm circumferences are 263 and 310 mm across
    # the two survey groups. Sample build independently of the donor's wrist.
    forearm_circ = .287*np.sqrt(lower/.255)*(1+.68*(girth[0]-1))
    upper_circ = .315*np.sqrt(upper/.310)*(1+.68*(girth[1]-1))
    # The scan profiles span 20% radiale–stylion length through the wrist.
    x = 1-.80*np.minimum(station/lower, 1)
    mix = np.clip((pronation+80)/120, 0, 1)
    coeff = (1-mix)*_SUPINATION+mix*_PRONATION
    forearm_axes = np.array([np.polyval(axis, x) for axis in coeff]).T*forearm_circ

    # Continue through the elbow into an upper-arm ellipse with a biceps belly.
    # This section is procedural. The survey's comparable biceps measure is flexed.
    fraction = (station-lower)/upper
    aspect = .86
    perimeter = np.pi*(3*(1+aspect)-np.sqrt((3+aspect)*(1+3*aspect)))
    upper_axes = upper_circ*np.array([1,aspect])/perimeter
    upper_axes = (1-.13*smoothstep((fraction-.52)/.48))[:, None]*upper_axes
    blend = smoothstep(fraction/.36)[:, None]
    return forearm_axes*(1-blend)+upper_axes*blend


def arm_path(opening, arm, corner_radius, shoulder_inset):
    """Join the two limb axes with a circular elbow fillet and transport frame."""
    lower = np.linalg.norm(arm.elbow-opening)
    upper = np.linalg.norm(arm.shoulder-arm.elbow)
    along = (arm.elbow-opening)/lower
    onward = (arm.shoulder-arm.elbow)/upper
    normal = unit(np.cross(along, onward))
    inward = np.cross(normal, along)
    angle = np.arctan2(onward@inward, onward@along)
    tangent = np.tan(angle/2)
    trim = min(corner_radius*tangent, .74*lower, .66*upper)
    radius = trim/tangent

    def samples(length):
        return np.linspace(0, length, int(np.ceil(length/.014))+1)[1:]

    near = samples(lower-trim)
    curved = samples(radius*angle)
    far = samples(upper-trim-shoulder_inset)
    phi = (curved/radius)[:, None]
    sine, cosine = np.sin(phi), np.cos(phi)
    entry = arm.elbow-trim*along
    exit = arm.elbow+trim*onward
    centers = np.vstack((opening+near[:, None]*along,
                         entry+radius*(sine*along+(1-cosine)*inward),
                         exit+far[:, None]*onward))
    axes = np.vstack((np.tile(along, (len(near),1)), cosine*along+sine*inward,
                      np.tile(onward, (len(far),1))))
    distance = np.r_[near, lower-trim+curved, lower-trim+radius*angle+far]
    station = np.r_[near, lower-trim+2*trim*curved/curved[-1], lower+trim+far]
    return centers, axes, normal, distance, station


def arm_surface(vertices, rest, faces, boundary, arm, appearance=None, shoulder_inset=0.):
    """Blend the original wrist boundary into independently sized limb sections."""
    rim, edges = boundary
    opening = vertices[rim].mean(axis=0)
    lower = np.linalg.norm(arm.elbow-opening)
    upper = np.linalg.norm(arm.shoulder-arm.elbow)
    girth = appearance["arm_girth"] if appearance else (1.,1.)
    proximal_sections = arm_sections(np.array([lower, lower+.36*upper]), lower, upper, arm.pronation, girth)
    centers, axes, normal, distance, station = arm_path(opening, arm, proximal_sections.max()+.020, shoulder_inset)
    sections = arm_sections(station, lower, upper, arm.pronation, girth)
    if appearance:
        cloth = smoothstep((distance+appearance["cuff"])/.025)
        sections *= (1+appearance["sleeve_allowance"]*cloth)[:, None]

    world_rim = vertices[rim]-opening
    wrist_radial, wrist_forward, wrist_normal = arm.hand_rotation.T
    wrist_plane = world_rim@np.stack((wrist_radial,wrist_normal)).T
    # Preserve vertex correspondence around the seam as the rim straightens.
    phase = wrist_plane/np.linalg.norm(wrist_plane, axis=0)
    phase /= np.linalg.norm(phase, axis=1)[:, None]
    major = arm.handed*np.cross(normal, axes)
    minor = np.cross(major, -axes)
    wrist_reference = Rotation.align_vectors(-axes[0], wrist_forward)[0].apply(wrist_radial)
    wrist_angle = np.arctan2(wrist_reference@minor[0], wrist_reference@major[0])
    twist = wrist_angle*(1-smoothstep(station/lower))[:, None]
    major, minor = (np.cos(twist)*major+np.sin(twist)*minor,
                    -np.sin(twist)*major+np.cos(twist)*minor)
    ellipse = phase[None,:,:]*sections[:,None,:]
    offsets = ellipse@np.stack((major,minor),axis=1)
    seam = smoothstep(distance/min(.067,.28*lower))[:, None, None]
    world_rings = centers[:,None,:]+(1-seam)*world_rim+seam*offsets

    rest_center = rest[rim].mean(axis=0)
    rest_offsets = np.stack((arm.handed*ellipse[:,:,0], np.zeros(ellipse.shape[:2]), ellipse[:,:,1]), axis=2)
    rest_centers = rest_center+distance[:,None]*[0.,-1.,0.]
    rest_rings = rest_centers[:,None,:]+(1-seam)*(rest[rim]-rest_center)+seam*rest_offsets
    strips = []
    index = np.arange(len(vertices), len(vertices)+len(rim)*len(centers)).reshape(len(centers), -1)
    for inner, outer in zip((rim, *index[:-1]), index):
        p,q = inner[edges].T
        u,v = outer[edges].T
        strips.extend((np.stack((q,p,u), axis=1), np.stack((q,u,v), axis=1)))
    return (np.vstack((vertices, world_rings.reshape(-1,3))),
            np.vstack((rest, rest_rings.reshape(-1,3))), np.vstack((faces, *strips)))
