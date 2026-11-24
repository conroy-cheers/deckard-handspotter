"""Seeded appearance, scene occluders, and lens haze for the stereo renderer."""
from dataclasses import replace

import numpy as np

from synthetic_arm import pose_hand, sample_arm


SKIN_COLORS = [0x372015, 0x52301F, 0x734934, 0xA77553, 0xC59574, 0xEBC1A6]
CLOTH_COLORS = [0x232B35, 0x514846, 0x355350, 0x927239, 0x79373D, 0xC9C4B8, 0x545884]
GARMENTS = (
    ("bare arms", (-1., -1.), 0., 0., 0),
    ("short-sleeve jersey", (-.39, -.29), .12, .91, 0),
    ("rolled-up plaid", (-.21, -.11), .22, .86, 1),
    ("ribbed sweater", (-.040, -.014), .24, .94, 2),
    ("striped athletic sleeve", (-.055, -.013), .11, .72, 3),
)


def color_channels(color):
    return np.array([(color >> shift) & 255 for shift in (16, 8, 0)])


def sample_appearance(rng):
    # Interpolate the palette in sRGB. Three decodes the resulting color as sRGB.
    i = int(rng.integers(len(SKIN_COLORS)-1))
    t = rng.random()
    color = (1-t)*color_channels(SKIN_COLORS[i])+t*color_channels(SKIN_COLORS[i+1])
    red, green, blue = np.rint(np.clip(color+rng.uniform(-7, 7, size=3), 0, 255)).astype(int)
    garment, cuffs, allowance, roughness, pattern = GARMENTS[rng.integers(len(GARMENTS))]
    return dict(skin_color=int((red<<16)|(green<<8)|blue),
                garment=garment, cloth_color=int(rng.choice(CLOTH_COLORS)),
                cuff=float(rng.uniform(*cuffs)), sleeve_allowance=allowance,
                cloth_roughness=roughness, pattern=pattern,
                hand_proportions=rng.uniform([.82,.86,.80],[1.18,1.15,1.23]).tolist(),
                wrist_proportions=rng.uniform([.85,.83],[1.18,1.22]).tolist(),
                arm_girth=rng.uniform([.76,.72],[1.29,1.38]).tolist(),
                body_proportions=rng.uniform([.87,.92,.91],[1.16,1.08,1.10]).tolist())


def shape_body(body, appearance):
    shoulder, upper, forearm = appearance["body_proportions"]
    return replace(body, half_shoulder_width=body.half_shoulder_width*shoulder,
                   upper_length=body.upper_length*upper, forearm_length=body.forearm_length*forearm)


def shape_hand(asset, appearance):
    def deform(points):
        points = points.copy()
        # Taper the wrist dimensions into the existing palm and finger shape.
        t = np.clip((points[:, 1]-.008)/.039, 0, 1)
        t = t*t*(3-2*t)
        wrist = np.array(appearance["wrist_proportions"])
        points[:, [0,2]] *= wrist+(1-wrist)*t[:, None]
        return points*appearance["hand_proportions"]

    return dict(id=asset["id"], vertices=deform(asset["vertices"]), joints=deform(asset["joints"]))


def crossing_targets(rng):
    near = int(rng.integers(2))
    handed = 2*near-1
    target = [-handed*rng.uniform(.08,.15), rng.uniform(-.09,-.035), rng.uniform(-.27,-.21)]
    return near, target


def reach_behind_forearm(rng, body, side, foreground, asset, scale):
    # Cast a ray through the foreground forearm and aim the other hand's
    # knuckles just off that ray. A depth gap keeps the arm surfaces apart.
    point = .80*foreground.wrist+.20*foreground.elbow
    depth = max(.36, -point[2]+.18)
    desired = point*(-depth/point[2])
    wrist_ray = foreground.wrist[:2]/-foreground.wrist[2]
    tangent = wrist_ray-point[:2]/-point[2]
    perpendicular = np.array([-tangent[1],tangent[0]])/max(np.linalg.norm(tangent), 1e-9)
    desired[:2] += perpendicular*rng.choice([-1, 1])*rng.uniform(.032,.072)
    target = desired+[0., -.015, .063]
    pose_seed = int(rng.integers(2**32))
    for _ in range(3):
        arm = sample_arm(np.random.default_rng(pose_seed), body, side, target)
        _, joints = pose_hand(asset, arm, side, scale)
        target += .88*(desired-joints[[5,9,13,17]].mean(axis=0))
    return arm


def scene_debris(rng, hands):
    if rng.random() >= .62:
        return []
    pieces = []
    for _ in range(int(rng.integers(7, 25))):
        depth = rng.uniform(.105, .63)
        position = [depth*rng.uniform(-.68,.68), depth*rng.uniform(-.57,.29), -depth]
        radius = rng.uniform(.0007, .0040)
        pieces.append(debris_piece(rng, position, radius))
    if hands and rng.random() < .55:
        hand = hands[rng.choice(list(hands))]
        point = np.asarray(hand["keypoints_pair_m"])[[0,5,9,13,17]].mean(axis=0)
        position = rng.uniform(.44,.67)*point+rng.uniform(-.012,.012,3)
        pieces.append(debris_piece(rng, position, rng.uniform(.003,.008)))
    return pieces


def debris_piece(rng, position, radius):
    return dict(position=list(position), scale=(radius*rng.uniform([.7,.24,.38],[1.6,1.1,1.35])).tolist(),
                rotation=rng.uniform(-np.pi,np.pi,3).tolist(), material=int(rng.integers(2)))


def lens_parameters(rng, severity):
    return dict(haze=float(severity*rng.uniform(.66, 1.)), center=rng.uniform(-.72,.72,2).tolist(),
                width=rng.uniform([.21,.18],[.53,.46]).tolist(), smudge=float(rng.uniform(.03,.23)))
