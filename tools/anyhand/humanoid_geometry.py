"""Shared arm retargeting and scene export for the humanoid pose banks."""
import hashlib
import json

import numpy as np


SIDES = ("left", "right")
HAND_GESTURES = ("open", "fist", "point", "pinch-index", "pinch-middle", "pinch-ring", "pinch-pinky")
EYES_PAIR_M = np.array([0., .005, .095])


def unit(vector):
    return vector/np.linalg.norm(vector)


def body_basis(left, right, up):
    across = right-left
    x = unit(across-np.dot(across, up)*up)
    return np.column_stack((x, up, np.cross(x, up)))


def frame_along(direction, guide):
    y = unit(direction)
    x = unit(guide-np.dot(guide, y)*y)
    return np.column_stack((x, y, np.cross(x, y)))


def palm_frame(points):
    return frame_along(points[[5,9,13,17]].mean(axis=0)-points[0], points[5]-points[17])


def reach_targets(record, side, shoulder, elbow, wrist, basis, eyes):
    """Solve shoulder-elbow-wrist in the body's own metric skeleton."""
    hand = record["hands"].get(side)
    sign = -1 if side == "left" else 1
    if hand:
        target_pair = np.asarray(hand["arm"]["wrist_pair_m"])
        pole_pair = np.asarray(hand["arm"]["elbow_pair_m"])-hand["arm"]["shoulder_pair_m"]
    else:
        target_pair = np.array([sign*.30, -.78, .23])
        pole_pair = np.array([sign*.4, -.4, .65])
    upper, lower = np.linalg.norm(elbow-shoulder), np.linalg.norm(wrist-elbow)
    if hand and record.get("reach_profile") == "hemisphere":
        # Retarget the amount and direction of extension in the donor's
        # shoulder frame. Absolute targets pile up at full extension on
        # short-armed identities, especially for the new lateral reaches.
        donor_shoulder = np.asarray(hand["arm"]["shoulder_pair_m"])
        donor_elbow = np.asarray(hand["arm"]["elbow_pair_m"])
        donor_span = np.linalg.norm(pole_pair)+np.linalg.norm(target_pair-donor_elbow)
        scale = (upper+lower)/donor_span
        target = shoulder+scale*(target_pair-donor_shoulder)@basis.T
    else:
        target = (target_pair-EYES_PAIR_M)@basis.T+eyes
    pole = pole_pair@basis.T
    direction = unit(target-shoulder)
    bounds = np.sqrt(upper**2+lower**2-2*upper*lower*np.cos(np.deg2rad([40,166])))
    distance = float(np.clip(np.linalg.norm(target-shoulder), *bounds))
    along = (upper**2-lower**2+distance**2)/(2*distance)
    normal = unit(pole-np.dot(pole,direction)*direction)
    goal_elbow = shoulder+along*direction+np.sqrt(max(0.,upper**2-along**2))*normal
    return goal_elbow, shoulder+distance*direction


def arm_rotation(old_shoulder, old_elbow, elbow, wrist, neutral_palm, forward):
    # Place the elbow's hinge axis perpendicular to the target bend plane.
    axis = np.cross(elbow-old_shoulder,wrist-elbow)
    source = frame_along(old_elbow-old_shoulder, neutral_palm[:,2])
    destination = frame_along(elbow-old_shoulder, axis if np.dot(axis,forward)>0 else -axis)
    return destination@source.T


def forearm_rotation(shoulder, elbow, wrist, target, guide):
    source = frame_along(wrist-elbow, guide)
    axis = np.cross(elbow-shoulder,target-elbow)
    if np.dot(axis,guide)<0:
        axis = -axis
    return frame_along(target-elbow, axis)@source.T


def wrist_rotation(points, elbow, source, basis):
    current = palm_frame(points)
    desired = current if source is None else basis@palm_frame(np.asarray(source["keypoints_pair_m"]))
    palm = desired[:,1]
    lower = unit(points[0]-elbow)
    limit = np.deg2rad(43.)
    alignment = np.dot(lower,palm)
    if alignment<np.cos(limit):
        cross = np.cross(lower,palm)
        axis = unit(cross) if np.linalg.norm(cross)>1e-8 else desired[:,0]
        limited = np.cos(limit)*lower+np.sin(limit)*np.cross(axis,lower)
        desired = frame_along(limited,axis)@frame_along(palm,axis).T@desired
    return desired@current.T


def turn_joint(local, world, parents, joint, delta):
    local[joint-1] = world[parents[joint-1], :3, :3].T @ delta @ world[joint, :3, :3]


def native_shoulder_rotation(current, bind, chain, elbow, wrist, forward):
    shoulder_joint, elbow_joint, wrist_joint = chain
    shoulder = current[shoulder_joint, :3, 3]
    upper = current[elbow_joint, :3, 3]-shoulder
    native_forearm = bind[wrist_joint, :3, 3]-bind[elbow_joint, :3, 3]
    # In the bind pose, flex the forearm toward the front of the body.
    # Carry this signed axis with the shoulder, even if the reach points back.
    hinge = np.cross(native_forearm, forward)
    shoulder_motion = current[shoulder_joint, :3, :3] @ bind[shoulder_joint, :3, :3].T
    target_hinge = np.cross(elbow-shoulder, wrist-elbow)
    return frame_along(elbow-shoulder, target_hinge) @ frame_along(upper, shoulder_motion @ hinge).T


def native_elbow_rotation(shoulder, elbow, wrist, goal):
    before, after = unit(wrist-elbow), unit(goal-elbow)
    axis = np.cross(before, after)
    if np.linalg.norm(axis) < 1e-7:
        if before @ after > 0:
            return np.eye(3)
        axis = np.cross(elbow-shoulder, after)
    return frame_along(after, axis) @ frame_along(before, axis).T


def retarget_arms(local, parents, world_for, rows, arms, points, bases, eyes, bind_poses=None):
    """Solve the shoulder, elbow and wrist in sequence using the fitted rig."""
    world = world_for(local)
    targets = []
    for i, (row, basis) in enumerate(zip(rows, bases, strict=True)):
        goals = []
        joints = world[i, :, :3, 3]
        for side,(s,e,w),indices in zip(SIDES,arms,points,strict=True):
            elbow,wrist = reach_targets(row,side,*joints[[s,e,w]],basis,eyes[i])
            if bind_poses is None:
                delta = arm_rotation(*joints[[s,e]],elbow,wrist,palm_frame(joints[indices]),-basis[:,2])
            else:
                delta = native_shoulder_rotation(world[i],bind_poses[i],(s,e,w),elbow,wrist,-basis[:,2])
            turn_joint(local[i],world[i],parents,s,delta)
            goals.append(wrist)
        targets.append(goals)
    world = world_for(local)
    for i, goals in enumerate(targets):
        for (s,e,w),indices,goal in zip(arms,points,goals,strict=True):
            joints = world[i, :, :3, 3]
            if bind_poses is None:
                delta = forearm_rotation(*joints[[s,e,w]],goal,palm_frame(joints[indices])[:,2])
            else:
                delta = native_elbow_rotation(*joints[[s,e,w]],goal)
            turn_joint(local[i],world[i],parents,e,delta)
    world = world_for(local)
    for i,(row,basis) in enumerate(zip(rows,bases,strict=True)):
        for side,(_,e,w),indices in zip(SIDES,arms,points,strict=True):
            delta = wrist_rotation(world[i,indices,:3,3],world[i,e,:3,3],row["hands"].get(side),basis)
            turn_joint(local[i],world[i],parents,w,delta)


def pair_points(points, basis, eyes):
    return (np.asarray(points)-eyes)@basis+EYES_PAIR_M


def hand_record(source, model, side, points, shoulder, elbow):
    arm = dict(shoulder_pair_m=shoulder.tolist(), elbow_pair_m=elbow.tolist(),
               wrist_pair_m=points[0].tolist())
    original = source["hands"].get(side)
    if original:
        arm["retarget_error_mm"] = float(1000*np.linalg.norm(
            points[0]-original["arm"]["wrist_pair_m"]))
    pose_hash = hashlib.sha256(np.asarray(points, dtype="<f4").tobytes()).hexdigest()
    return dict(asset=f'{model}-{pose_hash}-{side}', keypoints_pair_m=points.tolist(), arm=arm)


def write_bank(out, source, identity, records):
    with source.open("rb") as file:
        source_sha256 = hashlib.file_digest(file,"sha256").hexdigest()
    (out/"manifest.json").write_text(json.dumps(dict(
        source_sha256=source_sha256,identity=identity,records=records),indent=2)+"\n")


def scene_record(source, model, hands, **body):
    return dict(source, hands=hands, views={}, upper_body=dict(model=model, **body))


def hand_pose_plan(count, seed, mode):
    if mode == "prior":
        return [None] * count
    choices = HAND_GESTURES
    if mode == "mixed":
        choices += ("prior",) * len(HAND_GESTURES)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 24]))
    slots = []
    for _ in SIDES:
        draws = []
        while len(draws) < count:
            draws.extend(rng.permutation(choices).tolist())
        slots.append(draws[:count])
    return list(zip(*slots, strict=True))
