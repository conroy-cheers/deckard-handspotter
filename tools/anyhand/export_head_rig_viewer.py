"""Export a standalone Three.js inspection view of the synthetic Frame mount."""
import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from frame_camera import rays
from head_pose import NECK_PIVOT, baked_head_weights, look_angles
from humanoid_geometry import EYES_PAIR_M
from handspotter_cameras import QUAD_STREAMS, load_quad_rig, FrameHandspotterCamera


def encode(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def sensor_rim(view):
    directions, valid = rays(view.camera, view.circle, 256, 320)
    contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    return directions[contour[:, 1], contour[:, 0]].round(6).tolist()


def camera_data(rig, measured):
    source, _, cameras = load_quad_rig(rig)
    if not all(isinstance(camera, FrameHandspotterCamera) for camera in cameras):
        raise ValueError("The viewer requires a four-camera Frame KB4 rig")
    hardware = json.loads(measured.read_text())["cameras"]
    measured_transforms = [
        np.array(hardware[source[slot]["calibration"]["label"]]["parent_from_eye"]) for slot in QUAD_STREAMS
    ]
    midpoint = (measured_transforms[1][:3, 3] + measured_transforms[2][:3, 3]) / 2
    reference = np.linalg.inv(cameras[1].transform)
    measured_reference = np.linalg.inv(measured_transforms[1])
    if any(not np.allclose(reference @ view.transform, measured_reference @ factory, atol=1.e-5)
           for view,factory in zip(cameras,measured_transforms,strict=True)):
        raise ValueError("The rig and OpenVR comparison must use the same Frame calibration")
    result = []
    for slot, view, factory in zip(QUAD_STREAMS, cameras, measured_transforms, strict=True):
        aim, factory_aim = look_angles(view.transform[:3, 2]), look_angles(factory[:3, 2])
        factory_position = factory[:3, 3].copy()
        factory[:3, 3] -= midpoint
        result.append(dict(slot=slot, sensor=source[slot]["calibration"]["label"],
                           renderer=view.transform.tolist(), openvr=factory.tolist(),
                           factory_mm=(1000*factory_position).round(1).tolist(),
                           aim=aim.round(1).tolist(), openvr_aim=factory_aim.round(1).tolist(),
                           rim=sensor_rim(view)))
    return result


def body_data(bank, scene_ids):
    rows = json.loads((bank/"manifest.json").read_text())["records"]
    slots = {row["id"]: (index,row) for index,row in enumerate(rows)}
    result = []
    for name in scene_ids or [rows[0]["id"]]:
        index, row = slots[name]
        pose = row.get("head_pose", {})
        pivot = np.asarray(pose.get("pivot_reference_m", NECK_PIVOT))
        with np.load(bank/f"{index:05d}.npz", allow_pickle=False) as data:
            vertices, triangles, labels = (data[f"skin_{field}"] for field in ("vertices","faces","labels"))
            if "rest_vertices" not in data or "rest_arm_joints" not in data:
                raise ValueError("Bake the SOMA-X body bank with --cloth-reference to identify its arm surfaces")
            rest = data["rest_vertices"]
            shoulders_x = data["rest_arm_joints"][:, 0, 0]
        weights = baked_head_weights(vertices, triangles, labels, pivot)
        # Classify the arm in the rest T-pose so a lowered wrist keeps all of
        # its skin even when the torso is cropped for inspection.
        arm_vertices = (rest[:, 0] < shoulders_x.min()-.032) | (rest[:, 0] > shoulders_x.max()+.032)
        arm_faces = arm_vertices[triangles].any(axis=1) | (labels != 3)
        neutral_weights = baked_head_weights(rest, triangles, labels, pivot)
        neutral_weights[arm_vertices] = 0.
        torso = triangles[~arm_faces & (vertices[triangles,1].max(axis=1) > -.48)]
        cropped = np.concatenate((torso, triangles[arm_faces]))
        used, indices = np.unique(cropped, return_inverse=True)
        arms = {}
        for side, hand in row["hands"].items():
            arms[side] = {joint: hand["arm"][f"{joint}_pair_m"] for joint in ("shoulder","elbow","wrist")}
            arms[side]["fingers"] = hand["keypoints_pair_m"]
        result.append(dict(id=name, angles=pose.get("yaw_pitch_roll_deg",[0.,0.,0.]), pivot=pivot.tolist(),
                           positions=encode(vertices[used],"<f4"), indices=encode(indices,"<u4"),
                           torso_indices=3*len(torso), head_weights=encode(weights[used],"<f4"),
                           neutral_weights=encode(neutral_weights[used],"<f4"),
                           arm_vertices_with_head_weight=int(np.count_nonzero((weights > .5) & arm_vertices)),
                           arms=arms))
    return result


def javascript_url(source):
    return "data:text/javascript;base64," + base64.b64encode(source.encode()).decode("ascii")


def export(args):
    folder = Path(__file__).resolve().parent
    core = (args.three/"build/three.core.js").read_text()
    module = (args.three/"build/three.module.js").read_text().replace("./three.core.js", "three-core")
    orbit = (args.three/"examples/jsm/controls/OrbitControls.js").read_text()
    import_map = {"imports": {name: javascript_url(source) for name,source in
                              (("three-core",core),("three",module),("orbit-controls",orbit))}}
    data = dict(cameras=camera_data(args.rig_json,args.openvr_calibration),
                bodies=body_data(args.body_bank,args.scene), eyes=EYES_PAIR_M.tolist())
    html = (folder/"head_rig_viewer.html").read_text()
    contents = html.replace("HEAD_VIEWER_IMPORT_MAP",json.dumps(import_map,separators=(",",":")),1)
    contents = contents.replace("HEAD_VIEWER_DATA",json.dumps(data,separators=(",",":")),1)
    contents = contents.replace("HEAD_VIEWER_SCRIPT",(folder/"head_rig_viewer.mjs").read_text(),1)
    with args.out.open("x") as file:
        file.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("body-bank", "rig-json", "openvr-calibration", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--three",type=Path,default=os.environ.get("ANYHAND_THREE_JS"),
                        required="ANYHAND_THREE_JS" not in os.environ)
    parser.add_argument("--scene",action="append",help="Body-bank record ID. Repeat to compare different poses.")
    export(parser.parse_args())
