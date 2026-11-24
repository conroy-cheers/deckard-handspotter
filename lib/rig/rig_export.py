"""Export the renderer's native four-camera rig from a Frame's calibration."""
import json

import numpy as np

from calibration import load_frame
from handspotter import ORDER
from handspotter_data import QUAD_STREAMS
from handspotter_shards import digest


def export_rig(xrservice, device, output):
    cameras = load_frame(xrservice, device)
    records = json.loads(xrservice.read_text())["cameras"]
    circles = {row["sourceCamera"]: row["imageCircleDiameterInDeg"]
               for row in records if row["sourceCamera"] in ORDER}
    left, right = cameras["slam_left"], cameras["slam_right"]
    lateral = right.origin - left.origin
    lateral /= np.linalg.norm(lateral)
    forward = left.transform[:3, 2] + right.transform[:3, 2]
    forward -= lateral * np.dot(forward, lateral)
    forward /= np.linalg.norm(forward)
    alignment = np.eye(4)
    alignment[:3, :3] = [lateral, np.cross(lateral, forward), -forward]
    alignment[:3, 3] = -alignment[:3, :3] @ ((left.origin + right.origin) / 2)
    if not np.isfinite(alignment).all():
        raise ValueError("Frame front cameras cannot define a stereo basis")

    result = {}
    for slot, name in zip(QUAD_STREAMS, ORDER, strict=True):
        camera = cameras[name]
        result[slot] = {
            "source_calibration": {
                "label": name,
                "image_width": camera.width,
                "image_height": camera.height,
                "projection_model_type": "kb4",
                "intrinsics": camera.intrinsics.tolist(),
                "distortion": camera.distortion.tolist(),
                "image_circle_degrees": circles[name],
            },
            "pair_from_eye": (alignment @ camera.transform).tolist(),
        }

    with output.open("x") as stream:
        json.dump({
            "stream_order": QUAD_STREAMS,
            "cameras": result,
            "evidence": {"frame_calibration_sha256": digest(xrservice),
                         "frame_device_sha256": digest(device)},
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
