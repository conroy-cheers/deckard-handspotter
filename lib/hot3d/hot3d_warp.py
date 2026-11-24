"""Map the labeled HOT3D tracking pair into Frame's Handspotter front views."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from frame_camera import digest, rays
from calibration import quaternion_transform
from handspotter_cameras import Fisheye624Camera, FrameHandspotterCamera, QUAD_STREAMS, SIDES, load_quad_rig, pair_alignment
from hot3d_clip import TRACKING_STREAMS, open_clip


FRONT = slice(1, 3)
SUBSAMPLES = 2
LABEL_SCALE = 8 // SUBSAMPLES


@dataclass(frozen=True)
class ViewWarp:
    map_x: np.ndarray
    map_y: np.ndarray
    source_size: tuple
    covered_x: np.ndarray
    covered_y: np.ndarray
    covered_rows: np.ndarray
    covered_cols: np.ndarray
    target_support: int
    source_from_target: np.ndarray

    def pixels(self, image):
        if image.dtype != np.uint8 or image.shape[::-1] != self.source_size:
            raise ValueError("HOT3D pixels do not match their camera calibration")
        output = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        sums = output.reshape(128, SUBSAMPLES, 160, SUBSAMPLES).sum((1, 3), dtype=np.uint16)
        area = SUBSAMPLES**2
        return ((sums + area//2) // area).astype(np.uint8)

    def label(self, bounds, visibility):
        if bounds is None:
            return np.zeros(4, np.float32), True
        x0, y0, x1, y1 = np.asarray(bounds, dtype=np.float64) - .5
        matches = ((self.covered_x >= x0) & (self.covered_x < x1)
                   & (self.covered_y >= y0) & (self.covered_y < y1))
        support = int(matches.sum())
        if not support:
            return np.zeros(4, np.float32), True
        if visibility <= .1 or support * LABEL_SCALE**2 < 128:
            return np.zeros(4, np.float32), False
        row, col = self.covered_rows[matches], self.covered_cols[matches]
        return LABEL_SCALE * np.array((col.min(), row.min(), col.max()+1, row.max()+1), np.float32), True

    def provenance(self):
        return dict(source_size=self.source_size, source_from_target=self.source_from_target.tolist(),
                    fraction_of_frame_support=len(self.covered_rows)/self.target_support)


def build_warp(source, pair_from_eye, destination):
    lens = Fisheye624Camera(source["calibration"], pair_from_eye)
    camera = lens.camera
    source_from_target = pair_from_eye[:3, :3].T @ destination.transform[:3, :3]
    target_rays, frame_support = rays(destination.camera, destination.circle, destination.height, destination.width)
    source_rays = target_rays @ source_from_target.T
    angle = np.arctan2(np.linalg.norm(source_rays[..., :2], axis=-1), source_rays[..., 2])
    supported = frame_support & (angle <= lens.angular_limit)
    row, col = np.where(supported)
    u, v = camera.eye_to_window(source_rays[row, col]).T
    captured = ((u >= 0) & (u <= camera.width-1) & (v >= 0) & (v <= camera.height-1))
    row, col, u, v = (array[captured] for array in (row, col, u, v))
    if not len(row):
        raise ValueError("HOT3D and Frame have no common front-camera field of view")
    map_x = np.full(frame_support.shape, -1, np.float32)
    map_y = map_x.copy()
    map_x[row, col], map_y[row, col] = u, v
    return ViewWarp(map_x, map_y, (camera.width, camera.height), u, v, row, col,
                    int(frame_support.sum()), source_from_target)


def calibration_key(cameras):
    return json.dumps([cameras[stream]["calibration"] for stream in TRACKING_STREAMS], sort_keys=True)


@lru_cache(maxsize=4)
def target_pair(rig):
    _, _, target = load_quad_rig(rig, SUBSAMPLES)
    front = tuple(target[FRONT])
    if any(not isinstance(camera, FrameHandspotterCamera) for camera in front):
        raise ValueError("The destination must use calibrated Frame KB4 tracking cameras")
    return front


@lru_cache(maxsize=64)
def camera_warps(key, rig, targets=(0, 1)):
    cameras = json.loads(key)
    source_poses = [quaternion_transform(camera["T_device_from_camera"]) for camera in cameras]
    alignment = pair_alignment(*source_poses)
    destination_pair = target_pair(rig)
    return tuple(build_warp(
        dict(calibration=camera, T_world_from_camera=camera["T_device_from_camera"]),
        alignment @ pose, destination)
        for camera, pose, destination in zip(
            cameras, source_poses, (destination_pair[i] for i in targets), strict=True))


def prepare_clip(archive, rig, output, expected_hash=None, expected_device=None):
    """Write front bytes, warped boxes, and supervision masks for every labeled frame."""
    archive_hash = digest(archive)
    if expected_hash is not None and expected_hash != archive_hash:
        raise ValueError(f"HOT3D archive hash mismatch at {archive}")
    with open_clip(archive) as clip:
        frames = clip.frame_ids()
        if len(frames) != 150 or len(set(frames)) != len(frames):
            raise ValueError(f"Expected 150 distinct HOT3D frames in {archive}")
        image = np.empty((len(frames), 2, 128, 160), dtype=np.uint8)
        label = np.zeros((len(frames), 4, 2, 4), dtype=np.float32)
        valid = np.zeros((len(frames), 4, 2), dtype=bool)
        profiles = {}
        for index, frame in enumerate(frames):
            info = clip.json(f"{frame:06d}.info.json")
            if index == 0:
                device, sequence, participant = (info[name] for name in ("device", "sequence_id", "participant_id"))
                if expected_device is not None and expected_device.casefold() != device.casefold():
                    raise ValueError(f"Unexpected source device {device} in {archive}")
            elif (info["device"], info["sequence_id"], info["participant_id"]) != (device, sequence, participant):
                raise ValueError(f"Mixed source sequences in {archive}")
            cameras = clip.json(f"{frame:06d}.cameras.json")
            hands = clip.json(f"{frame:06d}.hands.json")
            key = calibration_key(cameras)
            warps = camera_warps(key, rig)
            if key not in profiles:
                profiles[key] = [view.provenance() for view in warps]
            for eye, (stream, warp) in enumerate(zip(TRACKING_STREAMS, warps, strict=True)):
                image[index, eye] = warp.pixels(clip.pixels(frame, stream))
                for hand, side in enumerate(SIDES):
                    annotation = hands.get(side, {})
                    label[index, eye+1, hand], valid[index, eye+1, hand] = warp.label(
                        annotation.get("boxes_amodal", {}).get(stream),
                        annotation.get("visibilities_modeled", {}).get(stream, 0.))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, data in (("front", image), ("boxes", label), ("valid", valid)):
            np.save(stage/f"{name}.npy", data)
        record = dict(schema=1, rig_sha256=digest(rig), archive_sha256=archive_hash,
                      hashes={name: digest(stage/f"{name}.npy") for name in ("front", "boxes", "valid")},
                      stream_order=list(QUAD_STREAMS), frames=frames, device=device,
                      sequence=sequence, participant=participant, source_warps=list(profiles.values()),
                      positive_boxes=int(((label[..., 2:] > label[..., :2]).all(axis=-1) & valid).sum()),
                      supervised_slots=int(valid.sum()))
        (stage/"clip.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        stage.rename(output)
    finally:
        if stage.is_dir():
            shutil.rmtree(stage)
    return record
