"""Ingest HOT3D-Clips into a Frame labeler workspace."""
import math
from pathlib import Path
import tarfile
import tempfile

from PIL import Image

from frame_camera import digest
from hot3d_clip import TRACKING_STREAMS, open_clip
from hot3d_warp import calibration_key, camera_warps
from workspace import CAMERAS, HANDS, SCHEMA, Workspace, image_name, labels_digest, select_groups, write_json


def channel_mapping(channels, rig):
    mapping = dict(zip(TRACKING_STREAMS, CAMERAS[:2], strict=True))
    if channels:
        pairs = [argument.split("=", 1) for argument in channels]
        if any(len(pair) != 2 or not pair[0] or pair[1] not in CAMERAS for pair in pairs):
            raise ValueError("Use --channel-map SOURCE=FRAME_CAMERA")
        mapping = dict(pairs)
        if len(mapping) != len(pairs) or len(set(mapping.values())) != len(mapping):
            raise ValueError("Map each source stream and destination camera at most once")
    if rig and (not mapping.keys() <= set(TRACKING_STREAMS) or not set(mapping.values()) <= set(CAMERAS[:2])):
        raise ValueError("A HOT3D Frame warp maps tracking streams to slam_left or slam_right")
    return mapping


def read_groups(clip, mapping):
    groups = []
    origin = previous = None
    identity = None
    for frame in clip.frame_ids():
        info = clip.json(f"{frame:06d}.info.json")
        clip_id = {field: info[field] for field in ("device", "sequence_id", "participant_id")}
        timestamp = info["ref_timestamp_ns"]
        if origin is None:
            origin, identity = timestamp, clip_id
        if clip_id != identity or (previous is not None and timestamp <= previous):
            raise ValueError("HOT3D frame timestamps must increase within one source sequence")
        time_us = round((timestamp-origin)/1000)
        views = {}
        for stream, camera in mapping.items():
            capture_ns = info["image_timestamps_ns"][stream]
            views[camera] = dict(source_stream=stream, source_timestamp_ns=capture_ns,
                                 pts_us=round((capture_ns-origin)/1000))
        groups.append(dict(sequence=frame, time_us=time_us, source_timestamp_ns=timestamp, views=views))
        previous = timestamp
    if not groups or len({g["sequence"] for g in groups}) != len(groups):
        raise ValueError("Expected distinct HOT3D frames with timestamps")
    return groups, identity


def imported_label(hands, side, stream, width, height, warp):
    result = {"source": "hot3d"}
    if hands is None:
        return dict(result, status="unknown")
    annotation = hands.get(side, {})
    bounds = annotation.get("boxes_amodal", {}).get(stream)
    visibility = annotation.get("visibilities_modeled", {}).get(stream, 0.)
    if bounds is None:
        return dict(result, status="absent")
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds) or not math.isfinite(visibility):
        raise ValueError(f"Invalid HOT3D label for {stream}/{side}")
    if warp:
        # Training stores boxes in the 1280x1024 reference grid. The editor
        # and COCO use the actual 160x128 PNG's pixel coordinates.
        box, valid = warp.label(bounds, visibility)
        if not valid:
            return dict(result, status="unknown")
        x1, y1, x2, y2 = (box/8).tolist()
    else:
        x1, y1, x2, y2 = bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return dict(result, status="absent")
    if visibility <= .1:
        return dict(result, status="unknown")
    return dict(result, status="box", box_xyxy=[x1, y1, x2, y2], box_mode="amodal")


def ingest_hot3d(archive, output, start, duration, every, *, hz=None, channels=None, rig=None):
    if output.exists():
        raise ValueError(f"Workspace already exists: {output}")
    mapping = channel_mapping(channels, rig)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_clip(archive) as clip, tempfile.TemporaryDirectory(dir=output.parent, prefix=".frame-labels-") as temp:
            groups, identity = read_groups(clip, mapping)
            selected = select_groups(groups, start, duration, every, hz)
            members = set(clip.archive.getnames())
            workspace = Path(temp)/"workspace"
            workspace.mkdir(mode=0o700)
            for camera in mapping.values():
                (workspace/"frames"/camera).mkdir(parents=True)
            layout, labels = {}, {}
            targets = tuple(CAMERAS.index(mapping.get(stream, CAMERAS[i]))
                            for i, stream in enumerate(TRACKING_STREAMS))
            for group in selected:
                frame = group["sequence"]
                cameras = clip.json(f"{frame:06d}.cameras.json")
                annotated = f"{frame:06d}.hands.json"
                hands = clip.json(annotated) if annotated in members else None
                warps = {}
                if rig:
                    warps = dict(zip(TRACKING_STREAMS,
                                     camera_warps(calibration_key(cameras), rig, targets), strict=True))
                for stream, camera in mapping.items():
                    pixels = clip.pixels(frame, stream)
                    if pixels is None:
                        raise ValueError(f"Invalid image for {stream}, frame {frame}")
                    calibration = cameras[stream]["calibration"]
                    if pixels.shape != (calibration["image_height"], calibration["image_width"]):
                        raise ValueError(f"HOT3D image size differs from calibration for {stream}, frame {frame}")
                    warp = warps.get(stream)
                    if warp:
                        pixels = warp.pixels(pixels)
                    height, width = pixels.shape
                    size = dict(width=width, height=height)
                    if layout.setdefault(camera, size) != size:
                        raise ValueError(f"HOT3D stream {stream} changes image size within the workspace")
                    Image.fromarray(pixels).save(workspace/image_name(frame, camera))
                    for side in HANDS:
                        labels[Workspace.key(frame, camera, side)] = imported_label(
                            hands, side, stream, width, height, warp)
            source = dict(kind="hot3d", archive_sha256=digest(archive), channel_map=mapping, **identity)
            if rig:
                source["warp_rig_sha256"] = digest(rig)
            stamps = [(group["sequence"], group["source_timestamp_ns"]) for group in groups]
            manifest = dict(schema=SCHEMA, take=archive.stem, timestamps_sha256=labels_digest(stamps),
                            source=source, cameras=layout, groups=selected)
            if hz is not None:
                manifest["sample_hz"] = hz
            write_json(workspace/"manifest.json", manifest)
            write_json(workspace/"reviewed.json", labels)
            workspace.rename(output)
    except (tarfile.TarError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid HOT3D archive {archive}: {error}") from error
    return len(selected)
