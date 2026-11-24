"""Import native take pixels and keep reviewed labels separate from proposals."""

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CAMERAS = ("slam_left", "slam_right", "upper_left", "upper_right")
HANDS = ("left", "right")
SCHEMA = 1


def read_json(path, default=None):
    if default is not None and not path.exists():
        return default
    return json.loads(path.read_text())


def labels_digest(labels):
    return hashlib.sha256(json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            json.dump(value, handle, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def read_groups(path):
    groups = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = int(row["group"])
            if not groups or sequence != groups[-1]["sequence"]:
                if groups and sequence < groups[-1]["sequence"]:
                    raise ValueError("Camera group sequence went backwards")
                groups.append(dict(sequence=sequence, views={}))
            views = groups[-1]["views"]
            camera = row["camera"]
            if camera not in CAMERAS or camera in views:
                raise ValueError(f"Invalid camera {camera!r} in sequence {sequence}")
            views[camera] = {
                "track": int(row["track"]),
                "pts_us": int(row["pts_us"]),
                **{field: float(row[field]) for field in (
                    "sensor_raw_s", "end_exposure_raw_s", "arrival_raw_s", "exposure_s")},
                **{field: int(row[field]) for field in (
                    "observer_mono_ns", "clock_raw_ns", "clock_mono_ns", "clock_uncertainty_ns")},
            }
    if not groups:
        raise ValueError("The take contains no frame groups")
    for group in groups:
        if set(group["views"]) != set(CAMERAS):
            raise ValueError(f"Incomplete camera group {group['sequence']}")
        group["time_us"] = min(view["pts_us"] for view in group["views"].values())
    return groups


def image_name(sequence, camera):
    return f"frames/{camera}/{sequence:012d}.png"


def sample_by_time(groups, begin, hz):
    by_tick = {}
    for group in groups:
        offset = (group["time_us"] - begin) * hz
        tick = math.floor(offset / 1_000_000 + 0.5)
        distance = abs(offset - tick * 1_000_000)
        if tick not in by_tick or distance < by_tick[tick][0]:
            by_tick[tick] = (distance, group)
    return [group for _, group in by_tick.values()]


def select_groups(groups, start, duration, every, hz):
    if not math.isfinite(start) or start < 0 or (
            duration is not None and (not math.isfinite(duration) or duration <= 0)) or every < 1:
        raise ValueError("Use a nonnegative start, positive duration and positive frame step")
    if hz is not None and (not math.isfinite(hz) or hz <= 0 or every != 1):
        raise ValueError("Use a positive cadence without a frame step")
    begin = round(start * 1e6)
    end = None if duration is None else begin + round(duration * 1e6)
    interval = [group for group in groups
                if group["time_us"] >= begin and (end is None or group["time_us"] < end)]
    selected = interval[::every] if hz is None else sample_by_time(interval, begin, hz)
    if not selected:
        raise ValueError("No complete camera groups in the requested interval")
    return selected


def ingest(take, output, start, duration, every, *, hz=None):
    import av
    from PIL import Image

    if output.exists():
        raise ValueError(f"Workspace already exists: {output}")
    csv_path = take / "timestamps.csv"
    groups = read_groups(csv_path)
    selected = select_groups(groups, start, duration, every, hz)
    chosen = {group["sequence"] for group in selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".frame-labels-") as temp:
        workspace = Path(temp) / "workspace"
        workspace.mkdir(mode=0o700)
        with av.open(str(take / "tracking.nut")) as video:
            streams = {s.metadata.get("title"): s for s in video.streams.video}
            if len(video.streams.video) != len(CAMERAS) or set(streams) != set(CAMERAS):
                raise ValueError("Expected four named Frame video tracks")
            by_track = {stream.index: name for name, stream in streams.items()}
            layout = {}
            for name, stream in streams.items():
                ctx = stream.codec_context
                if ctx.name != "rawvideo" or ctx.pix_fmt != "gray":
                    raise ValueError(f"{name} must contain native grayscale rawvideo packets")
                layout[name] = dict(width=ctx.width, height=ctx.height)
                (workspace / "frames" / name).mkdir(parents=True)
            cursor = dict.fromkeys(CAMERAS, 0)
            previous_pts = dict.fromkeys(CAMERAS, -1)
            extracted = 0
            for packet in video.demux(*streams.values()):
                if not packet.size:
                    continue
                name = by_track[packet.stream.index]
                index = cursor[name]
                cursor[name] += 1
                if index >= len(groups):
                    raise ValueError(f"{name} contains more packets than the timestamp CSV")
                group = groups[index]
                row = group["views"][name]
                if packet.pts is None or packet.time_base is None or packet.stream.index != row["track"]:
                    raise ValueError(f"Unidentified packet in {name}, group {group['sequence']}")
                packet_us = round(packet.pts * packet.time_base * 1_000_000)
                # The take stores sensor-derived microseconds in the sidecar.
                # The muxer can round to the nearest tick instead of truncating.
                expected_us = max(row["pts_us"], previous_pts[name] + 1)
                if abs(packet_us - expected_us) > 2:
                    raise ValueError(f"Video/CSV PTS mismatch in {name}, group {group['sequence']}: "
                                     f"{packet_us} vs {row['pts_us']} us")
                previous_pts[name] = packet_us
                if group["sequence"] not in chosen:
                    continue
                decoded = packet.decode()
                if len(decoded) != 1:
                    raise ValueError(f"Expected one raw frame per packet in {name}")
                frame = decoded[0]
                plane = frame.planes[0]
                image = Image.frombytes("L", (frame.width, frame.height), bytes(plane),
                                        "raw", "L", plane.line_size, 1)
                image.save(workspace / image_name(group["sequence"], name))
                extracted += 1
                if extracted == len(selected) * len(CAMERAS) and selected[-1] is not groups[-1]:
                    break
            if extracted != len(selected) * len(CAMERAS):
                raise ValueError(f"Found only {extracted} of {len(selected) * len(CAMERAS)} video frames")
        manifest = dict(schema=SCHEMA, take=take.name,
                        timestamps_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        cameras=layout, groups=selected)
        if hz is not None:
            manifest["sample_hz"] = hz
        write_json(workspace / "manifest.json", manifest)
        workspace.rename(output)
    return len(selected)


class Workspace:
    def __init__(self, root):
        self.root = root
        self.manifest = read_json(root / "manifest.json")
        source = self.manifest.get("source", {})
        self.is_hot3d = source.get("kind") == "hot3d"
        cameras = set(self.manifest["cameras"])
        self.cameras = tuple(camera for camera in CAMERAS if camera in cameras)
        if (self.manifest.get("schema") != SCHEMA or not cameras or not cameras <= set(CAMERAS)
                or (not self.is_hot3d and cameras != set(CAMERAS))):
            raise ValueError("Unsupported Frame labeling workspace")
        self.groups = self.manifest["groups"]
        if self.is_hot3d and any(set(group["views"]) != cameras for group in self.groups):
            raise ValueError("Incomplete HOT3D camera group")
        self.sequences = {group["sequence"] for group in self.groups}

    @staticmethod
    def key(sequence, camera, hand):
        return f"{sequence}/{camera}/{hand}"

    def load_labels(self):
        return read_json(self.root / "reviewed.json", {})

    def load_proposals(self):
        return read_json(self.root / "proposals.json", {"boxes": {}})["boxes"]

    def load_terra(self, *, corrected=True, contact=True):
        path = self.root / "terra-review.json"
        try:
            source = path.read_bytes()
        except FileNotFoundError:
            return {}
        labels = json.loads(source)
        if corrected:
            result = read_json(self.root / "terra-handedness.json", {})
            if result.get("source_sha256") == hashlib.sha256(source).hexdigest():
                labels.update(result["labels"])
            if contact:
                review = read_json(self.root / "terra-contact-review.json", {})
                if review.get("complete") and review["plan"]["source_sha256"] == labels_digest(labels):
                    labels.update(review["labels"])
        return labels

    def set_label(self, sequence, camera, hand, action, box=None):
        return self.edit_label(sequence, camera, hand, action, box)[1]

    def edit_label(self, sequence, camera, hand, action, box=None):
        with (self.root / ".reviewed.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._set_label(sequence, camera, hand, action, box)

    def _set_label(self, sequence, camera, hand, action, box):
        if type(sequence) is not int or sequence not in self.sequences or camera not in self.cameras or hand not in HANDS:
            raise ValueError("Unknown camera, hand or frame group")
        key = self.key(sequence, camera, hand)
        reviewed = self.load_labels()
        previous = reviewed.get(key)
        if action == "unknown":
            reviewed[key] = {"status": "unknown"}
        elif action == "absent":
            reviewed[key] = {"status": "absent"}
        elif action == "accept_terra":
            if key in reviewed:
                raise ValueError("This hand has already been reviewed")
            terra = self.load_terra().get(key)
            if terra is None or terra["status"] not in ("box", "absent") or box != terra.get("box_xyxy"):
                raise ValueError("This Terra proposal changed. Reload labels before reviewing it.")
            reviewed[key] = dict(terra, source="reviewed_terra")
        elif action == "accept":
            if key in reviewed:
                raise ValueError("This hand has already been reviewed")
            proposals = read_json(self.root / "proposals.json", {"boxes": {}})
            proposal = proposals["boxes"].get(key)
            if proposal is None:
                raise ValueError("No proposal to accept")
            if box != proposal["box_xyxy"]:
                raise ValueError("This proposal changed. Reload proposals before reviewing it.")
            reviewed[key] = {"status": "box", "source": "reviewed_sam", "seed": proposal["seed"],
                             "box_xyxy": proposal["box_xyxy"]}
            model = proposal.get("model") or proposals.get("models", {}).get(camera)
            if model:
                reviewed[key]["model"] = model
        elif action == "box":
            shape = self.manifest["cameras"][camera]
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) for v in box)
                    or not (0 <= box[0] < box[2] <= shape["width"])
                    or not (0 <= box[1] < box[3] <= shape["height"])):
                raise ValueError("Drag a nonempty box inside the image")
            reviewed[key] = {"status": "box", "source": "manual", "box_xyxy": box}
        else:
            raise ValueError("Unknown label action")
        write_json(self.root / "reviewed.json", reviewed)
        return previous, reviewed[key]

    def undo_edit(self, key, previous, expected):
        with (self.root / ".reviewed.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reviewed = self.load_labels()
            if reviewed.get(key) != expected:
                raise ValueError("This label changed. Reload labels before editing it.")
            if previous is None:
                del reviewed[key]
            else:
                reviewed[key] = previous
            write_json(self.root / "reviewed.json", reviewed)

    def export(self, output, *, include_terra=False):
        return self.export_counts(output, include_terra=include_terra)[1:]

    def export_counts(self, output, *, include_terra=False):
        grouped_boxes = 0
        reviewed = self.load_labels()
        if include_terra:
            reviewed = self.load_terra() | reviewed
        coco = dict(images=[], annotations=[], categories=[
            dict(id=i, name=f"{hand}_hand") for i, hand in enumerate(HANDS, 1)
        ])
        groups = []
        for group in self.groups:
            sequence = group["sequence"]
            views = {}
            for camera in self.cameras:
                hands = {hand: reviewed.get(self.key(sequence, camera, hand), {"status": "unknown"})
                         for hand in HANDS}
                grouped_boxes += sum(label["status"] == "box" for label in hands.values())
                img = image_name(sequence, camera)
                meta = self.manifest["cameras"][camera]
                views[camera] = dict(file_name=img, **meta, **group["views"][camera], hands=hands)
                # COCO has no per-class "unknown" state for an otherwise labeled image.
                if any(label["status"] == "unknown" for label in hands.values()):
                    continue
                image_id = len(coco["images"]) + 1
                coco["images"].append(dict(id=image_id, file_name=img, camera=camera,
                                           sequence=sequence, **meta))
                for i, hand in enumerate(HANDS, 1):
                    if hands[hand]["status"] == "box":
                        x, y, x2, y2 = hands[hand]["box_xyxy"]
                        w, h = x2 - x, y2 - y
                        annotation = dict(id=len(coco["annotations"]) + 1, image_id=image_id,
                                          category_id=i, bbox=[x, y, w, h], area=w*h, iscrowd=0)
                        if hands[hand].get("box_mode") == "amodal":
                            annotation["box_mode"] = "amodal"
                        coco["annotations"].append(annotation)
            exported = dict(sequence=sequence, time_us=group["time_us"], views=views)
            if self.is_hot3d:
                exported["source_timestamp_ns"] = group["source_timestamp_ns"]
            groups.append(exported)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "coco.json", coco)
        grouped = dict(schema=SCHEMA, take=self.manifest["take"],
                       timestamps_sha256=self.manifest["timestamps_sha256"], groups=groups)
        if self.is_hot3d:
            grouped["source"] = self.manifest["source"]
        write_json(output / "groups.json", grouped)
        return grouped_boxes, len(coco["images"]), len(coco["annotations"])
