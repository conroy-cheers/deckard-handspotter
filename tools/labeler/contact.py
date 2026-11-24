"""Review ambiguous hand contact using both front views and temporal context."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import batched
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

from calibration import load_frame
from handedness import hand_rays, overlap, stereo_pair
from terra import CADENCE, MODEL, decode, request_json, response_schema
from workspace import CAMERAS, HANDS, image_name, labels_digest, read_json, write_json

FRONT = CAMERAS[:2]
PROMPT = """Review touching or nearly touching hands of the Steam Frame wearer.
The camera views are partially rotated upright with the headset calibration.
Images with the same group number are simultaneous. Earlier and later groups
show the same wearer. Trace each visible hand back through its wrist and forearm
to its shoulder. Follow distinguishing features such as a wristband through
time and between views. Use the anatomy and arm attachment to determine
anatomical left and right. The apparent screen side can change.

For each requested group and camera, box the visible pixels of each hand
from wrist crease to fingertips, including thumb. Exclude forearm and wristband.
Boxes may overlap when hands touch. Keep each hand's boundary, including only
its own visible pixels. Use 'absent' when a hand is definitely out of view or
fully hidden. Use 'unknown' when there is insufficient evidence to localize
the hand or distinguish the two hands. Coordinates are integer [x1,y1,x2,y2]
normalized 0–1000 to the attached image, x right and y down.
Return only the requested JSON. Do not call tools.
"""


def flagged_groups(workspace, labels, cameras):
    for name in FRONT:
        camera = cameras[name]
        size = workspace.manifest["cameras"][name]
        if (camera.width, camera.height) != (size["width"], size["height"]):
            raise ValueError(f"{name} calibration dimensions differ from the recording")
    flags = []
    for index, group in enumerate(workspace.groups):
        sequence = group["sequence"]
        boxes = [[labels.get(workspace.key(sequence, camera, hand), {}).get("box_xyxy")
                  for hand in HANDS] for camera in FRONT]
        if max(overlap(*pair) for pair in boxes) < 0.35:
            continue
        if not all(boxes[0] + boxes[1]):
            flags.append(index)
            continue
        first, second = [hand_rays(cameras[name], pair) for name, pair in zip(FRONT, boxes)]
        same = stereo_pair(first[0], second[0])[0] + stereo_pair(first[1], second[1])[0]
        swapped = stereo_pair(first[0], second[1])[0] + stereo_pair(first[1], second[0])[0]
        if same - swapped < 0.48:
            flags.append(index)
    return flags


def review_spans(flags, length):
    spans = []
    for index in flags:
        begin, end = max(0, index - 2), min(length, index + 3)
        if spans and begin <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([begin, end])
    return spans


def rotate_box(box, turns):
    x1, y1, x2, y2 = box
    for _ in range(turns % 4):
        x1, y1, x2, y2 = y1, 1000 - x2, y2, 1000 - x1
    return [x1, y1, x2, y2]


def request_contact(codex, workspace, begin, end, turns):
    # Each request sees the frames immediately before and after its targets.
    # Quarter-turns retain every pixel and map axis-aligned boxes exactly.
    groups = workspace.groups
    targets = [group["sequence"] for group in groups[begin:end]]
    context = [group["sequence"] for group in groups[max(0, begin-1):end+1]]
    prompt = PROMPT + "\nAttached group order: " + ", ".join(map(str, context))
    prompt += "\nWithin each group, first SLAM left, then SLAM right."
    prompt += "\nReturn labels only for these groups: " + ", ".join(map(str, targets))
    with tempfile.TemporaryDirectory(prefix="frame-terra-contact-") as temp:
        directory = Path(temp)
        images = []
        rotations = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]
        for sequence in context:
            for camera in FRONT:
                original = workspace.root / image_name(sequence, camera)
                dest = directory / f"{sequence}-{camera}.png"
                with Image.open(original) as image:
                    rotated = rotations[turns[camera]]
                    (image.transpose(rotated) if rotated is not None else image).save(dest)
                images.append(dest)
        result = request_json(codex, directory, response_schema(targets, FRONT), images, prompt, effort="low")
        labels = decode(workspace, targets, result, FRONT,
                        box_transform=lambda camera, box: rotate_box(box, -turns[camera]))
    return {key: dict(label, source="terra_contact") for key, label in labels.items()}


def review_contacts(workspace, xrservice, device, workers, *, plan_only=False):
    if workspace.manifest.get("sample_hz") != CADENCE or not 1 <= workers <= 8:
        raise ValueError("Use a 10 Hz workspace and between one and eight review workers")
    baseline = workspace.load_terra(contact=False)
    if not baseline:
        raise ValueError("Run the initial Terra review before reviewing hand contact")
    cameras = load_frame(xrservice, device)
    flags = flagged_groups(workspace, baseline, cameras)
    spans = review_spans(flags, len(workspace.groups))
    groups = sum(end-begin for begin, end in spans)
    print(f"Flagged {len(flags)} frame groups in {len(spans)} spans covering {groups} groups", flush=True)
    if plan_only or not spans:
        return
    codex = shutil.which("codex")
    if codex is None:
        raise ValueError("Install Codex CLI and sign in with ChatGPT before running Terra review")

    turns = {}
    for name in FRONT:
        up = cameras[name].transform[1, :2]
        angle = math.atan2(up[1], up[0]) + math.pi / 2
        turns[name] = round(angle / (math.pi / 2)) % 4
    jobs = [(offset, min(offset+5, end)) for begin, end in spans for offset in range(begin, end, 5)]
    plan = dict(version=1, model=MODEL, source_sha256=labels_digest(baseline),
                calibration=[sha256(path.read_bytes()).hexdigest() for path in (xrservice, device)],
                flags=[workspace.groups[i]["sequence"] for i in flags],
                spans=[dict(first_frame=begin+1, last_frame=end) for begin, end in spans],
                jobs=[list(job) for job in jobs])
    path = workspace.root / "terra-contact-review.json"
    state = read_json(path, {})
    if state.get("plan") != plan:
        state = dict(plan=plan, completed=[], complete=False, labels={})
    pending = [(i, begin, end) for i, (begin, end) in enumerate(jobs) if i not in state["completed"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for wave in batched(pending, workers):
            tasks = [(i, begin, end, pool.submit(request_contact, codex, workspace, begin, end, turns))
                     for i, begin, end in wave]
            errors = []
            for i, begin, end, future in tasks:
                try:
                    state["labels"].update(future.result())
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    errors.append(f"Frames {begin+1}–{end}: {error}")
                    continue
                state["completed"].append(i)
                write_json(path, state)
                print(f"Reviewed frames {begin+1}–{end} with {MODEL}", flush=True)
            if errors:
                raise ValueError("\n".join(errors))
    state["complete"] = True
    write_json(path, state)
    unresolved = sum(label["status"] == "unknown" for label in state["labels"].values())
    print(f"Saved {len(state['labels'])} contact labels, including {unresolved} unknown", flush=True)
