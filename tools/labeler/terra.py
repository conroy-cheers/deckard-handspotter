"""Review synchronized camera groups using GPT-6.1 Terra through Codex."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from itertools import batched
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from workspace import CAMERAS, HANDS, image_name, read_json, sample_by_time, write_json

MODEL = "gpt-6.1-terra"
CADENCE = 10
PROMPT = """Label the headset wearer's anatomical left and right hands in these
headset tracking camera images. Use only the provided images. The camera
views in each group are simultaneous. Raw cameras can be sideways or upside
down. Identify anatomical handedness from the thumb, fingers, arm and context.
Do not use the screen half as a substitute for anatomical handedness.

For each camera and hand, return a tight axis-aligned box around visible hand
pixels from wrist crease to fingertips, including the thumb. Exclude forearm,
clothes, held objects, shadows and reflections. For hands cut off by the image
edge or an object, bound just the visible part. Image coordinates are normalized
integers from 0 to 1000 on each axis, with x increasing right and y increasing
down from the supplied image's top-left corner. Use [x1,y1,x2,y2]. Use 'absent'
for a hand definitely not visible in that view, and 'unknown' when visibility,
handedness or the box cannot be determined reliably. Report each image on its
own pixel axes. Check each camera for faint or partially visible
hands, and check the evidence separately for each frame. Return only the
requested JSON. Do not call tools.
"""


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def response_schema(sequences, cameras=CAMERAS):
    hand = {"anyOf": [
        {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
         "minItems": 4, "maxItems": 4},
        {"type": "string", "enum": ["absent", "unknown"]},
    ]}
    views = object_schema({camera: object_schema(dict.fromkeys(HANDS, hand)) for camera in cameras})
    return object_schema({str(sequence): views for sequence in sequences})


def decode(workspace, sequences, result, cameras=CAMERAS, *, box_transform=None):
    if not isinstance(result, dict) or set(result) != set(map(str, sequences)):
        raise ValueError("Terra response has the wrong frame groups")
    labels = {}
    for sequence in sequences:
        group = result[str(sequence)]
        if not isinstance(group, dict) or set(group) != set(cameras):
            raise ValueError(f"Terra response has the wrong cameras in group {sequence}")
        for camera in cameras:
            values = group[camera]
            if not isinstance(values, dict) or set(values) != set(HANDS):
                raise ValueError(f"Terra response has the wrong hands in group {sequence}, {camera}")
            shape = workspace.manifest["cameras"][camera]
            for hand, box in values.items():
                key = workspace.key(sequence, camera, hand)
                label = {"source": "terra", "model": MODEL}
                if box in ("absent", "unknown"):
                    label["status"] = box
                elif (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
                      and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                    x1, y1, x2, y2 = box_transform(camera, box) if box_transform else box
                    width, height = shape["width"], shape["height"]
                    label.update(status="box", box_xyxy=[
                        math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
                        math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000),
                    ])
                else:
                    raise ValueError(f"Invalid Terra box for {key}")
                labels[key] = label
    return labels


def request_json(codex, directory, schema, images, prompt, *, effort="none"):
    # Codex uses its existing login. Keep its working directory and output
    # schema isolated from the repository and attach only the selected views.
    schema_path = directory / "schema.json"
    answer = directory / "answer.json"
    write_json(schema_path, schema)
    process = subprocess.run([
        codex, "exec", "--ignore-user-config", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--disable", "shell_tool", "--disable", "multi_agent_v2",
        "-c", 'approval_policy="never"', "-c", f'model_reasoning_effort="{effort}"',
        "-m", MODEL, "--output-schema", str(schema_path), "-o", str(answer),
        "--image", ",".join(map(str, images)), "-",
    ], cwd=directory, input=prompt, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=150, check=False)
    if process.returncode or not answer.is_file():
        errors = [line for line in process.stderr.splitlines() if "error" in line.lower()]
        raise ValueError("Codex Terra request failed. " + " ".join(errors[-2:])[:900])
    return read_json(answer)


def request(codex, workspace, sequences):
    with tempfile.TemporaryDirectory(prefix="frame-terra-") as temp:
        images = [(workspace.root / image_name(seq, camera)).resolve()
                  for seq in sequences for camera in workspace.cameras]
        prompt = PROMPT + "\nGroup order: " + ", ".join(map(str, sequences))
        prompt += "\nWithin each group, image order: " + ", ".join(workspace.cameras) + "."
        result = request_json(codex, Path(temp), response_schema(sequences, workspace.cameras), images, prompt)
        return decode(workspace, sequences, result, workspace.cameras)


def review_selection(workspace, start, limit, hz):
    if not 0 <= start < len(workspace.groups) or (limit is not None and limit < 1):
        raise ValueError("Use a valid first group and a positive group count")
    groups = workspace.groups[start:] if limit is None else workspace.groups[start:start + limit]
    if hz is None:
        if workspace.manifest.get("sample_hz") != CADENCE:
            raise ValueError("Specify --hz to sample this workspace for Terra review")
        return groups
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("Use a positive frame rate")
    return sample_by_time(groups, groups[0]["time_us"], hz)


def review(workspace, start, limit, chunk, workers, *, hz=None):
    if not 1 <= chunk <= 8 or not 1 <= workers <= 8:
        raise ValueError("Use 1–8 groups per request and 1–8 workers")
    groups = review_selection(workspace, start, limit, hz)
    codex = shutil.which("codex")
    if codex is None:
        raise ValueError("Install Codex CLI and sign in with ChatGPT before running Terra review")
    with review_lock(workspace):
        return review_groups(codex, workspace, groups, chunk, workers)


@contextmanager
def review_lock(workspace):
    with (workspace.root / ".terra-review.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A Terra review is already running for this workspace") from error
        yield


def review_groups(codex, workspace, groups, chunk, workers):
    destination = workspace.root / "terra-review.json"
    labels = workspace.load_terra(corrected=False)
    pending = [group["sequence"] for group in groups if not all(
        workspace.key(group["sequence"], camera, hand) in labels for camera in workspace.cameras for hand in HANDS)]
    if not pending:
        return 0
    print(f"Reviewing {len(pending)} frame groups with {MODEL}", flush=True)
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Keep a bounded number of requests running so a quota or auth error
        # stops the run instead of launching more requests for every frame.
        for wave in batched(batched(pending, chunk), workers):
            futures = [(sequences, pool.submit(request, codex, workspace, sequences)) for sequences in wave]
            for sequences, future in futures:
                try:
                    labels.update(future.result())
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    failures.append(f"Groups {sequences}: {error}")
                    continue
                write_json(destination, labels)
                completed += len(sequences)
                print(f"Reviewed {completed}/{len(pending)} frame groups this run", flush=True)
            if failures:
                raise ValueError("\n".join(failures))
    return completed
