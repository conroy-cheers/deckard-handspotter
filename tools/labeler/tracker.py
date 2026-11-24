"""Bounded, independent per-camera SAM 3.1 box proposals."""

from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import gc
import hashlib
from pathlib import Path
from statistics import median

from workspace import HANDS, image_name, read_json, write_json


def windows(sequences, times_us, labels, radius):
    """Assign each unlabeled sample to its nearest manual seed in the same visibility interval."""
    if not sequences:
        return []
    cadence = median(b - a for a, b in zip(times_us, times_us[1:])) if len(sequences) > 1 else 0
    maximum_gap = max(500_000, 3 * cadence)
    intervals = []
    start = 0
    for i, sequence in enumerate(sequences):
        if labels.get(sequence, {}).get("status") == "absent":
            intervals.append(range(start, i))
            start = i + 1
        elif i > start and times_us[i] - times_us[i - 1] > maximum_gap:
            intervals.append(range(start, i))
            start = i
    intervals.append(range(start, len(sequences)))
    assignments = defaultdict(list)
    for interval in intervals:
        seeds = [i for i in interval if labels.get(sequences[i], {}).get("source") == "manual"]
        if not seeds:
            continue
        for index in interval:
            if sequences[index] in labels:
                continue
            right = bisect_right(seeds, index)
            nearest = min(seeds[max(0, right - 1):right + 1], key=lambda seed: abs(index - seed))
            if abs(index - nearest) <= radius:
                assignments[nearest].append(index)
    return list(assignments.items())


def largest_box(mask):
    import numpy as np
    from scipy.ndimage import label

    components, n = label(mask, structure=np.ones((3, 3), np.uint8))
    if not n:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    best = sizes.argmax()
    ys, xs = np.nonzero(components == best)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], int(sizes[best])


@contextmanager
def load_tracker(weights, device, threads):
    if not weights or Path(weights).suffix != ".safetensors" or not Path(weights).is_file():
        raise ValueError("A SAM 3.1 Multiplex safetensors checkpoint is required")
    try:
        import torch
        from comfy.cli_args import args
    except ModuleNotFoundError as error:
        raise ValueError("Run propagation using the frame-hand-labeler package with SAM 3.1") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("PyTorch cannot see a CUDA or ROCm device")
    if device == "cpu":
        torch.set_num_threads(threads)

    # Configure inference before ComfyUI initializes its device manager. The
    # labeler's CLI owns argument parsing and never starts the ComfyUI app.
    args.cpu = device == "cpu"
    args.disable_dynamic_vram = True
    from comfy import model_management, sd
    from comfy.ldm.sam3.tracker import SAM31Tracker

    dtype = torch.float32 if args.cpu else torch.float16
    patcher = sd.load_diffusion_model(weights, {"dtype": dtype}, disable_dynamic=True)
    if patcher is None or not isinstance(getattr(patcher.model.diffusion_model, "tracker", None), SAM31Tracker):
        raise ValueError("Expected a SAM 3.1 Multiplex checkpoint in ComfyUI format")
    try:
        model_management.load_model_gpu(patcher)
        with torch.inference_mode():
            yield patcher.model.diffusion_model, model_management.get_torch_device(), patcher.model.get_dtype()
    finally:
        model_management.unload_all_models()
        del patcher
        gc.collect()


def track_window(model, device, dtype, workspace, camera, seed, low, high, tracks):
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn.functional import interpolate
    from comfy.ldm.sam3.tracker import unpack_masks

    frames = []
    for group in workspace.groups[low:high+1]:
        with Image.open(workspace.root / image_name(group["sequence"], camera)) as image:
            frames.append(np.asarray(image.convert("L")))
    frames = torch.from_numpy(np.stack(frames)).unsqueeze(1).float().div_(255)
    height, width = frames.shape[-2:]
    size = model.tracker.image_size
    seed_offset = seed-low
    image = interpolate(frames[seed_offset:seed_offset+1].expand(-1, 3, -1, -1),
                        size=(size, size), mode="bilinear", align_corners=False).to(device, dtype)
    scale = torch.tensor([size/width, size/height], device=device, dtype=dtype)
    seed_masks = []
    for box, _ in tracks.values():
        corners = torch.tensor(box, device=device, dtype=dtype).reshape(1, 2, 2)*scale
        logits = model.forward_segment(image, box_inputs=corners)
        seed_masks.append(interpolate(logits, size=(height, width), mode="bilinear", align_corners=False) > 0)
    seed_masks = torch.cat(seed_masks).to(dtype)
    seed_sequence = workspace.groups[seed]["sequence"]

    # Multiplex starts at frame zero. Both passes reuse the same seed masks
    # and assign object slots in the original, deterministic hand order.
    for step, span in ((1, frames[seed_offset:]), (-1, frames[:seed_offset+1].flip(0))):
        if len(span) < 2:
            continue
        result = model.forward_video(span.expand(-1, 3, -1, -1), initial_masks=seed_masks,
                                     target_device=device, target_dtype=dtype)
        packed = result["packed_masks"]
        if packed is None:
            continue
        if packed.shape[:2] != (len(span), len(tracks)):
            raise ValueError("SAM 3.1 returned an unexpected frame or object count")
        for offset in range(1, len(span)):
            index = seed + step*offset
            masks = unpack_masks(packed[offset]).unsqueeze(1).float()
            masks = (interpolate(masks, size=(height, width), mode="bilinear", align_corners=False)[:, 0] > .5)
            for mask, (hand, (_, targets)) in zip(masks.cpu().numpy(), tracks.items(), strict=True):
                if index not in targets:
                    continue
                region = largest_box(mask)
                if region:
                    sequence = workspace.groups[index]["sequence"]
                    bounds, pixels = region
                    yield workspace.key(sequence, camera, hand), dict(box_xyxy=bounds,
                                                                      visible_pixels=pixels, seed=seed_sequence)


def prompt_snapshot(workspace, cameras):
    return {key: value for key, value in workspace.load_labels().items()
            if key.split("/")[1] in cameras and (value.get("source") == "manual" or value["status"] == "absent")}


def propagate(workspace, cameras, radius, device, threads, weights, *, start=0, limit=None):
    if radius < 1 or threads < 1 or not 0 <= start < len(workspace.groups) or (limit is not None and limit < 1):
        raise ValueError("Use a valid frame range, positive radius and positive thread count")
    cameras = set(cameras or workspace.cameras)
    if not cameras <= set(workspace.cameras):
        raise ValueError("Select cameras present in the workspace")
    sequences = [group["sequence"] for group in workspace.groups]
    times_us = [group["time_us"] for group in workspace.groups]
    end = len(sequences) if limit is None else min(start + limit, len(sequences))
    selected = set(map(str, sequences[start:end]))
    jobs = defaultdict(dict)
    prompts = prompt_snapshot(workspace, cameras)
    reviewed = workspace.load_labels()
    for camera in workspace.cameras:
        if camera not in cameras:
            continue
        for hand in HANDS:
            labels = {sequence: reviewed[key] for sequence in sequences
                      if (key := workspace.key(sequence, camera, hand)) in reviewed}
            for seed, targets in windows(sequences, times_us, labels, radius):
                targets = [index for index in targets if start <= index < end]
                if not targets:
                    continue
                low, high = min(seed, targets[0]), max(seed, targets[-1])
                jobs[camera, seed, low, high][hand] = labels[sequences[seed]]["box_xyxy"], set(targets)

    if not jobs and not any(value.get("source") == "manual" for value in prompts.values()):
        raise ValueError("Draw at least one manual hand box in the selected camera views first")
    if not jobs:
        print("No unlabeled targets in this range are within tracking distance of a seed", flush=True)

    with (workspace.root / ".propagate.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This workspace is already generating proposals") from error
        return run_jobs(workspace, cameras, prompts, jobs, device, threads, weights, selected)


def run_jobs(workspace, cameras, prompts, jobs, device, threads, weights, selected):
    output = workspace.root / "proposals.json"
    result = read_json(output, {"schema": 1, "models": {}, "boxes": {}})
    old_models = result.get("models", {})
    before = len(result["boxes"])
    for key, box in list(result["boxes"].items()):
        sequence, camera, _ = key.split("/")
        if camera in cameras and sequence in selected:
            del result["boxes"][key]
        elif "model" not in box and camera in old_models:
            box["model"] = old_models[camera]
    retained = len(result["boxes"])
    if jobs:
        print(f"Loading SAM 3.1 Multiplex for {len(jobs)} tracking windows", flush=True)
        with load_tracker(weights, device, threads) as (model, device, dtype):
            for i, ((camera, seed, low, high), tracks) in enumerate(jobs.items(), 1):
                print(f"[{i}/{len(jobs)}] {camera} {', '.join(tracks)}, seed "
                      f"{workspace.groups[seed]['sequence']}, {high - low + 1} frames on {device}", flush=True)
                result["boxes"].update(track_window(model, device, dtype, workspace, camera,
                                                    seed, low, high, tracks))
        with Path(weights).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        model = dict(name="sam3.1_multiplex", sha256=digest)
        for key, box in result["boxes"].items():
            sequence, camera, _ = key.split("/")
            if camera in cameras and sequence in selected:
                box["model"] = model
    if prompt_snapshot(workspace, cameras) != prompts:
        raise ValueError("Prompts changed during propagation. Run again to use the updated annotations")
    write_json(output, result)
    return len(result["boxes"]) - retained, before - retained
