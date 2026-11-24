"""Download the pinned public HOT3D training clips and prepare Frame shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2

from handspotter_shards import digest, read_hot3d_shard
from hot3d_warp import prepare_clip


PUBLISHER = "bop-benchmark/hot3d"
SOURCE_PATH = re.compile(r"train_(quest3|aria)/clip-\d{6}\.tar")


def read(path):
    return json.loads(path.read_text())


def write_atomic(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def load_source(path):
    source = read(path)
    revision = source["revision"]
    entries = source["archives"]
    names = set()
    if source["repo"] != PUBLISHER or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a pinned HOT3D-Clips source")
    for item in entries:
        match = SOURCE_PATH.fullmatch(item["path"])
        if (not match or match[1] != item["device"] or item["path"] in names
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or item["bytes"] <= 0):
            raise ValueError(f"Invalid HOT3D manifest entry {item}")
        names.add(item["path"])
    if not entries:
        raise ValueError("No labeled HOT3D clips in source manifest")
    return revision, entries


def download(url, archive, expected_size):
    temporary = archive.with_suffix(".part")
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "4",
               "--retry-all-errors", "--connect-timeout", "20", "--max-time", "900",
               "--speed-time", "90", "--speed-limit", "16384", "--output", str(temporary), url]
    offset = temporary.stat().st_size if temporary.is_file() else 0
    if 0 < offset < expected_size:
        resumed = subprocess.run([*command,"--continue-at","-"])
        if resumed.returncode not in (0, 33, 36):
            resumed.check_returncode()
        if resumed.returncode != 0:
            temporary.unlink()
            subprocess.run(command, check=True)
    elif offset != expected_size:
        subprocess.run(command, check=True)
    if temporary.stat().st_size != expected_size:
        raise ValueError(f"Unexpected download size for {archive.name}")
    temporary.replace(archive)


def build_clip(root, rig, rig_hash, revision, item, cached):
    path = Path(item["path"])
    output = root/"shards"/path.parent/path.stem
    if output.is_dir():
        record = read(output/"clip.json")
        if record["rig_sha256"] != rig_hash or record["archive_sha256"] != item["sha256"]:
            raise ValueError(f"Stale HOT3D shard at {output}")
        count = len(record["frames"])
        if count != 150:
            raise ValueError(f"Invalid cached HOT3D frame count at {output}")
        read_hot3d_shard(output, count)
        if "hashes" in record:
            for name in ("front", "boxes", "valid"):
                array_path = output / f"{name}.npy"
                if digest(array_path) != record["hashes"][name]:
                    raise ValueError(f"Changed cached HOT3D array: {array_path}")
        return "cached", record
    archive = root/"archives"/path.name
    # Existing local archives are read in place and left with their owner.
    local = cached.get(path.name)
    if local is not None and local.stat().st_size == item["bytes"] and digest(local) == item["sha256"]:
        return "local", prepare_clip(local, rig, output, item["sha256"], item["device"])
    url = f"https://huggingface.co/datasets/{PUBLISHER}/resolve/{revision}/{item['path']}"
    if not archive.is_file() or archive.stat().st_size != item["bytes"]:
        download(url, archive, item["bytes"])
    try:
        result = prepare_clip(archive, rig, output, item["sha256"], item["device"])
    except ValueError as exc:
        if "HOT3D archive hash mismatch" in str(exc):
            archive.unlink()
        raise
    archive.unlink()
    return "downloaded", result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "rig"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--local-archives", type=Path, nargs="*", default=())
    parser.add_argument("--toolkit", type=Path, help="Override the installed hand tracking toolkit")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.toolkit:
        sys.path.insert(0, str(args.toolkit.resolve()))
    cv2.setNumThreads(1)
    root, rig = args.root.resolve(), args.rig.resolve()
    revision, items = load_source(root/"source.json")
    rig_hash = digest(rig)
    cached = {p.name:p for directory in args.local_archives for p in directory.glob("clip-*.tar")}
    (root/"archives").mkdir(exist_ok=True)
    (root/"reports").mkdir(exist_ok=True)
    failures = {}
    completed = {}
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(build_clip, root, rig, rig_hash, revision, item, cached):item for item in items}
        for future in as_completed(pending):
            item = pending[future]
            try:
                method, stats = future.result()
            except Exception as exc:
                failures[item["path"]] = str(exc)
                print(f"{item['path']} failed: {exc}", file=sys.stderr, flush=True)
            else:
                completed[item["path"]] = stats
                print(f"{len(completed)}/{len(items)} {item['path']} {method} "
                      f"{stats['positive_boxes']} positive boxes", flush=True)
            if len(completed) % 16 == 0 or failures:
                write_atomic(root/"reports/progress.json", dict(
                    completed=len(completed), expected=len(items), failed=failures,
                    elapsed_seconds=round(time.monotonic()-start, 1),
                ))
    write_atomic(root/"reports/progress.json", dict(
        completed=len(completed), expected=len(items), failed=failures,
        elapsed_seconds=round(time.monotonic()-start, 1),
    ))
    if failures:
        raise SystemExit(f"{len(failures)} HOT3D clips failed. Run the same command to resume.")
    records = []
    for item in items:
        path = Path(item["path"])
        result = completed[item["path"]]
        records.append(dict(path=str(Path("shards")/path.parent/path.stem),
                            count=len(result["frames"]), device=result["device"],
                            positive_boxes=result["positive_boxes"], archive_sha256=result["archive_sha256"]))
    write_atomic(root/"training.json", dict(schema=1, rig_sha256=rig_hash, publisher=PUBLISHER,
                 revision=revision, groups=sum(row["count"] for row in records), shards=records))
    print(f"Prepared {len(records)} HOT3D clips for training", flush=True)


if __name__ == "__main__":
    main()
