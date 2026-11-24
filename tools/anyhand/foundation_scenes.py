"""Fetch the pinned Foundation scenes and define their stereo viewpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import pi
from pathlib import Path
import re
from urllib.request import urlopen


REPOSITORY = "mos9527/Foundation-Resources"
REVISION = "16a04b1b57481d0cb26793eac0498975e8cf7f53"
PLACEMENTS = {
    # Surface anchors mark the near edge and top in the original scene coordinates.
    "classroom": dict(anchor=[-2.03, 1.257, -5.30], yaw=pi, scale=1., light="workbench"),
    "dining-room": dict(anchor=[-.63, .909, -.624], yaw=0., scale=.36, light="desk"),
    "kitchen": dict(anchor=[.13, .863, -.705], yaw=0., scale=1., light="kitchen"),
    "bathroom": dict(anchor=[-1.355, .77, -1.94], yaw=0., scale=1., light="kitchen"),
    "bedroom": dict(anchor=[.03, .463, .945], yaw=0., scale=1., light="desk"),
    "living-room-2": dict(anchor=[.54, .456, 3.22], yaw=0., scale=1., light="desk"),
    "living-room-3": dict(anchor=[-1.42, .332, -2.24], yaw=pi, scale=1., light="workbench"),
    # The staircase and courtyard use floor anchors with standing eye heights.
    "staircase": dict(anchor=[.03, .025, 4.36], yaw=0., scale=1., light="workbench", height_offset=-1.18),
    "sponza": dict(anchor=[5.02, -1., .44], yaw=-pi/2, scale=1., light="courtyard",
                   height_offset=-1.20, far=36.),
}
ASSET_NAME = re.compile(r"[A-Za-z0-9_-]+\.(?:gltf|bin|png|jpe?g)")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def local_references(model):
    graph = json.loads(model)
    references = graph["buffers"] + graph.get("images", [])
    names = {entry["uri"] for entry in references}
    if any(ASSET_NAME.fullmatch(name) is None for name in names):
        raise ValueError("Expected local Foundation scene assets")
    return sorted(names)


def validate_scenes(root, selected):
    files = {}
    for name in selected:
        entrypoint = f"{name}.gltf"
        directory = root/name
        model = (directory/entrypoint).read_bytes()
        for asset in [entrypoint, *local_references(model)]:
            path = directory/asset
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as file:
                if file.read(len(LFS_PREFIX)) == LFS_PREFIX:
                    raise ValueError(f"Materialize the LFS file at {path}")
            files[f"{name}/{asset}"] = sha256(path.read_bytes()).hexdigest()
    return files


def fetch(destination, room, name):
    suffix = f"{REPOSITORY}/{REVISION}/{room}/{name}"
    with urlopen(f"https://raw.githubusercontent.com/{suffix}", timeout=40) as response:
        data = response.read()
    if data.startswith(LFS_PREFIX):
        metadata = dict(line.decode().split(" ", 1) for line in data.splitlines())
        with urlopen(f"https://media.githubusercontent.com/media/{suffix}", timeout=50) as response:
            data = response.read()
        if (len(data) != int(metadata["size"]) or
                f"sha256:{sha256(data).hexdigest()}" != metadata["oid"]):
            raise ValueError(f"LFS object failed verification: {room}/{name}")
    (destination/room/name).write_bytes(data)
    return data


def download(destination, selected=None):
    rooms = dict.fromkeys(selected) if selected else PLACEMENTS
    dependencies = []
    for room in rooms:
        (destination/room).mkdir(parents=True, exist_ok=True)
        model = fetch(destination, room, f"{room}.gltf")
        dependencies.extend((room, name) for name in ["README", *local_references(model)])
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, destination, room, name) for room, name in dependencies]
        for future in futures:
            future.result()
    print(f"Downloaded and verified {len(rooms)} scenes at {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scene", action="append", choices=tuple(PLACEMENTS),
                        help="Download a specific scene. Repeat to select multiple scenes.")
    args = parser.parse_args()
    download(args.destination, args.scene)
