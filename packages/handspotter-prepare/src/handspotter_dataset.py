"""Prepare mixed Handspotter shards from reviewed Frame workspaces and synthetic renders."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import tomllib
from types import SimpleNamespace

import numpy as np
from PIL import Image

from calibration import load_frame
from handspotter import ORDER, Plane, Preparation, U8
from handspotter_shards import SPLITS, digest, json_file

SIDES = ("left", "right")


def recorded(root, export, groups, xrservice, device):
    cameras = load_frame(xrservice, device)
    geometry = SimpleNamespace(cameras=[cameras[name] for name in ORDER])
    circles = {row["sourceCamera"]: row["imageCircleDiameterInDeg"]
               for row in json_file(xrservice)["cameras"] if row["sourceCamera"] in ORDER}
    preparation = Preparation(circles)
    sequences = set()
    for group in groups:
        sequence = group["sequence"]
        if sequence in sequences or set(group["views"]) != set(ORDER):
            raise ValueError(f"Duplicate or incomplete camera group {sequence} in {export}")
        sequences.add(sequence)
        images = []
        boxes = np.zeros((4, 2, 4), dtype=np.float32)
        valid = np.zeros((4, 2), dtype=bool)
        for slot, name in enumerate(ORDER):
            view = group["views"][name]
            width, height = cameras[name].width, cameras[name].height
            path = (root / view["file_name"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Frame path leaves its workspace: {path}")
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (width, height):
                    raise ValueError(f"Calibration and native image differ for {path}")
                if (view["width"], view["height"]) != image.size:
                    raise ValueError(f"Labeler dimensions differ for {path}")
                images.append(np.array(image))
            for hand, side in enumerate(SIDES):
                label = view["hands"][side]
                status = label["status"]
                if status not in ("absent", "box", "unknown"):
                    raise ValueError(f"Invalid label status {status!r} for {path}")
                valid[slot, hand] = status != "unknown"
                if status != "box":
                    continue
                edges = np.asarray(label["box_xyxy"], dtype=np.float64)
                if (edges.shape != (4,) or not np.isfinite(edges).all()
                        or (edges[:2] < 0).any() or (edges[2:] > [width, height]).any()
                        or (edges[2:] <= edges[:2]).any()):
                    raise ValueError(f"Invalid native box for {path}")
                # Convert half-open image edges into the model's pixel-centre frame.
                boxes[slot, hand] = edges * np.tile([1280 / width, 1024 / height], 2) - .5
        planes = (Plane * 4)(*(Plane(image.ctypes.data_as(U8), image.nbytes, image.shape[1],
                                     image.shape[0], image.strides[0]) for image in images))
        yield preparation.prepare(planes, geometry)[0], boxes, valid


def prepare(config_path, destination):
    config = tomllib.loads(config_path.read_text())
    resolve = lambda value: (config_path.parent / value).resolve()
    if destination.exists():
        raise ValueError(f"Dataset output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "splits": {split: [] for split in SPLITS}}
    assignments = {}
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".handspotter-") as temp:
        output = Path(temp) / "dataset"
        output.mkdir()
        for kind in ("recorded", "synthetic"):
            for index, item in enumerate(config.get(kind, [])):
                split = item["split"]
                if split not in SPLITS:
                    raise ValueError(f"Invalid split {split!r}")
                shard = f"{kind}-{index:04}"
                if kind == "recorded":
                    root = resolve(item["workspace"])
                    export = (root / item.get("labels", "labels/groups.json")).resolve()
                    labels = json_file(export)
                    if labels.get("source", {}).get("kind") == "hot3d":
                        raise ValueError("Frame recording preparation requires native captures. Use the HOT3D training corpus for warped clips.")
                    if labels["schema"] != 1 or not labels["groups"]:
                        raise ValueError(f"Expected a nonempty grouped labeler export at {export}")
                    identity = json_file(root / "manifest.json")["timestamps_sha256"]
                    if not identity or labels.get("timestamps_sha256") != identity:
                        raise ValueError(f"Label export {export} belongs to a different take than workspace {root}")
                    count = len(labels["groups"])
                    xrservice, device = (resolve(item[key]) for key in ("xrservice", "device_config"))
                    rows = recorded(root, export, labels["groups"], xrservice, device)
                    dtypes = (np.uint8, np.float32, np.bool_)
                    shapes = ((count, 4, 128, 160), (count, 4, 2, 4), (count, 4, 2))
                    arrays = [np.lib.format.open_memmap(output / f"{shard}-{key}.npy", mode="w+",
                                                        dtype=dtype, shape=shape)
                              for key, dtype, shape in zip(("images", "boxes", "valid"), dtypes, shapes)]
                    for row, values in enumerate(rows):
                        for array, value in zip(arrays, values, strict=True):
                            array[row] = value
                    for array in arrays:
                        array.flush()
                    source = {"take": identity, "labels_sha256": digest(export),
                              "xrservice_sha256": digest(xrservice), "device_config_sha256": digest(device)}
                    identities = {identity, item.get("session", identity)}
                else:
                    from handspotter_data import load_synthetic, synthetic_identities
                    if "path" in item:
                        root = resolve(item["path"])
                    else:
                        root = output / f"{shard}-render"
                        if "rig" in item:
                            rig = resolve(item["rig"])
                        else:
                            from rig_export import export_rig
                            rig = output / f"{shard}-rig.json"
                            export_rig(resolve(item["xrservice"]), resolve(item["device_config"]), rig)
                        if item.get("preset") == "procedural":
                            from handspotter_poses import pose_bank
                            inputs = output / f"{shard}-poses"
                            inputs.mkdir()
                            pose, topology = pose_bank(inputs, item["seed"], min(128, item["count"]))
                            poses = [pose]
                        else:
                            poses = [resolve(value) for value in item["poses"]]
                            topology = resolve(item["topology"])
                        subprocess.run([
                            "handspotter-render", "--rig-json", str(rig), "--source",
                            *map(str, poses),
                            "--topology", str(topology), "--out", str(root),
                            "--count", str(item["count"]), "--seed", str(item["seed"]),
                            "--handspotter-render-scale", str(item.get("render_scale", 2)), "--tensors-only",
                            "--face-size", str(item.get("face_size", 192)),
                            "--environment", item.get("environment", "all"),
                            *[f"--{name.replace('_', '-')}" for name in (
                                "appearance_variation", "capture_noise", "slight_blur", "lens_decentering",
                                "arm_motion_blur", "camera_motion_blur", "hemisphere_reach", "head_look",
                            ) if item.get(name, False)],
                        ], check=True)
                    arrays = load_synthetic(root, split, 0, 0)
                    data = json_file(root / "dataset.json")
                    identities = synthetic_identities(data["records"], arrays[0])
                    source = {"render_sha256": digest(root / "dataset.json"),
                              "rig_sha256": digest(root / "rig.json")}
                    count = len(arrays[0])
                    if not count:
                        raise ValueError(f"Empty render: {root}")
                    for key, array in zip(("images", "boxes", "valid"), arrays, strict=True):
                        np.save(output / f"{shard}-{key}.npy", array, allow_pickle=False)
                for identity in identities:
                    if assignments.setdefault((kind, identity), split) != split:
                        raise ValueError(f"{kind} source {identity} crosses dataset splits")
                supervised = int(np.count_nonzero(arrays[2]))
                if not supervised:
                    raise ValueError(f"{split} {shard} has no reviewed labels")
                hashes = {key: digest(output / f"{shard}-{key}.npy")
                          for key in ("images", "boxes", "valid")}
                if kind == "recorded":
                    groups = [dict(sequence=group["sequence"], time_us=group["time_us"])
                              for group in labels["groups"]]
                    sidecar = output / f"{shard}-groups.json"
                    sidecar.write_text(json.dumps(groups) + "\n")
                    source.update(group_index=sidecar.name, group_index_sha256=digest(sidecar))
                manifest["splits"][split].append(
                    dict(kind=kind, shard=shard, groups=count, hashes=hashes, **source)
                )
                print(f"{split} {shard}: {count} groups, {supervised} supervised slots")
        if not any(manifest["splits"].values()):
            raise ValueError("Configure at least one synthetic or recorded source")
        (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.config.resolve(), args.out.resolve())
    except (OSError, ValueError) as error:
        parser.exit(1, f"handspotter-prepare: {error}\n")


if __name__ == "__main__":
    main()
