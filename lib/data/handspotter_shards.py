"""Read and verify prepared Handspotter dataset shards."""
import hashlib
import json

import numpy as np

SPLITS = ("training", "validation", "test")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_file(path):
    return json.loads(path.read_text())


def split_shards(dataset, split):
    manifest = json_file(dataset / "dataset.json")
    if manifest["schema"] != 1 or not manifest["splits"][split]:
        raise ValueError(f"Dataset has no usable {split} split: {dataset}")
    return manifest["splits"][split]


def load_split(dataset, split, source=None):
    shards = [item for item in split_shards(dataset, split)
              if source is None or item["kind"] == source]
    if not shards:
        raise ValueError(f"Dataset has no {source} groups in {split}: {dataset}")
    arrays = []
    for item in shards:
        count = item["groups"]
        columns = []
        for name, shape, dtype in (
            ("images", (count, 4, 128, 160), np.uint8),
            ("boxes", (count, 4, 2, 4), np.float32),
            ("valid", (count, 4, 2), np.bool_),
        ):
            path = dataset / f"{item['shard']}-{name}.npy"
            if digest(path) != item["hashes"][name]:
                raise ValueError(f"Dataset content changed since preparation: {path}")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.shape != shape or data.dtype != dtype:
                raise ValueError(f"Dataset shape or dtype differs from the manifest: {path}")
            columns.append(data)
        arrays.append(columns)
    return tuple(np.concatenate(column) for column in zip(*arrays, strict=True))


def read_hot3d_shard(path, count):
    images, boxes, valid = (
        np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("front", "boxes", "valid")
    )
    if ((images.shape, images.dtype) != ((count, 2, 128, 160), np.uint8)
            or (boxes.shape, boxes.dtype) != ((count, 4, 2, 4), np.float32)
            or (valid.shape, valid.dtype) != ((count, 4, 2), np.bool_)):
        raise ValueError(f"Invalid HOT3D training arrays at {path}")
    return images, boxes, valid
