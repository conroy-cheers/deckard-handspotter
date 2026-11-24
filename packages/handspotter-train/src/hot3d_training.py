"""Lazy Handspotter input batches from a warped HOT3D corpus."""
from functools import lru_cache
import json

import numpy as np


from handspotter_shards import read_hot3d_shard


class FrontCorpus:
    def __init__(self, root, rig_hash):
        self.root = root.resolve()
        manifest = json.loads((self.root/"training.json").read_text())
        if manifest["schema"] != 1 or manifest["rig_sha256"] != rig_hash:
            raise ValueError(f"HOT3D training corpus at {root} uses a different Frame rig or schema")
        self.revision = manifest["revision"]
        self.shards = [(self.root/item["path"]).resolve() for item in manifest["shards"]]
        counts = [int(item["count"]) for item in manifest["shards"]]
        if (not counts or min(counts) < 1 or sum(counts) != manifest["groups"]
                or len(set(self.shards)) != len(self.shards)
                or any(not path.is_relative_to(self.root) for path in self.shards)):
            raise ValueError(f"Invalid HOT3D training manifest at {root}")
        self.ends = np.cumsum(counts)
        self.counts = counts
        self.read = lru_cache(maxsize=64)(read_hot3d_shard)

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, indices):
        shard_ids = np.searchsorted(self.ends, indices, side="right")
        first = self.ends - self.counts
        groups = len(indices)
        images = np.zeros((groups, 4, 128, 160), dtype=np.uint8)
        boxes = np.empty((groups, 4, 2, 4), dtype=np.float32)
        valid = np.empty((groups, 4, 2), dtype=bool)
        for shard in np.unique(shard_ids):
            selected = shard_ids == shard
            offsets = indices[selected] - first[shard]
            front, labels, masks = self.read(self.shards[shard], self.counts[shard])
            images[selected, 1:3] = front[offsets]
            boxes[selected] = labels[offsets]
            valid[selected] = masks[offsets]
        return images, boxes, valid
