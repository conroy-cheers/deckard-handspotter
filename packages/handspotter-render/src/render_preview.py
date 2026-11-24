"""Read one completed batch sample. All displayed pixels come from batch artifacts."""
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def preview(root, index, mode):
    dataset = json.loads((root / "dataset.json").read_text())
    records = dataset["records"]
    streams = dataset["stream_order"]
    if not 0 <= index < len(records):
        raise ValueError("Sample index outside this batch")
    detector = (root / "handspotter.npy").is_file()
    if mode == "detector" and not detector:
        raise ValueError("This batch has no Handspotter detector inputs")
    record = records[index]
    tensor = np.load(root / "handspotter.npy", mmap_mode="r", allow_pickle=False) if mode == "detector" else None
    return dict(count=len(records), index=index, id=record["id"],
                views=preview_views(streams, record, tensor[index] if tensor is not None else None, root, mode),
                setup=record.get("render_setup"), detector=detector,
                images=all("image" in view and "mask" in view for view in record["views"].values()))


def preview_views(streams, record, tensor, root=None, mode="detector"):
    views = []
    for eye, stream in enumerate(streams):
        view = record["views"][stream]
        if tensor is not None:
            image = Image.fromarray(tensor[eye].squeeze())
        else:
            path = Path(view[mode])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid batch image path")
            with Image.open(root / path) as source:
                image = source.copy()
            if mode == "mask":
                image = image.point(lambda value: min(255, value * 120))
        boxes = {}
        for side, hand in view["visible_hands"].items():
            box = hand["box_xyxy"]
            # Handspotter boxes use 1280x1024 reference pixels.
            boxes[side] = [v / 8 for v in box] if mode == "detector" else box
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        views.append(dict(stream=stream, width=image.width, height=image.height, boxes=boxes,
                          png=base64.b64encode(encoded.getvalue()).decode("ascii")))
    return views


if __name__ == "__main__":
    print(json.dumps(preview(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])))
