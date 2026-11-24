"""Exercise the packaged live session against a separately exported classroom sample."""
import json
from pathlib import Path
from subprocess import PIPE, Popen
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from urllib.request import Request, urlopen


def edit(process, index=0, mode="scene", **job):
    process.stdin.write(json.dumps(dict(job={"seed": 17, "count": 2, **job}, index=index, mode=mode)) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def manifest(reply):
    with urlopen(f"{reply['url']}v/{reply['version']}/job.json", timeout=10) as response:
        return json.load(response)


if __name__ == "__main__":
    dataset = json.loads(Path(sys.argv[1]).read_text())
    batch_streams = dataset["stream_order"]
    batch, empty = dataset["records"]
    assert empty["render_setup"]["empty_room"]
    assert not empty["hands"] and all(not view["visible_hands"] for view in empty["views"].values())
    # Recover detector bytes and hand bounds independently from native PNGs.
    root = Path(sys.argv[1]).parent
    tensors = np.load(root/"handspotter.npy")
    coco = COCO(root/"coco.json")
    image_ids = coco.getImgIds()
    assert len(image_ids) == len(tensors)*len(batch_streams) and coco.anns
    assert len(coco.anns) == len(coco.dataset["annotations"])
    assert {i: cat["name"] for i, cat in coco.cats.items()} == {1: "left_hand", 2: "right_hand"}
    images = {image["file_name"]: image for image in coco.loadImgs(image_ids)}
    for slot, record in enumerate(dataset["records"]):
        for eye, stream in enumerate(batch_streams):
            view = record["views"][stream]
            image = np.asarray(Image.open(root/view["image"]))
            expected = (image.reshape(128, 8, 160, 8).sum((1, 3)) + 32) // 64
            np.testing.assert_array_equal(tensors[slot, eye], expected)
            mask = np.asarray(Image.open(root/view["mask"]))
            coco_image = images[view["image"]]
            assert (coco_image["height"], coco_image["width"]) == image.shape
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[coco_image["id"]]))
            assert len(annotations) == len(view["visible_hands"])
            by_category = {ann["category_id"]: ann for ann in annotations}
            for label, side in enumerate(("left", "right"), start=1):
                rows, cols = np.where(mask == label)
                hand = view["visible_hands"].get(side)
                assert (hand is not None) == bool(rows.size)
                if hand:
                    assert hand["visible_pixels"] == rows.size
                    assert hand["box_xyxy"] == [cols.min(), rows.min(), cols.max()+1, rows.max()+1]
                    ann = by_category[label]
                    assert not ann["iscrowd"] and ann["area"] == rows.size
                    assert ann["bbox"] == [cols.min(), rows.min(), np.ptp(cols)+1, np.ptp(rows)+1]
                    np.testing.assert_array_equal(coco.annToMask(ann), mask == label)
                    points = np.array(ann["keypoints"]).reshape(21, 3)
                    assert points.shape[0] == len(coco.cats[label]["keypoints"])
                    assert ann["num_keypoints"] == np.count_nonzero(points[:, 2])
                    for (x, y, v), (col, row), in_front in zip(
                        points, hand["keypoints_xy"], hand["keypoints_in_front"], strict=True
                    ):
                        if in_front and 0 <= col < mask.shape[1] and 0 <= row < mask.shape[0]:
                            assert (x, y, v) == (col, row, 2 if mask[int(row), int(col)] == label else 1)
                        else:
                            assert (x, y, v) == (0, 0, 0)
    with Popen(["anyhand-studio", "batch", "--studio-session"], stdin=PIPE, stdout=PIPE, text=True) as process:
        try:
            initial = edit(process)
            assert "error" in edit(process, environment="foundation-classroom", foundation_scenes="/missing")
            classroom = edit(process, environment="foundation-classroom")
            assert initial["url"] == classroom["url"]
            scene = manifest(classroom)
            assert scene["geometry"] == initial["version"]
            assert scene["room"] == dict(layout=batch["environment"], lighting=batch["lighting"])
            assert scene["viewport_setup"] == batch["render_setup"]
            negative = edit(process, index=1, environment="foundation-classroom")
            assert manifest(negative)["viewport_setup"] == empty["render_setup"]
            # Exercise the detector handoff with a known black camera readback.
            options = dict(face_size=96, handspotter_render_scale=2, tensors_only=True)
            capture = edit(process, index=1, mode="detector", **options)
            camera_job = manifest(capture)
            again = edit(process, index=1, mode="detector", environment="desk", **options)
            updated = manifest(again)
            assert updated["optics"] == camera_job["optics"]
            assert updated["geometry"] == camera_job["geometry"]
            base = f"{again['url']}v/{again['version']}/"
            for eye, (width, height) in enumerate(updated["sensor_sizes"]):
                upload = Request(base+f"output/0-{eye}.bin", data=bytes(64 + 160*128),
                                 headers={"X-Capture-Token": updated["capture"]["token"]})
                with urlopen(upload) as response:
                    assert response.status == 204
            process.stdin.write(json.dumps({"finish": again["version"]})+"\n")
            process.stdin.flush()
            views = json.loads(process.stdout.readline())["views"]
            assert [view["stream"] for view in views] == batch_streams
            assert all((v["width"], v["height"]) == (160, 128) and not v["boxes"] for v in views)
            changed = edit(process, environment="foundation-classroom", seed=2)
            assert manifest(changed)["geometry"] == changed["version"]
        finally:
            process.stdin.close()
        assert process.wait(timeout=30) == 0
