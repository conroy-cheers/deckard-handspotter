from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from hot3d_input import ingest_hot3d
from workspace import Workspace, read_json


class HOT3DInputTests(unittest.TestCase):
    def test_mapped_channel_exports_known_labels_and_keeps_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/"clip.tar"
            with tarfile.open(archive, "w") as clip:
                def add(frame, name, payload):
                    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                    entry = tarfile.TarInfo(f"{frame:06d}.{name}")
                    entry.size = len(data)
                    clip.addfile(entry, BytesIO(data))

                image = BytesIO()
                Image.new("L", (32, 24), 90).save(image, format="JPEG")
                for frame in (0, 1):
                    stamp = 1_000_000_000 + frame*100_000_000
                    add(frame, "info.json", dict(device="Aria", participant_id="P0", sequence_id="test",
                                                ref_timestamp_ns=stamp, image_timestamps_ns={"1201-2": stamp+1000}))
                    add(frame, "cameras.json", {"1201-2": {"calibration": {"image_width": 32, "image_height": 24}}})
                    add(frame, "image_1201-2.jpg", image.getvalue())
                add(0, "hands.json", {"left": {"boxes_amodal": {"1201-2": [3, 5, 11, 16]},
                                               "visibilities_modeled": {"1201-2": .8}}})
            dest = root/"workspace"
            self.assertEqual(ingest_hot3d(archive, dest, 0, None, 1, channels=["1201-2=slam_right"]), 2)
            workspace = Workspace(dest)
            self.assertEqual(workspace.cameras, ("slam_right",))
            self.assertEqual(workspace.groups[1]["time_us"], 100_000)
            self.assertEqual(workspace.groups[1]["views"]["slam_right"]["pts_us"], 100_001)
            labels = dest/"labels"
            self.assertEqual(workspace.export(labels), (1, 1))
            coco = read_json(labels/"coco.json")
            self.assertEqual(coco["annotations"][0]["bbox"], [3, 5, 8, 11])
            self.assertEqual(coco["annotations"][0]["box_mode"], "amodal")
            groups = read_json(labels/"groups.json")["groups"]
            self.assertEqual(groups[0]["views"]["slam_right"]["hands"]["right"]["status"], "absent")
            self.assertEqual(groups[1]["views"]["slam_right"]["hands"]["right"]["status"], "unknown")
            workspace.set_label(1, "slam_right", "left", "absent")
            workspace.set_label(1, "slam_right", "right", "absent")
            self.assertEqual(workspace.export(labels), (2, 1))
