import csv
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import av
import numpy as np
from PIL import Image
from contact import review_spans, rotate_box
from geometry import Camera
from handedness import resolved_slots, stereo_motion_link

from tracker import windows
from terra import MODEL, decode, review_selection
from workspace import CAMERAS, HANDS, Workspace, image_name, ingest, labels_digest, read_json, sample_by_time, write_json


class TestFrameLabels(unittest.TestCase):
    def test_contact_spans_and_rotations(self):
        self.assertEqual(review_spans([0, 3, 11], 12), [[0, 6], [9, 12]])
        box = [122, 234, 678, 984]
        self.assertEqual(rotate_box(box, 1), [234, 322, 984, 878])
        for turns in range(4):
            self.assertEqual(rotate_box(rotate_box(box, turns), -turns), box)

    def test_contact_overlay_requires_completion_and_matching_baseline(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "manifest.json", dict(schema=1, cameras=dict.fromkeys(CAMERAS), groups=[]))
            workspace = Workspace(root)
            key = "1/slam_left/left"
            raw = {key: {"status": "box", "box_xyxy": [1, 2, 3, 4]}}
            write_json(root / "terra-review.json", raw)
            revised = {key: {"status": "unknown", "source": "terra_contact"}}
            overlay = dict(plan=dict(source_sha256=labels_digest(raw)), complete=False, labels=revised)
            path = root / "terra-contact-review.json"
            write_json(path, overlay)
            self.assertEqual(workspace.load_terra(), raw)
            overlay["complete"] = True
            write_json(path, overlay)
            self.assertEqual(workspace.load_terra(), revised)
            changed = raw | {"1/slam_left/right": {"status": "absent"}}
            write_json(root / "terra-review.json", changed)
            self.assertEqual(workspace.load_terra(), changed)

    def test_mechanical_association_keeps_identity_through_a_crossing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            layout = dict.fromkeys(CAMERAS, {"width": 640, "height": 480})
            groups = [dict(sequence=i, time_us=i*100_000) for i in range(17)]
            write_json(root / "manifest.json", dict(schema=1, cameras=layout, groups=groups))
            workspace = Workspace(root)
            cameras = {}
            for name, x in zip(CAMERAS, (-.055, .055, -.075, .075)):
                transform = np.diag([1., -1., -1., 1.])
                if x > 0:
                    transform[:3, :3] = transform[:3, :3] @ [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
                transform[0, 3] = x
                cameras[name] = Camera(name, 640, 480, "kb4", [315, 315, 320, 240], [0]*8, transform)

            raw, expected = {}, {}
            for i, x in enumerate(np.clip(np.linspace(-.49, .15, len(groups)), -.20, .15)):
                for name in CAMERAS[:2]:
                    swap = 5 <= i <= 7 or (name == "slam_right" and i == 14)
                    if swap:
                        expected[i, name] = True
                    points = [(x, -.23, -.65), (-x, -.07, -.61)]
                    for slot, hand in enumerate(HANDS):
                        (u, v), _ = cameras[name].project(points[slot ^ swap])
                        raw[workspace.key(i, name, hand)] = {"status": "box",
                                                            "box_xyxy": [u-35, v-35, u+35, v+35]}
            flips, _ = resolved_slots(workspace, raw, cameras, [])
            self.assertEqual(flips, expected)

    def test_stereo_positions_bridge_motion_without_box_overlap(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            layout = dict.fromkeys(CAMERAS, {"width": 640, "height": 480})
            groups = [dict(sequence=i, time_us=i*100_000) for i in range(3)]
            write_json(root / "manifest.json", dict(schema=1, cameras=layout, groups=groups))
            workspace = Workspace(root)
            cameras = {}
            for name, x in zip(CAMERAS, (-.055, .055, -.075, .075)):
                transform = np.diag([1., -1., -1., 1.])
                transform[0, 3] = x
                cameras[name] = Camera(name, 640, 480, "kb4", [315, 315, 320, 240], [0]*8, transform)
            raw = {}
            for i, shift in enumerate((0., .17, 0.)):
                for name in CAMERAS[:2]:
                    points = [(-.27+shift, -.21, -.61), (.25+shift, -.05, -.55)]
                    for slot, hand in enumerate(HANDS):
                        (u, v), _ = cameras[name].project(points[slot ^ (i == 1)])
                        raw[workspace.key(i, name, hand)] = {"status": "box",
                                                            "box_xyxy": [u-12, v-12, u+12, v+12]}
            flips, report = resolved_slots(workspace, raw, cameras, [])
            self.assertEqual(flips, {(1, name): True for name in CAMERAS[:2]})
            self.assertEqual(report["spatial_links"], 4)

    def test_stereo_motion_defers_when_hands_are_close(self):
        old = [np.array([-.10, -.20, -.55]), np.array([.10, -.20, -.55])]
        new = [old[1] + [.03, 0, 0], old[0] + [.03, 0, 0]]
        self.assertIsNone(stereo_motion_link(old, new, 100_000))

    def test_timestamp_sampling_keeps_nearest_group_per_tick(self):
        times = [0, 18_000, 85_000, 103_000, 117_000, 199_000, 399_000]
        groups = [dict(sequence=i, time_us=time) for i, time in enumerate(times)]
        self.assertEqual([g["sequence"] for g in sample_by_time(groups, 0, 10)], [0, 3, 5, 6])

    def test_ingest_and_review(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            take = path / "take"
            take.mkdir()
            rows = []
            with av.open(str(take / "tracking.nut"), "w", format="nut") as out:
                streams = [out.add_stream("rawvideo", rate=60) for _ in CAMERAS]
                for stream, name in zip(streams, CAMERAS):
                    stream.width = 16
                    stream.height = 8
                    stream.pix_fmt = "gray"
                    stream.codec_context.time_base = Fraction(1, 1_000_000)
                    stream.metadata["title"] = name
                for sequence, time in ((20, 0), (23, 50_000), (24, 66_667)):
                    for i, (stream, name) in enumerate(zip(streams, CAMERAS)):
                        frame = av.VideoFrame(16, 8, "gray")
                        plane = frame.planes[0]
                        plane.update(bytes([sequence + i]) * plane.buffer_size)
                        frame.pts = time + i
                        frame.time_base = Fraction(1, 1_000_000)
                        out.mux(stream.encode(frame))
                        rows.append(dict(group=sequence, track=i, camera=name, pts_us=time + i,
                                         sensor_raw_s=100 + (time + i) / 1e6, end_exposure_raw_s=0,
                                         arrival_raw_s=0, exposure_s=0, observer_mono_ns=0,
                                         clock_raw_ns=0, clock_mono_ns=0, clock_uncertainty_ns=0))
            with (take / "timestamps.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            workspace_dir = path / "labels"
            self.assertEqual(ingest(take, workspace_dir, 0.04, None, 1), 2)
            workspace = Workspace(workspace_dir)
            self.assertEqual([group["sequence"] for group in workspace.groups], [23, 24])
            for i, camera in enumerate(CAMERAS):
                with Image.open(workspace_dir / image_name(23, camera)) as frame:
                    self.assertEqual(frame.getextrema(), (23 + i, 23 + i))

            workspace.set_label(23, "slam_left", "left", "box", [1, 2, 5, 6])
            with self.assertRaises(ValueError):
                workspace.set_label(23, "slam_left", "left", "box", [5, 0, 1, 1])
            model = {"name": "sam3.1_multiplex", "sha256": "test-checkpoint"}
            proposal = {"box_xyxy": [2, 3, 6, 7], "visible_pixels": 10, "seed": 23, "model": model}
            write_json(workspace_dir / "proposals.json", {"boxes": {"24/slam_left/left": proposal}})
            with self.assertRaisesRegex(ValueError, "proposal changed"):
                workspace.set_label(24, "slam_left", "left", "accept", [1, 1, 2, 2])
            accepted = workspace.set_label(24, "slam_left", "left", "accept", proposal["box_xyxy"])
            self.assertEqual(accepted["source"], "reviewed_sam")
            self.assertEqual(accepted["model"], model)
            workspace.set_label(24, "slam_left", "left", "unknown")
            output = workspace_dir / "labels"
            self.assertEqual(workspace.export_counts(output), (1, 0, 0))
            previous, change = workspace.edit_label(23, "slam_left", "right", "unknown")
            key = workspace.key(23, "slam_left", "right")
            workspace.undo_edit(key, previous, change)
            self.assertNotIn(key, workspace.load_labels())
            workspace.set_label(23, "slam_left", "right", "absent")
            with self.assertRaisesRegex(ValueError, "label changed"):
                workspace.undo_edit(key, previous, change)
            self.assertEqual(workspace.export(output), (1, 1))
            self.assertEqual(read_json(output / "coco.json")["annotations"][0]["bbox"], [1, 2, 4, 4])
            groups = read_json(output / "groups.json")["groups"]
            self.assertEqual(groups[1]["views"]["slam_left"]["hands"]["left"], {"status": "unknown"})

            response = {"23": {camera: dict.fromkeys(HANDS, "unknown") for camera in CAMERAS}}
            response["23"]["slam_left"].update(left=[125, 125, 500, 500], right="absent")
            response["23"]["upper_left"].update(left=[125, 0, 1000, 1000], right="absent")
            terra = decode(workspace, [23], response)
            write_json(workspace_dir / "terra-review.json", terra)
            self.assertEqual(workspace.export(output, include_terra=True), (2, 2))
            frames = read_json(output / "groups.json")["groups"]
            self.assertEqual(frames[0]["views"]["slam_left"]["hands"]["left"]["source"], "manual")
            self.assertEqual(frames[0]["views"]["upper_left"]["hands"]["left"]["model"], MODEL)
            self.assertEqual(frames[0]["views"]["upper_left"]["hands"]["left"]["box_xyxy"], [2, 0, 16, 8])
            response["23"]["upper_left"]["left"] = [125, 0, 0, 1000]
            with self.assertRaisesRegex(ValueError, "Invalid Terra box"):
                decode(workspace, [23], response)

            rows[4]["pts_us"] += 5
            with (take / "timestamps.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "PTS mismatch"):
                ingest(take, path / "mismatched", 0, None, 1)
            self.assertFalse((path / "mismatched").exists())

    def test_propagation_stops_at_absences_gaps_and_radius(self):
        sequences = [20, 21, 23, 24, 25, 26, 27, 28, 29]
        times = [0, 20000, 60000, 80000, 100000, 120000, 140000, 900000, 920000]
        labels = {21: {"status": "box", "source": "manual"},
                  23: {"status": "absent"},
                  26: {"status": "box", "source": "manual"},
                  27: {"status": "unknown"},
                  29: {"status": "box", "source": "reviewed_sam2"}}
        self.assertEqual(windows(sequences, times, labels, 1), [(1, [0]), (5, [4])])

    def test_terra_samples_only_available_groups_in_selected_interval(self):
        times = [0, 20_000, 49_000, 80_000, 120_000, 170_000, 260_000]
        workspace = SimpleNamespace(
            manifest={},
            groups=[dict(sequence=i, time_us=time) for i, time in enumerate(times)],
        )
        selection = review_selection(workspace, 1, 5, 20)
        self.assertEqual([group["sequence"] for group in selection], [1, 3, 4, 5])
        self.assertEqual(review_selection(workspace, 1, 5, 120), workspace.groups[1:6])
        with self.assertRaisesRegex(ValueError, "Specify --hz"):
            review_selection(workspace, 0, None, None)
