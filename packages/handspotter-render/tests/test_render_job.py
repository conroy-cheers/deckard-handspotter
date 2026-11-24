import argparse
from pathlib import Path
import unittest

from render_handspotter import resolve_sample
from render_job import job_arguments, job_fields


class RenderJobTests(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", nargs="+", type=Path)
        parser.add_argument("--face-size", type=int)
        parser.add_argument("--capture-noise", action="store_true")
        parser.add_argument("--soma-worker")
        self.fields = job_fields(parser)

    def test_paths_flags_and_zero_use_cli_semantics(self):
        job = {"source": ["one.npz", "sub/two.npz"], "face_size": 0, "capture_noise": False}
        self.assertEqual(job_arguments(job, self.fields, Path("/jobs")), [
            "--source", "/jobs/one.npz", "/jobs/sub/two.npz", "--face-size", "0",
        ])

    def test_job_cannot_choose_executables_or_coerce_invalid_types(self):
        for job in ({"soma_worker": "worker"}, {"source": "poses.npz"},
                    {"face_size": True}, {"capture_noise": "false"}):
            with self.subTest(job=job), self.assertRaises(ValueError):
                job_arguments(job, self.fields, Path("/jobs"))

class SampleSetupTests(unittest.TestCase):
    def test_room_precedence_and_seeded_garment(self):
        record = {"environment": {"name": "kitchen"}, "lighting": {"fixture": "window"}}
        kwargs = dict(index=3, seed=7, names=["foundation-classroom"], wardrobe="garmentcode-balanced",
                      cloth_mode="fitted", library=["tee", "hoodie"], record=record)
        inherited = resolve_sample(environment=None, **kwargs)
        self.assertEqual(inherited.scene(7, record=record), (record["environment"], record["lighting"]))
        override = resolve_sample(environment="foundation-classroom", **kwargs)
        self.assertEqual(override.added_garment, "tee")
        self.assertEqual(override.scene(7, record=record)[0]["foundation"]["model"], "classroom")
        self.assertEqual(record["environment"]["name"], "kitchen")

    def test_generated_sample_matches_its_position_in_full_batch(self):
        def sample(i):
            return resolve_sample(i, 42, ["kitchen", "desk"], "all", None, "drape", {})
        batch = [sample(i) for i in range(8)]
        self.assertEqual(sample(5), batch[5])
        self.assertEqual({s.environment for s in batch[4:6]}, {"kitchen", "desk"})
        self.assertIsNone(sample(5).cloth_mode)

    def test_room_up_and_legacy_surface_anchor(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from render_handspotter import ROOM_PIVOT, scene_setup, upright_layout

        for seed in range(8):
            layout, _ = scene_setup(seed, 2, "foundation-classroom", [], True)
            np.testing.assert_allclose(Rotation.from_quat(layout["rotation"]).apply([0, 1, 0]), [0, 1, 0])
            self.assertEqual(upright_layout(layout), layout)

        tilted = Rotation.from_euler("YXZ", [.3, .51, -.12])
        legacy = dict(layout, rotation=tilted.as_quat().tolist())
        level = upright_layout(legacy)
        rotation = Rotation.from_quat(level["rotation"])
        np.testing.assert_allclose(rotation.apply([0, 1, 0]), [0, 1, 0])
        np.testing.assert_allclose(rotation.apply(ROOM_PIVOT) + level["origin"],
                                   tilted.apply(ROOM_PIVOT) + legacy["origin"])
