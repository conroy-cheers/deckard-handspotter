import unittest

import numpy as np

from handspotter_cameras import Fisheye624Camera, rig_camera, stereo_streams
from stereo_images import resize_gray, visible_boxes


class StereoTests(unittest.TestCase):
    def test_integral_resize_rounds_block_means_and_boxes_cover_masks(self):
        pixels = np.random.default_rng(7).integers(256, size=(1024, 1280), dtype=np.uint8)
        total = pixels.reshape(128, 8, 160, 8).sum((1, 3), dtype=np.uint32)
        np.testing.assert_array_equal(resize_gray(pixels, 160, 128), (total + 32) // 64)
        mask = np.zeros((120, 200), np.uint8)
        mask[5:25, 30:50] = 1
        self.assertEqual(visible_boxes(mask), {"left": dict(box_xyxy=[30, 5, 50, 25], visible_pixels=400)})

    def test_stream_selection_uses_labels(self):
        records = {k: {"calibration": {"label": v}} for k, v in
                   [("R7", "camera-slam-right"), ("C", "rgb"), ("L4", "camera-slam-left")]}
        self.assertEqual(stereo_streams(records), ("L4", "R7"))

    def test_lens_sweep_uses_source_calibration_at_target_scale(self):
        target = dict(label="front", image_width=320, image_height=256,
                      projection_model_type="kb4", image_circle_degrees=170,
                      intrinsics=[130, 132, 159.5, 127.5], distortion=[0.] * 8)
        kb4 = dict(target, image_width=200, image_height=144, intrinsics=[98, 96, 97, 73])
        fisheye = dict(projection_model_type="CameraModelType.FISHEYE624", image_width=200,
                       image_height=144, projection_params=[98, 96, 97, 73] + [0.] * 12)
        points = np.array([[0., 0., 1.], [.1, -.2, 1.]])
        for source in (kb4, fisheye):
            with self.subTest(source=source["projection_model_type"]):
                rig = dict(projection_model_type="lens_blend", source=source, frame=target, fraction=.25)
                views = [rig_camera(lens, np.eye(4), 2) for lens in (source, target, rig)]
                a, b, mixed = [view.project(points)[0] for view in views]
                np.testing.assert_allclose(mixed, .75*a + .25*b)

    def test_resized_fisheye_preserves_both_focal_formats(self):
        points = np.array([[0., 0., 1.], [.2, -.1, 1.]])
        for focal in ([90.], [90., 85.]):
            with self.subTest(focal=focal):
                cal = dict(projection_model_type="CameraModelType.FISHEYE624",
                           serial_number="fixture", label="left", image_width=240, image_height=160,
                           projection_params=[*focal, 119.5, 79.5, .01] + [0.] * 11)
                native = Fisheye624Camera(cal, np.eye(4))
                scaled = Fisheye624Camera(cal, np.eye(4), 2)
                pixels, _ = native.project(points)
                resized, _ = scaled.project(points)
                np.testing.assert_allclose(resized, (pixels + .5) * [320/240, 256/160] - .5)
                rays, valid, _ = scaled.rays()
                np.testing.assert_allclose(scaled.camera.eye_to_window(rays[valid]),
                                           np.stack(np.where(valid)[::-1], axis=-1), atol=1e-5)
