import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hot3d_warp import build_warp
from handspotter_cameras import FrameHandspotterCamera


class Hot3DFrameTests(unittest.TestCase):
    @staticmethod
    def warp(rotation, translation):
        destination = np.eye(4)
        destination[:3, :3] = rotation
        destination[:3, 3] = translation
        intrinsic = [400., 400., 639.5, 511.5]
        frame = FrameHandspotterCamera(dict(label="slam_left", image_width=1280, image_height=1024,
                                   intrinsics=intrinsic, distortion=[0.]*8, image_circle_degrees=180),
                               destination, render_scale=2)
        pose = dict(quaternion_wxyz=[1., 0., 0., 0.], translation_xyz=[0., 0., 0.])
        source = dict(calibration=dict(label="camera-slam-left", serial_number="fixture",
                                      image_width=1280, image_height=1024,
                                      projection_model_type="CameraModelType.FISHEYE624",
                                      projection_params=[400., 639.5, 511.5]+[0.]*12),
                      T_world_from_camera=pose)
        return build_warp(source, np.eye(4), frame)

    def test_inverse_map_rotates_pixels_and_annotations_together(self):
        warp = self.warp(Rotation.from_euler("z", 90, degrees=True).as_matrix(), [0., 0., 0.])
        np.testing.assert_allclose((warp.map_x[100, 180], warp.map_y[100, 180]),
                                   (749.5, 593.5), atol=1.e-4)
        box, valid = warp.label([713, 553, 786, 634], 1.)
        self.assertTrue(valid)
        self.assertLess(box[0], 721.5)
        self.assertGreater(box[2], 721.5)
        self.assertLess(box[1], 401.5)
        self.assertGreater(box[3], 401.5)
        source = np.zeros((1024, 1280), np.uint8)
        source[553:634, 713:786] = 255
        result = warp.pixels(source)
        self.assertEqual(result[50, 90], 255)
        self.assertEqual(result[64, 80], 0)

    def test_translations_leave_the_warp_unchanged(self):
        rotation = Rotation.from_euler("xyz", [10, 25, -45], degrees=True).as_matrix()
        a = self.warp(rotation, [0., 0., 0.])
        b = self.warp(rotation, [1.2, -.3, 4.1])
        np.testing.assert_array_equal(a.map_x, b.map_x)
        np.testing.assert_array_equal(a.map_y, b.map_y)
        self.assertFalse(a.label([550, 430, 700, 610], 0.)[1])
        self.assertTrue(a.label(None, 0.)[1])


if __name__ == "__main__":
    unittest.main()
