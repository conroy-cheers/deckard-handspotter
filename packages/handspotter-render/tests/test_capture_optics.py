import unittest

import numpy as np

from capture_optics import LensDecentering
from render_handspotter import detector_optics
from handspotter_cameras import rig_camera


class CaptureOpticsTests(unittest.TestCase):
    lens = LensDecentering(np.array([604., 502.]), np.array([5., -4.]), np.array([.004, -.004]))

    def test_projection_inverts_decentering_at_native_and_tensor_scales(self):
        for width, height in ((1280, 1024), (320, 256)):
            with self.subTest(width=width):
                src = np.stack(self.lens.nominal_grid(width, height), axis=-1)
                y, x = np.indices((height, width))
                projected = self.lens.project(src, width)
                np.testing.assert_allclose(projected, np.stack((x, y), axis=-1), atol=1.e-6, rtol=0)

    def test_gpu_uniforms_use_detector_pixel_coordinates_for_each_model(self):
        kb4 = dict(label="front", image_width=640, image_height=480,
                   projection_model_type="kb4", image_circle_degrees=170,
                   intrinsics=[260, 262, 319.5, 239.5], distortion=[.01] + [0.] * 7)
        fisheye = dict(image_width=320, image_height=240, projection_model_type="CameraModelType.FISHEYE624",
                       projection_params=[120, 125, 159.5, 118.5] + [.02] + [0.] * 11)
        blend = dict(projection_model_type="lens_blend", source=fisheye, frame=kb4, fraction=.3)
        for calibration in (kb4, fisheye, blend):
            with self.subTest(model=calibration["projection_model_type"]):
                camera = rig_camera(calibration, np.eye(4), 2)
                optics = detector_optics(camera, self.lens)
                self.assertEqual(optics["decenter"], self.lens.record())
                a = np.array([.37, -.26])
                radius = np.linalg.norm(a)
                ray = np.r_[a * np.sin(radius)/radius, np.cos(radius)]
                def pixel(endpoint):
                    radial = np.polynomial.polynomial.polyval(radius**2, [1., *endpoint["coefficients"][:6]])
                    return np.array(endpoint["c"]) + a * radial * endpoint["f"]
                u = optics["fraction"]
                expected = camera.project(ray[None])[0][0]
                np.testing.assert_allclose((1-u)*pixel(optics["source"]) + u*pixel(optics["frame"]), expected)


if __name__ == "__main__":
    unittest.main()
