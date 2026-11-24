"""Seeded optical and shutter augmentation in the synthetic camera pipeline."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LensDecentering:
    center: np.ndarray
    offset: np.ndarray
    tangential: np.ndarray

    @classmethod
    def sample(cls, rng, camera):
        axis = camera.origin + camera.transform[:3, 2]
        center, _ = camera.project(axis[None, :])
        scale = camera.width / 1280
        center_native = (center[0] + .5) / scale - .5
        return cls(center_native, rng.uniform(-5., 5., 2), rng.uniform(-.004, .004, 2))

    def displacement(self, x, y, scale):
        # Brown tangential residual on the nominal fisheye image plane. All
        # parameters are defined in the 1280×1024 window, even for small Handspotter rasters.
        center = (self.center + .5) * scale - .5
        norm = 640 * scale
        u, v = (x - center[0]) / norm, (y - center[1]) / norm
        p1, p2 = self.tangential
        r2 = u*u + v*v
        return (scale*self.offset[0] + norm*(2*p1*u*v + p2*(r2 + 2*u*u)),
                scale*self.offset[1] + norm*(p1*(r2 + 2*v*v) + 2*p2*u*v))

    def project(self, pixels, width):
        x, y = np.moveaxis(pixels, -1, 0)
        dx, dy = self.displacement(x, y, width / 1280)
        return np.stack((x + dx, y + dy), axis=-1)

    def nominal_grid(self, width, height):
        y, x = np.indices((height, width), dtype=np.float64)
        src_x, src_y = x, y
        # With the sampled coefficients the displacement slope is small.
        for _ in range(6):
            dx, dy = self.displacement(src_x, src_y, width / 1280)
            src_x, src_y = x - dx, y - dy
        return src_x, src_y

    def record(self):
        return dict(center_native_xy_px=self.center.tolist(), shift_native_xy_px=self.offset.tolist(),
                    tangential_p1_p2=self.tangential.tolist(), normalization_native_px=640)


def sample_motion(args, index, hands):
    def rng(effect):
        return np.random.default_rng(np.random.SeedSequence([args.seed, index, 15, effect]))

    def rotation(random, low, high):
        axis = random.normal(size=3)
        return (axis * (random.uniform(low, high) / np.linalg.norm(axis))).tolist()

    motion = {}
    if args.arm_motion_blur:
        arms = {}
        for side, name in enumerate(("left", "right")):
            if name in hands:
                random = rng(side)
                arm = hands[name]["arm"]
                points = np.asarray(hands[name]["keypoints_pair_m"])
                arms[name] = dict(shoulder=arm["shoulder_pair_m"], elbow=arm["elbow_pair_m"],
                                  wrist=arm["wrist_pair_m"], palm=points[[5, 9, 13, 17]].mean(axis=0).tolist(),
                                  rotation_deg=rotation(random, .4, 1.8) if random.random() < .12 else [0., 0., 0.])
        if any(any(arm["rotation_deg"]) for arm in arms.values()):
            motion["arms"] = arms
    if args.camera_motion_blur:
        random = rng(2)
        if random.random() < .12:
            motion["camera"] = dict(rotation_deg=rotation(random, .12, .45),
                                    translation_m=random.uniform(-.001, .001, 3).tolist())
    return motion


def sample_optical_blur(seed, index, eye):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, eye, 16]))
    return float(rng.uniform(.55, 1.65)) if rng.random() < .75 else 0.
