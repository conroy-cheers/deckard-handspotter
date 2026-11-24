"""Handspotter camera order, input tensor, and native image preparation."""
import ctypes as C
import os

import numpy as np

U8 = C.POINTER(C.c_uint8)


class Plane(C.Structure):
    _fields_ = [("data", U8), ("bytes", C.c_size_t), ("width", C.c_uint32),
                ("height", C.c_uint32), ("stride", C.c_uint32)]


ORDER = ("upper_left", "slam_left", "slam_right", "upper_right")
INPUT = (1, 4, 128, 160)


def support(camera, circle):
    if camera.model != "kb4" or not 0 < circle <= 360:
        raise ValueError(f"Invalid Handspotter lens for {camera.name}")
    theta = np.deg2rad(circle / 2)
    limit = theta * (1 + sum(k * theta**(2*i) for i,k in enumerate(camera.distortion[:4],1)))
    fx,fy,cx,cy = camera.intrinsics
    x = (np.arange(camera.width)-cx)/fx
    y = (np.arange(camera.height)-cy)/fy
    return np.ascontiguousarray(np.hypot(x[None,:],y[:,None]) <= limit, dtype=np.uint8)


class Preparation:
    def __init__(self, circles):
        if circles is None or not all(name in circles for name in ORDER):
            raise ValueError("Handspotter needs the Frame tracking-camera image circles")
        self.circles = circles
        self.camera_lenses = None
        self.library = C.CDLL(os.environ["FRAME_HAND_PIPELINE_LIBRARY"])
        self.resize = self.library.fhg_prepare_handspotter
        self.resize.argtypes = [C.POINTER(Plane), C.POINTER(U8), U8]
        self.resize.restype = C.c_int
        self.error = self.library.fhg_error
        self.error.argtypes = []
        self.error.restype = C.c_char_p

    def prepare(self, planes, geometry):
        cameras = geometry.cameras
        lenses = tuple(
            (c.name, c.width, c.height, c.model, tuple(c.intrinsics), tuple(c.distortion))
            for c in cameras
        )
        if self.camera_lenses != lenses:
            names = [camera.name for camera in cameras]
            if len(names) != 4 or set(names) != set(ORDER):
                raise ValueError("Handspotter needs the four calibrated Frame tracking cameras")
            self.order = [names.index(name) for name in ORDER]
            self.masks = [support(cameras[i], self.circles[name]) for name, i in zip(ORDER, self.order)]
            self.mask_pointers = (U8*4)(*[mask.ctypes.data_as(U8) for mask in self.masks])
            self.camera_lenses = lenses
        ordered = (Plane*4)(*[planes[i] for i in self.order])
        if any((plane.height,plane.width) != mask.shape for plane,mask in zip(ordered,self.masks,strict=True)):
            raise ValueError("Handspotter camera plane differs from its calibrated ROI")
        pixels = np.empty(INPUT,dtype=np.uint8)
        if self.resize(ordered,self.mask_pointers,pixels.ctypes.data_as(U8)):
            raise RuntimeError(self.error().decode())
        return pixels
