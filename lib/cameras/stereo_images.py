"""Resize calibrated grayscale images and measure visible hand extents."""
import cv2
import numpy as np

from handspotter_data import SIDES


def resize_gray(image, width, height):
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty grayscale byte image")
    h, w = image.shape
    if h % height == 0 and w % width == 0:
        dy, dx = h // height, w // width
        area = dx * dy
        total = image.reshape(height, dy, width, dx).sum((1, 3), dtype=np.uint64)
        return ((total + area // 2) // area).astype(np.uint8)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def visible_boxes(mask):
    output={}
    for side,label in zip(SIDES,(1,2)):
        y,x=np.where(mask==label)
        if not len(x):
            continue
        rectangle=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        output[side]=dict(box_xyxy=rectangle,visible_pixels=len(x))
    return output
