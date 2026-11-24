"""Read pixels and annotations from HOT3D clip archives."""
from contextlib import contextmanager
import json
import re
import tarfile

import cv2
import numpy as np


TRACKING_STREAMS = ("1201-1", "1201-2")
FRAME_FILE = re.compile(r"(\d{6})\.info\.json")


class Clip:
    def __init__(self,archive):
        self.archive=archive

    def frame_ids(self):
        return sorted(int(match[1]) for member in self.archive.getnames()
                      if (match := FRAME_FILE.fullmatch(member)))

    def data(self,member):
        with self.archive.extractfile(member) as file:
            return file.read()

    def json(self,member):
        return json.loads(self.data(member))

    def pixels(self,index,stream):
        name=f'{index:06d}.image_{stream}.jpg'
        return cv2.imdecode(np.frombuffer(self.data(name),np.uint8),cv2.IMREAD_GRAYSCALE)


@contextmanager
def open_clip(path):
    with tarfile.open(path) as file:
        yield Clip(file)
