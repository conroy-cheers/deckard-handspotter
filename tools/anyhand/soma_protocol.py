"""Framed binary pose packets shared by the batch renderer and desktop host."""
import json
import struct
import subprocess

import numpy as np


LENGTH = struct.Struct("<I")
MAX_PACKET = 32 << 20


def packet(metadata, arrays):
    payload = bytearray()
    descriptors = {}
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        descriptors[name] = dict(offset=len(payload), shape=array.shape, dtype=array.dtype.str)
        payload.extend(memoryview(array).cast("B"))
        payload.extend(b"\0" * (-len(payload) % 4))
    header = json.dumps(dict(metadata, arrays=descriptors), separators=(",", ":"), allow_nan=False).encode()
    header += b" " * (-len(header) % 4)
    return LENGTH.pack(len(header)) + header + payload


def read_packet(stream):
    prefix = stream.read(LENGTH.size)
    if len(prefix) != LENGTH.size:
        raise EOFError("SOMA-X worker exited before sending a response")
    size = LENGTH.unpack(prefix)[0]
    if not LENGTH.size <= size <= MAX_PACKET:
        raise ValueError("Invalid SOMA-X response length")
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("SOMA-X worker sent an incomplete response")
    header = LENGTH.unpack_from(payload)[0]
    metadata = json.loads(payload[LENGTH.size:LENGTH.size+header])
    if "error" in metadata:
        raise ValueError(metadata["error"])
    arrays = {}
    for name, desc in metadata.pop("arrays").items():
        arrays[name] = np.ndarray(desc["shape"], dtype=desc["dtype"], buffer=payload,
                                  offset=LENGTH.size+header+desc["offset"])
    return metadata,arrays


class Worker:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()

    def request(self, **request):
        self.process.stdin.write(json.dumps(request, allow_nan=False).encode()+b"\n")
        self.process.stdin.flush()
        return read_packet(self.process.stdout)
