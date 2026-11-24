"""Run a workspace's model commands without blocking its editor."""
from collections import deque
from importlib.util import find_spec
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Lock, Thread


class ModelJobs:
    def __init__(self, workspace, entry):
        self.workspace = workspace
        self.entry = entry
        self.lock = Lock()
        self.process = None
        self.closed = False
        self.serial = 0
        self.kind = None
        self.state = "idle"
        self.log = deque(maxlen=6)
        weights = Path(os.environ.get("FRAME_HAND_LABELER_WEIGHTS", ""))
        self.capabilities = {
            "sam": bool(find_spec("comfy") and find_spec("torch")
                        and weights.suffix == ".safetensors" and weights.is_file()),
            "terra": shutil.which("codex") is not None,
        }

    def settings(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected model settings")
        kind = data.get("kind")
        first, last = data.get("first"), data.get("last")
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(self.workspace.groups):
            raise ValueError("Choose a group range inside the clip")
        if kind not in self.capabilities or not self.capabilities[kind]:
            raise ValueError("SAM requires the model-enabled package. Terra requires Codex CLI on PATH.")
        root = str(self.workspace.root.resolve())
        scope = ["--start-index", str(first-1), "--limit", str(last-first+1)]
        if kind == "sam":
            cameras, radius, device = data.get("cameras"), data.get("radius"), data.get("device")
            if (not isinstance(cameras, list) or not cameras or not set(cameras) <= set(self.workspace.cameras)
                    or type(radius) is not int or radius < 1 or device not in ("auto", "cpu", "cuda")):
                raise ValueError("Select cameras, a device and a positive tracking radius")
            args = ["propagate", root, *scope, "--radius", str(radius), "--device", device]
            for camera in cameras:
                args.extend(("--camera", camera))
            return args
        hz = data.get("hz")
        if type(hz) not in (float, int) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("Enter a positive Terra frame rate")
        return ["terra", root, *scope, "--hz", str(hz)]

    def start(self, data):
        args = self.settings(data)
        with self.lock:
            if self.closed:
                raise ValueError("The workspace is closing")
            if self.process is not None:
                raise ValueError("A model run is already active")
            self.process = subprocess.Popen(
                [sys.executable, "-B", str(self.entry), *args], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True,
            )
            self.serial += 1
            self.kind = data["kind"]
            self.state = "running"
            self.log.clear()
            Thread(target=self.collect, args=(self.process,), daemon=True).start()

    def collect(self, process):
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    if line.strip():
                        self.log.append(line.strip()[-800:])
        code = process.wait()
        with self.lock:
            if self.state == "stopping":
                self.state = "cancelled"
            else:
                self.state = "failed" if code else "complete"
            self.process = None

    def stop(self):
        with self.lock:
            process = self.process
            if process is None or self.state == "stopping":
                return
            self.state = "stopping"
            self.signal(process, signal.SIGTERM)
        Thread(target=self.reap, args=(process,), daemon=True).start()

    @staticmethod
    def signal(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def reap(self, process):
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.signal(process, signal.SIGKILL)

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
        if process is not None:
            self.stop()
            self.reap(process)

    def status(self):
        with self.lock:
            return dict(serial=self.serial, kind=self.kind, state=self.state, log=list(self.log))
