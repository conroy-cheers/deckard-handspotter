"""End-to-end CPU smoke workflow using generated inputs."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import av
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PIL import Image

from handspotter import ORDER
from handspotter_data import QUAD_STREAMS, load_synthetic
from handspotter_shards import load_split
from handspotter_model import PORTS, check_contract, check_interface
from workspace import Workspace, ingest


def write_json(path, data):
    path.write_text(json.dumps(data) + "\n")


def calibration(root):
    axes = {"plus_x": [1, 0, 0], "plus_z": [0, 0, 1], "position": [0, 0, 0]}
    write_json(root / "device.json", axes)
    write_json(root / "xrservice.json", {"cameras": [
        dict(sourceCamera=name, width=160, height=128, imageCircleDiameterInDeg=170,
             extrinsics={**axes, "position": [30 * i, 0, 0]},
             intrinsics=[dict(cameraModel="kb", k1=0, k2=0, k3=0, k4=0,
                              fx=90, fy=90, cx=80, cy=64)])
        for i, name in enumerate(ORDER)
    ]})


def recording(root, name, code):
    take = root / name
    take.mkdir()
    rows = []
    tbase = Fraction(1, 1_000_000)
    with av.open(str(take / "tracking.nut"), mode="w") as container:
        streams = []
        for camera in ORDER:
            stream = container.add_stream("rawvideo", rate=60)
            stream.width, stream.height, stream.pix_fmt = 160, 128, "gray"
            stream.codec_context.time_base = tbase
            stream.metadata["title"] = camera
            streams.append(stream)
        for index, (camera, stream) in enumerate(zip(ORDER, streams, strict=True)):
            frame = av.VideoFrame(160, 128, "gray")
            plane = frame.planes[0]
            plane.update(bytes([code + index]) * plane.buffer_size)
            frame.pts, frame.time_base = index, tbase
            container.mux(stream.encode(frame))
            rows.append(dict(group=1, track=index, camera=camera, pts_us=index,
                             sensor_raw_s=100, end_exposure_raw_s=0, arrival_raw_s=0,
                             exposure_s=0, observer_mono_ns=code, clock_raw_ns=0,
                             clock_mono_ns=0, clock_uncertainty_ns=0))
    with (take / "timestamps.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = root / f"{name}-labels"
    assert ingest(take, output, 0, None, 1) == 1
    workspace = Workspace(output)
    for camera in ORDER:
        workspace.set_label(1, camera, "left", "box", [60, 40, 100, 90])
    workspace.set_label(1, ORDER[0], "right", "absent")
    workspace.export(output / "labels")
    return output


def run():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "dataset.toml"
        calibration(root)
        parts = ['''[[synthetic]]
split = "training"
preset = "procedural"
xrservice = "xrservice.json"
device_config = "device.json"
seed = 899
count = 3
face_size = 96
''']
        for index, split in enumerate(("training", "validation", "test")):
            take = recording(root, split, 40 + 10 * index)
            parts.append(f'''[[recorded]]
split = "{split}"
workspace = "{take.name}"
xrservice = "xrservice.json"
device_config = "device.json"
''')
        mismatched = root / "mismatched"
        config.write_text(parts[1] + 'labels = "../validation-labels/labels/groups.json"\n')
        result = subprocess.run(["handspotter", "prepare", "--config", config, "--out", mismatched],
                                capture_output=True, text=True)
        assert result.returncode and "belongs to a different take" in result.stderr, result.stderr
        assert not mismatched.exists()

        config.write_text("\n".join(parts))
        data, model, export, scores = (root / name for name in ("data", "model", "export", "scores.json"))

        def command(*args):
            subprocess.run(["handspotter", *map(str, args)], check=True)

        subprocess.run(["handspotter-prepare", "--config", config, "--out", data], check=True)
        images, boxes, valid = load_split(data, "training")
        assert images.shape == (4, 4, 128, 160)
        assert images[1:].any()
        np.testing.assert_equal(images[0, :, 64, 80], [40, 41, 42, 43])
        np.testing.assert_equal(boxes[0, 2, 0], [479.5, 319.5, 799.5, 719.5])
        np.testing.assert_equal(valid[0, :, 1], [True, False, False, False])

        render = data / "synthetic-0000-render"
        # A viewport request stages one scene but retains its original RNG index.
        poses = data / "synthetic-0000-poses"
        preview = root / "preview"
        command("render", "--rig-json", data / "synthetic-0000-rig.json", "--source", poses / "poses.npz",
                "--topology", poses / "topology.npz", "--count", 3, "--seed", 899, "--face-size", 96,
                "--handspotter-render-scale", 2, "--tensors-only", "--environment", "all", "--capture-sample", 2, "--out", preview)
        np.testing.assert_array_equal(np.load(preview / "handspotter.npy"), np.load(render / "handspotter.npy")[2:3])
        preview_setup = json.loads((preview / "dataset.json").read_text())["records"][0]["render_setup"]
        batch_record = json.loads((render / "dataset.json").read_text())["records"][2]
        assert preview_setup == batch_record["render_setup"]
        assert preview_setup["index"] == 2 and preview_setup["environment"] == batch_record["environment"]["name"]
        _, direct_boxes, _ = load_synthetic(render, "training", 0, 0)
        np.testing.assert_array_equal(boxes[1:], direct_boxes)

        # A render with no hands still has a split identity.
        empty = root / "empty-render"
        empty.mkdir()
        write_json(empty / "rig.json", {})
        write_json(empty / "dataset.json", dict(stream_order=QUAD_STREAMS, records=[
            dict(hands={}, views={name: dict(visible_hands={}) for name in QUAD_STREAMS})
        ]))
        np.save(empty / "handspotter.npy", images[1:2])
        duplicate = root / "duplicate.toml"
        duplicate.write_text("".join(
            f'[[synthetic]]\nsplit = "{split}"\npath = "empty-render"\n'
            for split in ("training", "test")
        ))
        result = subprocess.run(["handspotter-prepare", "--config", duplicate, "--out", root / "duplicate"],
                                capture_output=True, text=True)
        assert result.returncode and "crosses dataset splits" in result.stderr, result.stderr

        # Two different sensors, with stream keys independent of the device.
        rig = json.loads((data / "synthetic-0000-rig.json").read_text())
        left, right = (rig["cameras"][name] for name in ("front_left", "front_right"))
        right["source_calibration"] = dict(
            label="auxiliary", serial_number="fixture", image_width=192, image_height=144,
            projection_model_type="CameraModelType.FISHEYE624",
            projection_params=[99., 101., 95.5, 71.5] + [0.] * 12,
        )
        stereo = root / "stereo.json"
        write_json(stereo, dict(stream_order=["sensor-B", "sensor-A"],
                                cameras={"sensor-A": right, "sensor-B": left}))
        output = root / "stereo-render"
        command("render", "--calibration", stereo, "--source", poses / "poses.npz",
                "--topology", poses / "topology.npz", "--seed", 29, "--count", 1,
                "--face-size", 96, "--negative-every", 0, "--out", output)
        manifest = json.loads((output / "dataset.json").read_text())
        assert manifest["stream_order"] == ["sensor-B", "sensor-A"]
        for stream, size in [("sensor-B", (160, 128)), ("sensor-A", (192, 144))]:
            with Image.open(output / manifest["records"][0]["views"][stream]["image"]) as image:
                assert image.size == size
        assert not (output / "handspotter.npy").exists()

        config = root / "train.toml"
        config.write_text('''dataset = "data"
out = "model"
device = "cpu"
threads = 1
epochs = 1
qat_epochs = 1
batch = 4
resume = true
recorded_fraction = 0.5
brightness = 0.05
''')
        command("train", "--config", config)
        original = (model / "handspotter-frame-integer.onnx").read_bytes()
        # Re-create the state at an interruption between float training and QAT.
        (model / "qat-resume.pt").unlink()
        command("train", "--config", config)
        assert original == (model / "handspotter-frame-integer.onnx").read_bytes()
        config = root / "dataset.toml"
        command("export", model / "folded.pt", "--out", export)
        integer = export / "handspotter-frame-integer.onnx"
        decoded = export / "handspotter-frame-dequantized.onnx"
        command("evaluate", integer, "--dataset", data, "--out", scores)
        assert json.loads(scores.read_text())["metrics"]["labels"] == 4
        assert integer.read_bytes() == (model / integer.name).read_bytes()

        raw = check_contract(onnx.load(integer))
        check_interface(onnx.load(decoded), decoded=True)
        fast_path = json.loads((export / "dsp.json").read_text())["fast_path"]
        assert list(fast_path) == [f"conv{i}" for i in range(16)]
        assert fast_path == json.loads((model / "dsp.json").read_text())["fast_path"]
        # The contract rejects operators outside the lowering's set and INT32 overflow.
        unsupported = onnx.load(integer)
        unsupported.graph.node.append(onnx.helper.make_node("Relu", ["conv0_acc"], ["conv0_relu"]))
        overflow = onnx.load(integer)
        bias = next(item for item in overflow.graph.initializer if item.name == "conv0_bias")
        bias.CopyFrom(numpy_helper.from_array(np.full_like(numpy_helper.to_array(bias), 2**31 - 1), bias.name))
        for broken, reason in ((unsupported, "operators ['Relu']"), (overflow, "conv0 is not a fixed-point")):
            try:
                check_contract(broken)
            except ValueError as error:
                assert reason in str(error), error
            else:
                raise AssertionError(f"Contract accepted a graph with {reason}")
        grids = {item.tensor_name: {f.key: f.value for f in item.quant_parameter_tensor_names}
                 for item in raw.graph.quantization_annotation}
        constants = {item.name: numpy_helper.to_array(item) for item in raw.graph.initializer}

        def infer(path):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.log_severity_level = 3
            return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(
                None, {"images": images[:1]})

        for (name, shape), codes, values in zip(PORTS.items(), infer(integer), infer(decoded), strict=True):
            assert codes.shape == values.shape == shape and codes.dtype == np.uint8
            scale = constants[grids[name]["SCALE_TENSOR"]]
            zero = constants[grids[name]["ZERO_POINT_TENSOR"]]
            np.testing.assert_allclose(values, (codes.astype(np.float32) - zero) * scale)

        config.write_text("\n".join(parts + [parts[1].replace('split = "training"', 'split = "test"')]))
        result = subprocess.run(["handspotter", "prepare", "--config", config, "--out", root / "leaked"],
                                capture_output=True, text=True)
        assert result.returncode and "crosses dataset splits" in result.stderr


if __name__ == "__main__":
    run()
