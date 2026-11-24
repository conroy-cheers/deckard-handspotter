# Train Handspotter on synthetic hands and your own Frame footage

The x86_64 Linux flake provides a headless renderer, dataset preparation, training, ONNX export and evaluation. A [recorded take](frame-takes.md) holds four synchronized native grayscale streams and a timestamp CSV. Allow about 10 GB per minute for a take. All generated data lives in the directory you choose. The published model is described in the [Handspotter model card](handspotter-model-card.md).

## Record and label

1. Record takes on the Frame in the [take format](frame-takes.md).
2. Copy the take and the headset's `/persist/xrservice.json` and `/persist/device_config.json` to your workstation. Keep calibrations alongside the takes they describe.
3. Run the commands below. In the editor, mark anatomical left and right separately in every view you review. Mark **Absent** as a reviewed negative and **Unknown** when you cannot judge. Proposals from the optional SAM tracker require review to enter this export. Use an independent take for validation and another for test. [Labeler guide](../tools/labeler/README.md)

```sh
nix run .#frame-hand-labeler-tools -- ingest /path/to/take ./session-a-labels --hz 10
nix run .#frame-hand-labeler-tools -- serve ./session-a-labels
nix run .#frame-hand-labeler-tools -- export ./session-a-labels
```

## Prepare one mixed dataset

Copy [dataset.toml](../examples/dataset.toml) to the directory that holds your label workspaces. Specify each recording's workspace, split and calibration. Use the same `session` value for clips cut from the same physical session.

The example uses the built-in procedural pose generator, so it needs only your recordings and headset calibration. The procedural hands are coarse training and pipeline examples. Supply a detailed pose bank and topology to improve the range of appearances. A pose-bank NPZ contains `ids`, wrist-local vertices `[N,V,3]` and joints `[N,21,3]`. A topology NPZ contains oriented triangle indices `faces` with a single open wrist boundary. The renderer also accepts existing AnyHand posed archives and the corresponding topology OBJ. [Pose bank and PBR rendering details](../tools/anyhand/README.md)

```sh
nix run .#handspotter -- prepare --config /path/to/dataset.toml --out /path/to/dataset
```

The preparation command derives a four-view KB4 renderer rig from the supplied Frame calibration, renders the requested synthetic groups, applies the deployed native preprocessing to your recordings, and writes one manifest with counts, hashes and provenance. For a rig you supply directly, set `rig` in the synthetic entry. To reuse a previous render, set `path` to its output directory instead of supplying renderer inputs. Run `nix run .#handspotter -- render --help` for scene, pose, optics, exposure and occlusion controls. The PBR renderer uses Chromium WebGL 2 over Vulkan. On a NixOS AMD workstation with multiple Vulkan ICDs, select the RADV ICD if necessary. Set `ANYHAND_SOFTWARE_RENDER=1` to use Chromium's software renderer for small runs.

Each shard has NumPy arrays for `images` (uint8 `[N,4,128,160]`), `boxes` (float32 `[N,4,2,4]`) and `valid` (bool `[N,4,2]`). The view order is upper-left, SLAM-left, SLAM-right, upper-right. The hand order is anatomical left, right. Boxes use the 1280×1024 reference pixel-centre coordinate system. An absent hand has a valid zero box. Unknown labels have their validity bit cleared. Training requires all three splits and at least one labeled hand in each.

## Train, export and evaluate

Copy [train.toml](../examples/train.toml) beside the dataset and set `dataset` and `out`. The example runs on CPU. For a smoke run, use one float epoch and one QAT epoch. For a full run, choose group counts, batch size and schedule based on available memory and held-out performance. The CLI accepts overrides for the TOML settings. Select `.#rocm.handspotter-train` for AMD or `.#cuda.handspotter-train` for NVIDIA. For either package, pass `--config /path/to/train.toml --device cuda` to train on the GPU. Set `recorded_fraction` to control the expected recorded share of randomly sampled training groups. Omit it to shuffle the complete dataset each epoch.

```sh
nix run .#handspotter -- train --config /path/to/train.toml
nix run .#handspotter -- export /path/to/model/folded.pt --out /path/to/export
nix run .#handspotter -- evaluate /path/to/export/handspotter-frame-integer.onnx \
  --dataset /path/to/dataset --source recorded --out /path/to/scores.json
```

The graph is generated from the source architecture and learned parameters. Training records software versions, seed, optimizer settings, held-out scores and model artifacts. The integer graph takes one four-camera group and returns uint8 centres, sizes, confidence and `conv11` features. A downstream temporal model can optionally consume the features. The decoded graph returns the same ports in real values. Export checks the integer graph against the contract the Frame's Hexagon lowering relies on: opset 21 integer operators only, static shapes, a uint8 grid on every input and output, and fixed-point convolutions whose INT32 intermediates cannot overflow. It also writes `dsp.json`, which lists the convolutions that qualify for the DSP's faster staged U16 path; the others remain exact but run slower on the device. Training writes the same `dsp.json` beside its exported models. Evaluate held-out recorded groups with `--source recorded` and use `--source synthetic` to report the renderer cohort separately. Omit the source filter to evaluate the whole split. Confirm detector quality on private Frame footage before selecting a model for deployment.

```sh
nix build --no-link .#handspotter.tests.smoke
```

The CPU workflow check generates temporary takes and renders a procedural shard with software WebGL, runs the mixed dataset adapter, trains from scratch and checks independently exported ONNX in ONNX Runtime.
