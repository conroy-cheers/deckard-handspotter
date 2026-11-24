# Handspotter

Handspotter is a four-camera Steam Frame detector that predicts
boxes and confidence scores for the wearer's left and right hands. It is
designed to run on the Frame's Hexagon DSP. It includes headset recording, a
workstation labeler, procedural rendering, mixed-data preparation, training
from random initialization, quantization and integer ONNX export.

The repository's output is a uint8 ONNX model built only from integer
operators and fixed-point requantization that the Frame's QNN lowering
accepts. Every export is checked against that contract and reports which
convolutions qualify for the DSP's faster staged U16 path. Compiling the model
with QNN and running it on the headset happen outside this repository.

The Nix flake provides CPU, ROCm, and CUDA training packages on x86_64 Linux.
The labeler also builds on AArch64 Linux, so takes can be ingested on the
headset. Recording happens on the headset with tooling outside this
repository; the [take format](docs/frame-takes.md) specifies what the labeler
ingests.

- [Recorded take format](docs/frame-takes.md)
- [Generate a mixed dataset, train and export](docs/handspotter-training.md)
- [Render synthetic data and use AnyHand Studio](tools/anyhand/README.md)
- [Final checkpoint](docs/handspotter-model-card.md) and [reference results](docs/handspotter-release-validation.md)

```sh
nix run .#anyhand-studio
nix run .#anyhand-studio -- batch --out ./starter-render
nix run . -- --help
nix run . -- prepare --config /path/to/dataset.toml --out /path/to/dataset
nix run . -- train --config /path/to/train.toml
nix flake check
```

The CPU check generates a temporary take and synthetic scene,
runs the mixed-data pipeline, and verifies the exported ONNX with ONNX Runtime.
Use the [example configs](examples) with your own data. For older fine-tuning
and analysis scripts, `nix develop` provides the Python dependencies. AnyHand
rendering has a separate `anyhand-pbr` development shell.

### Standalone workflow packages

On x86_64 Linux, run any stage directly with `nix run .#PACKAGE -- --help`.

| Stage | Package |
| --- | --- |
| Export a camera rig | `handspotter-rig` |
| Render synthetic training images | `handspotter-render` |
| Prepare recorded and synthetic shards | `handspotter-prepare` |
| Train a detector | `handspotter-train` |
| Export a folded checkpoint | `handspotter-export` |
| Evaluate an ONNX model | `handspotter-evaluate` |
| Warp a HOT3D clip or build a corpus | `hot3d-frame`, `hot3d-frame-corpus` |

Each command has its source and Nix recipe under [packages/](packages).
The commands use [shared libraries](lib) for dataset I/O, camera geometry,
model export, evaluation, and HOT3D conversion. Standalone
`handspotter-prepare` includes the renderer for recipes that generate new images.

### Select a training or labeling backend

On x86_64 Linux, the `cpu`, `rocm`, and `cuda` package sets each expose
`handspotter`, `handspotter-train`, and the SAM-enabled `frame-hand-labeler`.
Choose a backend independently for each command. PyTorch uses the device
name `cuda` for both NVIDIA CUDA and AMD ROCm.

```sh
nix run .#cpu.handspotter-train -- --config train.toml --device cpu
nix run .#rocm.handspotter-train -- --config train.toml --device cuda
nix run .#cuda.frame-hand-labeler -- propagate workspace --device cuda
```

Flake consumers can select, for example,
`handspotter.legacyPackages.x86_64-linux.cuda.handspotter-train`.
The ordinary `packages` output also exposes the GPU variants with
`-rocm` and `-cuda` suffixes. The CUDA package set enables NVIDIA’s
unfree toolkit dependencies. Preparation, rendering, export, and ONNX
evaluation use the same packages across all three suites.

Project code is licensed under MIT. The included checkpoint weights are licensed under
CC BY-NC-SA 4.0; see the [model card](docs/handspotter-model-card.md#license).
External assets and dependencies retain their own terms. [License](LICENSE)
