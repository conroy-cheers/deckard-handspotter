"""Export a folded Handspotter checkpoint as integer and decoded ONNX."""
import argparse
import json
from pathlib import Path

import onnx
import torch

from handspotter_model import dsp_fast_path_layers
from handspotter_torch import Handspotter, exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = Handspotter()
    model.fold_normalization()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    raw, decoded = exports(model)
    args.out.mkdir(parents=True)
    for variant, graph in (("integer", raw), ("dequantized", decoded)):
        onnx.save(graph, args.out / f"handspotter-frame-{variant}.onnx")
    fast_path = dsp_fast_path_layers(raw)
    (args.out / "dsp.json").write_text(json.dumps(dict(fast_path=fast_path), indent=2) + "\n")
    slow = [name for name, qualifies in fast_path.items() if not qualifies]
    print(f"Hexagon staged U16 path: {len(fast_path) - len(slow)}/{len(fast_path)} convolutions"
          + (f"; exact integer arithmetic for {', '.join(slow)}" if slow else ""))


if __name__ == "__main__":
    main()
