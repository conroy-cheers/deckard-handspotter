"""Evaluate an integer or decoded Handspotter ONNX model."""
import argparse
from pathlib import Path

import onnx

from handspotter_data import write_json
from handspotter_model import PORTS, check_interface, decoded_graph
from handspotter_shards import SPLITS, digest, load_split

from handspotter_metrics import evaluate_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--source", choices=("recorded", "synthetic"))
    parser.add_argument("--threshold", type=float, default=.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.threads < 1 or not 0 <= args.threshold <= 1:
        parser.error("Expected positive threads and a threshold between zero and one")
    graph = onnx.load(args.model)
    onnx.checker.check_model(graph, full_check=True)
    if [item.name for item in graph.graph.output] == list(PORTS):
        graph = decoded_graph(graph)
    else:
        check_interface(graph, decoded=True)
    data = load_split(args.dataset, args.split, args.source)
    write_json(args.out, {
        "split": args.split,
        "source": args.source,
        "threshold": args.threshold,
        "dataset_sha256": digest(args.dataset / "dataset.json"),
        "model_sha256": digest(args.model),
        "metrics": evaluate_export(graph, data, args.threads, args.threshold),
    })


if __name__ == "__main__":
    main()
