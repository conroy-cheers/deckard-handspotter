"""Evaluate an integer or decoded Handspotter ONNX model."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

from handspotter_data import detector_metrics


def evaluate_export(artifact, data, workers, threshold=.2):
    images,box,valid = data
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    runtime = ort.InferenceSession(artifact.SerializeToString(), options, providers=["CPUExecutionProvider"])
    def infer(sample):
        center,size,score = runtime.run(["centers","sizes","scores"],{"images":sample[None]})
        return center.reshape(4,2,2)*1280,size.reshape(4,2)*640,score.reshape(4,2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(infer,images))
    return detector_metrics(*(np.stack(values) for values in zip(*predictions,strict=True)),box,valid,threshold=threshold)
