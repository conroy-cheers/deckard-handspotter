"""Differentiable integer tail for fine-tuning an exported Handspotter ONNX model."""
import copy

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from handspotter_integer import accumulator_ranges, convolution_clusters, fixed_point_epilogue
from handspotter_model import DECODED_PORTS, PORTS, check_contract, check_interface


FEATURE = "conv10"


def rounded(x, forward):
    """Use the exact discrete forward value and its continuous-input gradient."""
    return forward.detach() + (x - x.detach())


class IntegerConv(nn.Module):
    def __init__(self, model, cluster, constants):
        super().__init__()
        conv, add = (model.graph.node[i] for i in cluster["members"][:2])
        prefix = fixed_point_epilogue(model,cluster,constants)
        if prefix is None or any(constants[name].item() for name in conv.input[2:]):
            raise ValueError("Expected a zero-origin Handspotter integer convolution")
        self.weight_name = conv.input[1]
        self.bias_name = add.input[1]
        self.padding = constants[self.weight_name].shape[-1]//2
        self.register_buffer("original_weight",torch.from_numpy(constants[self.weight_name].astype(np.float32)))
        self.register_buffer("original_bias",torch.from_numpy(prefix["source_bias"].astype(np.float64)))
        for field in ("divisor","multiplier","rounding"):
            self.register_buffer(field,torch.from_numpy(prefix[field].astype(np.int64)))
        self.register_buffer("gain",(self.multiplier/(self.divisor.double()*16384)).float())
        self.weight_delta = nn.Parameter(torch.zeros_like(self.original_weight))
        self.bias_delta = nn.Parameter(torch.zeros(len(self.original_bias)))

    def quantized(self):
        weight = self.original_weight+self.weight_delta
        weight = rounded(weight,weight.round().clamp(-128,127))
        bias = self.original_bias+self.bias_delta.double()/self.gain.double()
        return weight,rounded(bias,bias.round())

    def forward(self, image):
        weight,bias = self.quantized()
        # The export bound below keeps every raw convolution sum within the
        # exactly representable integer range of FP32. Bias and shifts use I64.
        accumulator = F.conv2d(image,weight,padding=self.padding)
        broadcast = (1,-1,1,1)
        with torch.no_grad():
            biased = accumulator.to(torch.int64)+bias.to(torch.int64).reshape(broadcast)
            first = torch.div(biased,self.divisor.reshape(broadcast),rounding_mode="floor")
            product = first*self.multiplier.reshape(broadcast)+self.rounding.reshape(broadcast)
            discrete = torch.div(product,16384,rounding_mode="floor").clamp(0,255).float()
        continuous = accumulator*self.gain.reshape(broadcast)+self.bias_delta.reshape(broadcast)
        return rounded(continuous,discrete)

    def export(self, constants):
        weights,bias = (value.detach().cpu().numpy() for value in self.quantized())
        weights = weights.astype(np.int8)
        low,high = accumulator_ranges(weights,0,1,len(bias))
        if max(-int(low.min()),int(high.max()))>2**24:
            raise ValueError("Convolution exceeds exact FP32 accumulation range")
        i32 = np.iinfo(np.int32)
        if np.any(bias<i32.min) or np.any(bias>i32.max):
            raise ValueError("Updated bias overflows INT32")
        bias = bias.astype(np.int64)
        for accum in (low+bias,high+bias):
            product = (accum//self.divisor.cpu().numpy())*self.multiplier.cpu().numpy()
            for value in (accum,product,product+self.rounding.cpu().numpy()):
                if np.any(value<i32.min) or np.any(value>i32.max):
                    raise ValueError("Updated fixed-point epilogue can overflow INT32")
        return {self.weight_name:weights,self.bias_name:bias.astype(np.int32).reshape(constants[self.bias_name].shape)}


class DetectorTail(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = check_contract(source)
        self.constants = {value.name:numpy_helper.to_array(value) for value in source.graph.initializer}
        clusters = convolution_clusters(source)
        self.layers = nn.ModuleList(IntegerConv(source,clusters[f"conv{i}"],self.constants) for i in range(11,16))
        self.grid = {annotation.tensor_name:{field.key:self.constants[field.value].item()
                        for field in annotation.quant_parameter_tensor_names}
                     for annotation in source.graph.quantization_annotation}
        self.register_buffer("sigmoid",torch.from_numpy(self.constants["sigmoid_lut"].copy()))
        for layer in self.layers:
            layer.export(self.constants)

    @staticmethod
    def pool(data):
        mean = data.mean(dim=(-2,-1))
        return rounded(mean,((data.sum(dim=(-2,-1))+10)/20).floor())

    def forward(self, feature):
        spatial = self.layers[0](feature)
        split = self.layers[1](spatial).reshape(-1,128,4,5)
        centers,sizes,confidence = [self.pool(head(split)) for head in self.layers[2:]]
        sizes = rounded(sizes,sizes.clamp(min=self.grid["sizes_pooled"]["ZERO_POINT_TENSOR"]))
        return (centers.reshape(-1,4,2,2), sizes.reshape(-1,4,2),
                confidence.reshape(-1,4,2), spatial)

    def decode(self, values):
        center,size,confidence,_ = values
        fields = [("centers_pooled",center),("sizes_pooled",size),("scores_pooled",confidence)]
        return tuple((data-self.grid[name]["ZERO_POINT_TENSOR"])*self.grid[name]["SCALE_TENSOR"] for name,data in fields)

    def bytes(self, values):
        center,size,confidence,feature = values
        return tuple(value.detach().cpu().numpy().astype(np.uint8) for value in (
            center.reshape(-1,8,2), size.reshape(-1,8), self.sigmoid[confidence.long()].reshape(-1,8),feature))

    def export(self, template=None):
        updates = {}
        for layer in self.layers:
            updates.update(layer.export(self.constants))
        model = copy.deepcopy(self.source if template is None else template)
        replaced = set()
        for init in model.graph.initializer:
            if init.name in updates:
                init.CopyFrom(numpy_helper.from_array(updates[init.name],init.name))
                replaced.add(init.name)
        if replaced!=set(updates):
            raise ValueError("Export template does not contain the trained tail")
        if [output.name for output in model.graph.output]==list(DECODED_PORTS):
            check_interface(model, decoded=True)
        else:
            check_contract(model)
        onnx.checker.check_model(model,full_check=True)
        return model


def session(model, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(),options,providers=["CPUExecutionProvider"])


def prefix_session(source, threads=1):
    # ONNX's extractor rejects quantization annotations. They are metadata
    # for consumers of the full graph and this private prefix returns raw codes.
    prefix = onnx.ModelProto()
    prefix.CopyFrom(source)
    del prefix.graph.quantization_annotation[:]
    return session(onnx.utils.Extractor(prefix).extract_model(["images"],[FEATURE]),threads)


def verify_export(tail, artifact, prefix, images):
    exact = session(artifact)
    for image in images:
        feed = {"images":image[None]}
        feature = prefix.run([FEATURE],feed)[0]
        with torch.no_grad():
            replay = tail.bytes(tail(torch.from_numpy(feature.astype(np.float32))))
        for expected,actual in zip(exact.run(list(PORTS),feed),replay,strict=True):
            np.testing.assert_array_equal(expected,actual)
