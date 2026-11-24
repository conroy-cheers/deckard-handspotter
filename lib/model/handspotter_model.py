"""Handspotter architecture and deployable integer ONNX graph."""
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from handspotter_integer import convolution_clusters, dsp_fast_path, epilogue_ranges

# Input channels, output channels, square kernel, stride.
LAYERS = (
    (1, 32, 3, 1), (32, 64, 3, 2), (64, 64, 3, 1),
    (64, 128, 3, 2), (128, 128, 3, 1), (128, 256, 3, 2),
    (256, 256, 3, 1), (1024, 300, 3, 2), (300, 300, 3, 1),
    (300, 300, 3, 2), (300, 300, 3, 1), (300, 150, 3, 1),
    (150, 512, 1, 1), (128, 4, 1, 1), (128, 2, 1, 1), (128, 2, 1, 1),
)
INPUT = (1, 4, 128, 160)
PORTS = {
    "centers_u8": (1, 8, 2),
    "sizes_u8": (1, 8),
    "scores_u8": (1, 8),
    "features_u8": (1, 150, 4, 5),
}
DECODED_PORTS = {name.removesuffix("_u8"): shape for name, shape in PORTS.items()}
INPUT_GRID = (1 / 255, 0)
# Bound most hidden activations at 6. The conv11 grid (exported as features_u8)
# and the head grids are constants of the exported uint8 interface. The
# confidence head pools logit codes.
GRIDS = [(6 / 255, 0)] * 11 + [
    (0.02994581125676632, 0),
    (6 / 255, 0),
    (0.032785166054964066, 175),
    (0.004779368173331022, 126),
    (0.09969951212406158, 123),
]
SCORE_GRID = (1 / 255, 0)
IR_VERSION = 10
OPSET = 21
# Every operator the integer export emits. The Frame's QNN lowering accepts these.
INTEGER_OPS = {"Add", "Cast", "Clip", "ConvInteger", "Div", "Gather", "Identity", "Max",
               "Mod", "Mul", "ReduceSum", "Reshape", "Sub"}


def integer_graph(parameters):
    """Build the integer detector from per-layer (weight, bias, divisor, multiplier)."""
    nodes, constants, annotations = [], [], []

    def const(name, value, dtype):
        constants.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, out, *inputs, **attrs):
        nodes.append(helper.make_node(kind, inputs, [out], **attrs))
        return out

    def grid(name, encoding):
        scale, zero = encoding
        annotations.append((name, {
            "SCALE_TENSOR": const(f"{name}_scale", scale, np.float32),
            "ZERO_POINT_TENSOR": const(f"{name}_zero", zero, np.uint8),
        }))
        return name

    def reshape(name, source, shape, encoding):
        op("Reshape", name, source, const(f"{name}_shape", shape, np.int64))
        return grid(name, encoding)

    def floor_divide(name, source, divisor):
        remainder = op("Mod", f"{name}_remainder", source, divisor)
        multiple = op("Sub", f"{name}_multiple", source, remainder)
        return op("Div", name, multiple, divisor)

    i32 = np.int32
    input_zero = const("conv_input_zero", 0, np.uint8)
    weight_zero = const("conv_weight_zero", 0, np.int8)
    minimum = const("code_min", 0, i32)
    maximum = const("code_max", 255, i32)
    rounding = const("requant_rounding", 8192, i32)
    denominator = const("requant_denominator", 16384, i32)
    axes = const("spatial_axes", [2, 3], np.int64)
    pool_bias = const("pool_rounding", 10, i32)
    pool_count = const("pool_count", 20, i32)
    grid("images", INPUT_GRID)
    source = reshape("views", "images", [4, 1, 128, 160], INPUT_GRID)
    for index, (layer, values) in enumerate(zip(LAYERS, parameters, strict=True)):
        incoming, outgoing, kernel, stride = layer
        name = f"conv{index}"
        weight, bias, divisor, multiplier = values
        weight = const(f"{name}_weight", weight.reshape(outgoing, incoming, kernel, kernel), np.int8)
        bias, divisor, multiplier = [
            const(f"{name}_{key}", array.reshape(1, outgoing, 1, 1), i32)
            for key, array in (("bias", bias), ("divisor", divisor), ("multiplier", multiplier))
        ]
        if index == 7:
            source = reshape("fused_views", source, [1, 1024, 16, 20], GRIDS[6])
        if index == 13:
            head_input = reshape("head_views", source, [4, 128, 4, 5], GRIDS[12])
        tensor = source if index < 13 else head_input
        acc = op("ConvInteger", f"{name}_acc", tensor, weight, input_zero, weight_zero,
                 pads=[kernel // 2] * 4, strides=[stride] * 2, group=1)
        biased = op("Add", f"{name}_biased", acc, bias)
        divided = floor_divide(f"{name}_divided", biased, divisor)
        product = op("Mul", f"{name}_product", divided, multiplier)
        rounded = op("Add", f"{name}_rounded", product, rounding)
        shifted = floor_divide(f"{name}_shifted", rounded, denominator)
        clipped = op("Clip", f"{name}_clipped", shifted, minimum, maximum)
        source = op("Cast", name, clipped, to=TensorProto.UINT8)
        encoding = GRIDS[index]
        grid(name, encoding)
        if index == 11:
            grid(op("Identity", "features_u8", source), encoding)
        if index < 13:
            continue
        port = tuple(PORTS)[index - 13]
        head = port.removesuffix("_u8")
        spatial = op("Cast", f"{head}_spatial", source, to=TensorProto.INT32)
        total = op("ReduceSum", f"{head}_sum", spatial, axes, keepdims=1)
        total = op("Add", f"{head}_rounded", total, pool_bias)
        mean = op("Div", f"{head}_mean", total, pool_count)
        pooled = grid(op("Cast", f"{head}_pooled", mean, to=TensorProto.UINT8), encoding)
        if head == "sizes":
            zero = const("size_floor", encoding[1], np.uint8)
            pooled = grid(op("Max", "sizes_relu", pooled, zero), encoding)
        if head == "scores":
            scale, zero = encoding
            logits = (np.arange(256) - zero) * scale
            lut = const("sigmoid_lut", np.rint(255 / (1 + np.exp(-logits))), np.uint8)
            indices = op("Cast", "scores_indices", pooled, to=TensorProto.INT64)
            encoding = SCORE_GRID
            pooled = grid(op("Gather", "scores_lut", lut, indices), encoding)
        reshape(port, pooled, PORTS[port], encoding)
    graph = helper.make_graph(
        nodes, "handspotter_integer", [helper.make_tensor_value_info("images", TensorProto.UINT8, INPUT)],
        [helper.make_tensor_value_info(name, TensorProto.UINT8, shape) for name, shape in PORTS.items()],
        constants,
    )
    for name, grid_parameters in annotations:
        annotation = graph.quantization_annotation.add(tensor_name=name)
        for key, value in grid_parameters.items():
            annotation.quant_parameter_tensor_names.add(key=key, value=value)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION)


def check_interface(model, decoded=False):
    """Require exactly the image input and the exported outputs, in order."""
    def signature(values):
        return [(value.name, value.type.tensor_type.elem_type,
                 tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)) for value in values]
    ports, dtype = (DECODED_PORTS, TensorProto.FLOAT) if decoded else (PORTS, TensorProto.UINT8)
    expected_inputs = [("images", TensorProto.UINT8, INPUT)]
    expected_outputs = [(name, dtype, shape) for name, shape in ports.items()]
    graph = model.graph
    if signature(graph.input) != expected_inputs or signature(graph.output) != expected_outputs:
        kind = "decoded" if decoded else "integer"
        raise ValueError(f"Unexpected {kind} Handspotter model interface: inputs {signature(graph.input)}, "
                         f"outputs {signature(graph.output)}")
    return model


def check_contract(model):
    """Require the integer graph properties the Frame's Hexagon lowering relies on."""
    check_interface(model)
    graph = model.graph
    problems = []
    if model.ir_version != IR_VERSION or [(o.domain, o.version) for o in model.opset_import] != [("", OPSET)]:
        problems.append(f"IR {model.ir_version} with opsets {[(o.domain, o.version) for o in model.opset_import]}")
    unexpected = sorted({f"{n.domain}::{n.op_type}" if n.domain else n.op_type for n in graph.node} - INTEGER_OPS)
    if unexpected:
        problems.append(f"operators {unexpected}")
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    annotations = {item.tensor_name: {field.key: field.value for field in item.quant_parameter_tensor_names}
                   for item in graph.quantization_annotation}
    for value in [*graph.input, *graph.output]:
        fields = annotations.get(value.name, {})
        scale = constants.get(fields.get("SCALE_TENSOR"))
        zero = constants.get(fields.get("ZERO_POINT_TENSOR"))
        if (scale is None or zero is None or scale.shape or zero.shape or scale.dtype != np.float32
                or zero.dtype != np.uint8 or not np.isfinite(scale) or scale <= 0):
            problems.append(f"quantization grid of {value.name}")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    for value in inferred.graph.value_info:
        dims = value.type.tensor_type.shape.dim
        if not value.type.tensor_type.HasField("shape") or any(dim.dim_value <= 0 for dim in dims):
            problems.append(f"dynamic shape of {value.name}")
    clusters = convolution_clusters(model)
    convolutions = [node.output[0] for node in graph.node if node.op_type == "ConvInteger"]
    if len(clusters) != len(convolutions):
        problems.append("ConvInteger outside a uint8-bounded fixed-point cluster")
    for name, cluster in clusters.items():
        if epilogue_ranges(model, cluster, constants) is None:
            problems.append(f"{name} is not a fixed-point convolution within INT32")
    if problems:
        raise ValueError("Handspotter integer contract violated: " + "; ".join(problems))
    return model


def dsp_fast_path_layers(model):
    """Map each convolution to whether it qualifies for the Hexagon staged U16 path."""
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    clusters = convolution_clusters(model)
    order = sorted(clusters, key=lambda name: clusters[name]["members"][0])
    return {name: dsp_fast_path(model, clusters[name], constants) for name in order}


def decoded_graph(raw):
    check_interface(raw)
    model = copy.deepcopy(raw)
    graph = model.graph
    grids = {
        item.tensor_name: {field.key: field.value for field in item.quant_parameter_tensor_names}
        for item in graph.quantization_annotation
    }
    del graph.output[:]
    for name, shape in PORTS.items():
        target = name.removesuffix("_u8")
        params = grids[name]
        graph.node.append(helper.make_node(
            "DequantizeLinear", [name, params["SCALE_TENSOR"], params["ZERO_POINT_TENSOR"]], [target],
        ))
        graph.output.append(helper.make_tensor_value_info(target, TensorProto.FLOAT, shape))
    return check_interface(model, decoded=True)
