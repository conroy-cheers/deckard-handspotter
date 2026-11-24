"""Locate Handspotter's integer convolutions in an exported ONNX graph.

Each exported convolution is a ``ConvInteger`` followed by a fixed-point
epilogue: per-channel bias, floor division, multiplier and rounding offset,
a second floor division by 16384, and a byte clamp. The nodes between the
uint8 input and uint8 output form one convolution cluster.
"""
from collections import defaultdict
from functools import lru_cache

import numpy as np
import onnx
from onnx import TensorProto as T, helper

INTEGER_TYPES = {T.UINT8, T.INT8, T.UINT16, T.INT16, T.UINT32, T.INT32, T.UINT64, T.INT64}
ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Mod", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot",
               "BitShift", "Clip", "Min", "Max", "Cast", "Neg", "Abs", "Sign"}
I32 = (-(1 << 31), (1 << 31) - 1)
EPILOGUE = ["ConvInteger", "Add", "Mod", "Sub", "Div", "Mul", "Add", "Mod", "Sub", "Div", "Clip", "Cast"]


def convolution_clusters(model):
    """Return each ConvInteger and its wide-integer epilogue, keyed by uint8 output."""
    # Inference supplies intermediate types without reordering nodes.
    g = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    constants = {t.name for t in g.initializer}
    types = {v.name: v.type.tensor_type.elem_type for v in [*g.input, *g.value_info, *g.output]}
    for t in g.initializer:
        types[t.name] = t.data_type

    def member(node):
        if node.domain not in ("", "ai.onnx") or not all(types.get(k) in INTEGER_TYPES for k in node.output if k):
            return False
        if node.op_type == "Gather":
            return node.input[0] in constants
        return node.op_type == "ConvInteger" or node.op_type in ELEMENTWISE

    producer = {k: i for i, n in enumerate(g.node) for k in n.output if k}
    consumers = defaultdict(list)
    for i, n in enumerate(g.node):
        for k in n.input:
            if k:
                consumers[k].append(i)
    parent = list(range(len(g.node)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Join integer nodes along every wide integer edge. Bytes bound clusters.
    candidates = {i for i, n in enumerate(g.node) if member(n)}
    for value, i in producer.items():
        if i not in candidates or types.get(value) not in INTEGER_TYPES or types[value] == T.UINT8:
            continue
        for j in consumers[value]:
            if j in candidates:
                parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i in sorted(candidates):
        groups[find(i)].append(i)

    outputs = {v.name for v in g.output}
    clusters = {}
    for members in groups.values():
        ids = set(members)
        produced = {k for i in members for k in g.node[i].output if k}
        inputs = sorted({k for i in members for k in g.node[i].input
                         if k and k not in produced and k not in constants})
        exported = sorted(k for k in produced if k in outputs or set(consumers[k]) - ids)
        sources = [i for i in members if g.node[i].op_type == "ConvInteger"]
        if len(exported) != 1 or types[exported[0]] != T.UINT8 or len(sources) != 1 or len(inputs) != 1:
            continue
        conv = g.node[sources[0]]
        if (types[inputs[0]] != T.UINT8 or inputs[0] != conv.input[0] or len(conv.input) < 2
                or any(k and k not in constants for k in conv.input[1:])):
            continue
        clusters[exported[0]] = dict(members=sorted(members), input=inputs[0], output=exported[0])
    return clusters


def fixed_point_epilogue(model, cluster, constants):
    """Read the per-channel bias, divisor, multiplier and rounding of one cluster."""
    nodes = model.graph.node
    chain = [nodes[i] for i in cluster["members"]]
    if [n.op_type for n in chain] != EPILOGUE:
        return None
    conv, bias, mod, sub, div, mul, add, mod2, sub2, div2, clip, cast = chain
    if not (bias.input[0] == conv.output[0] and mod.input[0] == bias.output[0] and
            list(sub.input) == [bias.output[0], mod.output[0]] and list(div.input) == [sub.output[0], mod.input[1]] and
            mul.input[0] == div.output[0] and add.input[0] == mul.output[0] and mod2.input[0] == add.output[0] and
            list(sub2.input) == [add.output[0], mod2.output[0]] and list(div2.input) == [sub2.output[0], mod2.input[1]] and
            clip.input[0] == div2.output[0] and cast.input[0] == clip.output[0]):
        return None
    if any(helper.get_attribute_value(a) != 0 for n in (mod, mod2) for a in n.attribute if a.name == "fmod"):
        return None
    try:
        channels = constants[conv.input[1]].shape[0]
        channel = lambda name: np.broadcast_to(constants[name], (1, channels, 1, 1)).reshape(channels).astype(np.int64)
        source_bias, divisor, multiplier, rounding = [channel(n.input[1]) for n in (bias, div, mul, add)]
        if not (np.all(constants[div2.input[1]] == 16384) and np.all(constants[clip.input[1]] == 0)
                and np.all(constants[clip.input[2]] == 255)):
            return None
    except (KeyError, ValueError):
        return None
    if np.any(divisor < 1) or np.any(divisor > 65536) or np.any(divisor & (divisor - 1)) or np.any(multiplier <= 0):
        return None
    return dict(divisor=divisor, source_bias=source_bias, multiplier=multiplier, rounding=rounding)


def accumulator_ranges(w, x_zero, group, channels):
    """Exact per-output-channel ConvInteger accumulator bounds for x in [0, 255]."""
    w = w.astype(np.int64)
    lo_x, hi_x = -int(x_zero), 255 - int(x_zero)
    flat = w.reshape(w.shape[0], -1)
    lo = np.minimum(flat * lo_x, flat * hi_x).sum(1)
    hi = np.maximum(flat * lo_x, flat * hi_x).sum(1)
    return np.minimum(lo, 0), np.maximum(hi, 0)  # Zero padding contributes 0.


def epilogue_ranges(model, cluster, constants):
    """Exact bounds of each int32 intermediate in one convolution, or None if any overflows."""
    epilogue = fixed_point_epilogue(model, cluster, constants)
    if epilogue is None:
        return None
    conv = model.graph.node[cluster["members"][0]]
    weights = constants[conv.input[1]]
    if weights.dtype != np.int8:
        return None
    x_zero = int(constants[conv.input[2]]) if len(conv.input) > 2 and conv.input[2] else 0
    low, high = accumulator_ranges(weights, x_zero, 1, len(epilogue["source_bias"]))
    bias, divisor = epilogue["source_bias"], epilogue["divisor"]
    multiplier, rounding = epilogue["multiplier"], epilogue["rounding"]
    stages = {"accumulator": (low, high), "biased": (low + bias, high + bias)}
    stages["divided"] = tuple(np.floor_divide(value, divisor) for value in stages["biased"])
    stages["product"] = tuple(value * multiplier for value in stages["divided"])
    stages["rounded"] = tuple(value + rounding for value in stages["product"])
    if any(np.any(lo < I32[0]) or np.any(hi > I32[1]) for lo, hi in stages.values()):
        return None
    return {name: (int(lo.min()), int(hi.max())) for name, (lo, hi) in stages.items()}


@lru_cache(None)
def _staged_residues(divisor):
    """Exhaust all residues of the periodic, four-code native prefix identity."""
    r = np.arange(divisor, dtype=np.int64)
    numerator = 8 * r - divisor + 4
    denominator = 2 * divisor
    whole, rem = numerator // denominator, numerator % denominator
    even = whole + ((2 * rem > denominator) | ((2 * rem == denominator) & (whole % 2 != 0)))
    up = (2 * numerator + denominator) // (2 * denominator)
    return bool(np.all(even // 4 == 0) and np.all(up // 4 == 0))


def dsp_fast_path(model, cluster, constants):
    """Whether a convolution qualifies for the Hexagon lowering's staged U16 path.

    That path runs the convolution natively with four codes per pre-shift
    integer. Layers that do not qualify remain correct but are lowered to
    slower exact integer arithmetic.
    """
    epilogue = fixed_point_epilogue(model, cluster, constants)
    if epilogue is None:
        return False
    d, b, m, r = (epilogue[k] for k in ("divisor", "source_bias", "multiplier", "rounding"))
    if not all(_staged_residues(int(v)) for v in np.unique(d)):
        return False
    low = (16384 - r + m - 1) // m - 1
    high = low + 16383
    if np.any((low * m + r) // 16384 > 0) or np.any((high * m + r) // 16384 < 255):
        return False
    if np.any(low * m + r < I32[0]) or np.any(high * m + r > I32[1]):
        return False
    native_bias = b - low * d
    native_bias_codes = native_bias * 256 + 128 - d * 32
    conv = model.graph.node[cluster["members"][0]]
    w = constants[conv.input[1]]
    if w.dtype != np.int8:
        return False
    xz = int(constants[conv.input[2]]) if len(conv.input) > 2 and conv.input[2] else 0
    lo, hi = accumulator_ranges(w, xz, 1, len(b))
    checks = [native_bias, native_bias_codes, lo + b, hi + b, lo + native_bias, hi + native_bias,
              lo * 256 + native_bias_codes, hi * 256 + native_bias_codes]
    for product in ((lo + b) // d * m, (hi + b) // d * m):
        checks += [product, product + r]
    return not any(np.any(value < I32[0]) or np.any(value > I32[1]) for value in checks)
