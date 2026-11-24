"""Trainable Handspotter network and folded checkpoint conversion."""
import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from handspotter_model import GRIDS, INPUT_GRID, LAYERS, check_contract, decoded_graph, integer_graph


def quantize(value, scale, zero, low, high):
    return (value/scale+zero).round().clamp(low,high)


def ste(value, discrete):
    return discrete.detach()+(value-value.detach())


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel, stride, activation):
        super().__init__()
        self.conv = nn.Conv2d(incoming,outgoing,kernel,stride,padding=kernel//2,bias=not activation)
        self.norm = nn.BatchNorm2d(outgoing) if activation else nn.Identity()

    def weights(self):
        weights = self.conv.weight.float()
        scales = weights.detach().abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-7)/127
        return quantize(weights,scales,0,-127,127),scales

    def forward(self, value, output_scale, zero, activated, qat):
        if qat:
            codes,scale = self.weights()
            weight = ste(self.conv.weight,codes*scale)
            value = F.conv2d(value,weight,self.conv.bias,self.conv.stride,self.conv.padding)
            codes = quantize(value.float(),output_scale,zero,0,255)
            bounded = value.float().clamp(-zero*output_scale,(255-zero)*output_scale)
            return ste(bounded,(codes-zero)*output_scale)
        value = self.conv(value)
        if activated and value.is_cuda and torch.version.hip:
            # Use ATen's normalization kernel on ROCm. This keeps convolution
            # on MIOpen and avoids its runtime dependency on C++ headers here.
            norm = self.norm
            if norm.training:
                norm.num_batches_tracked.add_(1)
            value,_,_ = torch.ops.aten.native_batch_norm(
                value,norm.weight,norm.bias,norm.running_mean,norm.running_var,
                norm.training,norm.momentum,norm.eps,
            )
        else:
            value = self.norm(value)
        return value.clamp(0,output_scale*255) if activated else value


class Handspotter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales, self.zeros = zip(*GRIDS)
        self.layers = nn.ModuleList(
            ConvBlock(incoming, outgoing, kernel, stride, index < 13)
            for index, (incoming, outgoing, kernel, stride) in enumerate(LAYERS)
        )
        with torch.no_grad():
            for layer,bias in zip(self.layers[13:],(.45,.14,0.),strict=True):
                layer.conv.bias.fill_(bias)
        self.qat = False

    def fold_normalization(self):
        self.eval()
        for layer in self.layers[:13]:
            layer.conv = fuse_conv_bn_eval(layer.conv,layer.norm)
            layer.norm = nn.Identity()
        self.qat = True

    def block(self, index, value):
        return self.layers[index](value,self.scales[index],self.zeros[index],index<13,self.qat)

    def forward(self, images):
        count = images.shape[0]
        value = (images.reshape(-1,1,128,160).float()/255).contiguous(memory_format=torch.channels_last)
        for index in range(7):
            value = self.block(index,value)
        value = value.reshape(count,1024,16,20).contiguous(memory_format=torch.channels_last)
        for index in range(7,13):
            value = self.block(index,value)
        value = value.reshape(count*4,128,4,5).contiguous(memory_format=torch.channels_last)
        pooled = []
        for index in range(13,16):
            plane = self.block(index,value).float()
            mean = plane.mean(dim=(-2,-1))
            if self.qat:
                scale,zero = self.scales[index],self.zeros[index]
                spatial_codes = quantize(plane,scale,zero,0,255)
                code = ((spatial_codes.sum(dim=(-2,-1))+10)/20).floor()
                mean = ste(mean,(code-zero)*scale)
            pooled.append(mean)
        return (pooled[0].reshape(count,4,2,2),pooled[1].reshape(count,4,2),
                pooled[2].reshape(count,4,2))


def exports(model):
    parameters = []
    for index, layer in enumerate(model.layers):
        codes,weight_scale = layer.weights()
        weights = codes.detach().cpu().numpy().astype(np.int8)
        weight_scale = weight_scale.detach().cpu().numpy().reshape(-1).astype(np.float64)
        input_scale = INPUT_GRID[0] if index==0 else model.scales[12 if index>=13 else index-1]
        scale = model.scales[index]
        desired = input_scale*weight_scale/scale
        divisor = np.exp2(np.clip(np.floor(np.log2(2048/(16384*desired))),0,15)).astype(np.int64)
        multiplier = np.maximum(1,np.rint(desired*16384*divisor)).astype(np.int64)
        real = multiplier/(16384*divisor)
        bias_float = layer.conv.bias.detach().cpu().numpy()
        bias = np.rint((bias_float/scale+model.zeros[index])/real+(divisor-1)/2).astype(np.int64)
        low = np.minimum(weights,0).sum(axis=(1,2,3),dtype=np.int64)*255+bias
        high = np.maximum(weights,0).sum(axis=(1,2,3),dtype=np.int64)*255+bias
        limit = np.iinfo(np.int32)
        first_low = (low // divisor) * divisor
        rounded_low = (low // divisor) * multiplier + 8192
        rounded_high = (high // divisor) * multiplier + 8192
        for value in (bias, first_low, high, rounded_low - 8192,
                      (rounded_low // 16384) * 16384, rounded_high):
            if np.any(value < limit.min) or np.any(value > limit.max):
                raise ValueError(f"INT32 overflow in conv{index}")
        parameters.append((weights, bias, divisor, multiplier))
    raw = check_contract(integer_graph(parameters))
    decoded = decoded_graph(raw)
    for artifact in (raw, decoded):
        onnx.checker.check_model(artifact, full_check=True)
    return raw, decoded
