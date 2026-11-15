# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-preserving module swaps for backends that lack an op.

The model in model.py uses plain PyTorch ops. Each swap below computes the
same function from more widely supported ops; export_phonon.py selects them
per backend:

  decompose_elu           MLX and Vulkan have no aten.elu kernel; without it
                          the Mimi decoder splits at every ELU.
  conv_transpose1d_as_2d  Vulkan runs transposed convolution on 4-D inputs only.
  depthwise_conv_transpose_as_mul
                          Vulkan computes the Mimi upsampler (depthwise,
                          kernel = 2 * stride) incorrectly; this runs it as a
                          multiply and an overlap-add.
  use_webgpu_compat_ops   WebGPU runtimes built before these runtime fixes
                          (off by default; --webgpu-compat-ops):
                            * native_layer_norm resize hook (the flow LM's
                              dynamic sequence length),
                            * reduce ops reading a Bool keepdim argument,
                            * aten.elu and aten.silu kernels,
                            * 1-D transposed convolution.
  compute_weights_in      --dtype bf16 (CUDA and MLX; FP32 is the default):
                          linear, convolution and embedding weights and their
                          matmuls in bf16; norms, attention, RoPE and the
                          streaming state stay in float32. Not
                          function-preserving.

Each function swaps modules in place in the programs from build_programs() and
returns how many it replaced.
"""

import torch
import torch.nn.functional as F

from executorch.examples.models.phonon.model import FlowLMStep, StreamingConvTranspose1d
from torch import nn


class LayerNormFromRMSNorm(nn.Module):
    """LayerNorm whose mean is a linear with weight 1/D (exact for D a power
    of two) followed by the RMS-norm pattern, which follows a resized sequence
    length without a native_layer_norm or mean.dim resize hook."""

    def __init__(self, norm: nn.LayerNorm):
        super().__init__()
        (dim,) = norm.normalized_shape
        self.weight, self.bias, self.eps = norm.weight, norm.bias, norm.eps
        self.register_buffer("mean_weight", torch.full((1, dim), 1.0 / dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - F.linear(x, self.mean_weight)
        rms = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + self.eps)
        return xc * rms * self.weight + self.bias


class ELUFromExp(nn.Module):
    """ELU(alpha=1) = relu(x) + exp(min(x, 0)) - 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + torch.exp(torch.clamp(x, max=0.0)) - 1.0


class SiLUFromSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvTranspose1dAs2d(nn.Module):
    """ConvTranspose1d as a conv_transpose2d over a height-1 image. The 4-D
    weight is stored as a parameter: Vulkan prepacks only constant weights."""

    def __init__(self, convtr: nn.ConvTranspose1d):
        super().__init__()
        self.weight = nn.Parameter(
            convtr.weight.detach().unsqueeze(2), requires_grad=False
        )
        self.bias = convtr.bias
        self.stride, self.groups = convtr.stride[0], convtr.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(
            x.unsqueeze(2),
            self.weight,
            self.bias,
            stride=(1, self.stride),
            groups=self.groups,
        )
        return y.squeeze(2)


class ComputeIn(nn.Module):
    """Runs `module` with its weights in `dtype` on a float32 input and
    returns float32. Other attributes (bias, kernel_size, ...) are read from
    the wrapped module."""

    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.inner = module.to(dtype)
        self.dtype = dtype

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_floating_point():
            return self.inner(x.to(self.dtype)).to(x.dtype)
        return self.inner(x).float()


class DepthwiseConvTranspose1dAsMul(nn.Module):
    """Depthwise ConvTranspose1d with kernel_size == 2 * stride, without bias:
    each input step scales the kernel, and the halves of neighboring steps
    overlap."""

    def __init__(self, convtr: nn.ConvTranspose1d):
        super().__init__()
        (self.stride,) = convtr.stride
        channels, _, kernel = convtr.weight.shape
        self.weight = nn.Parameter(
            convtr.weight.detach().reshape(1, channels, 1, kernel), requires_grad=False
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        s = self.stride
        y = x.unsqueeze(3) * self.weight  # [b, c, n, 2s]
        head = y[..., :s].reshape(b, c, n * s)
        tail = y[..., s:].reshape(b, c, n * s)
        zeros = torch.zeros(b, c, s, dtype=x.dtype)
        return torch.cat([head, zeros], 2) + torch.cat([zeros, tail], 2)


def _swap(root: nn.Module, cls: type, make) -> int:
    count = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, cls):
                setattr(parent, name, make(child))
                count += 1
    return count


def decompose_elu(programs: dict[str, nn.Module]) -> int:
    return sum(_swap(p, nn.ELU, lambda m: ELUFromExp()) for p in programs.values())


def decompose_silu(programs: dict[str, nn.Module]) -> int:
    return sum(
        _swap(p, nn.SiLU, lambda m: SiLUFromSigmoid()) for p in programs.values()
    )


def conv_transpose1d_as_2d(programs: dict[str, nn.Module]) -> int:
    count = 0
    for program in programs.values():
        for module in program.modules():
            if isinstance(module, StreamingConvTranspose1d) and isinstance(
                module.convtr, nn.ConvTranspose1d
            ):
                module.convtr = ConvTranspose1dAs2d(module.convtr)
                count += 1
    return count


def depthwise_conv_transpose_as_mul(programs: dict[str, nn.Module]) -> int:
    count = 0
    for program in programs.values():
        for module in program.modules():
            convtr = (
                getattr(module, "convtr", None)
                if isinstance(module, StreamingConvTranspose1d)
                else None
            )
            if (
                isinstance(convtr, nn.ConvTranspose1d)
                and convtr.bias is None
                and convtr.groups == convtr.in_channels == convtr.out_channels
                and convtr.kernel_size[0] == 2 * convtr.stride[0]
            ):
                module.convtr = DepthwiseConvTranspose1dAsMul(convtr)
                count += 1
    return count


def decompose_dynamic_layer_norm(programs: dict[str, nn.Module]) -> int:
    """Only the flow LM norms that see the dynamic sequence length."""
    count = 0
    for program in programs.values():
        if isinstance(program, FlowLMStep):
            for layer in program.layers:
                layer.norm1 = LayerNormFromRMSNorm(layer.norm1)
                layer.norm2 = LayerNormFromRMSNorm(layer.norm2)
                count += 2
    return count


WEIGHT_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.ConvTranspose1d,
    nn.Embedding,
    ConvTranspose1dAs2d,
)


def compute_weights_in(programs: dict[str, nn.Module], dtype: torch.dtype) -> int:
    return sum(
        _swap(p, WEIGHT_MODULES, lambda m: ComputeIn(m, dtype))
        for p in programs.values()
    )


def use_webgpu_compat_ops(programs: dict[str, nn.Module]) -> int:
    return sum(
        swap(programs)
        for swap in (
            decompose_dynamic_layer_norm,
            decompose_elu,
            decompose_silu,
            conv_transpose1d_as_2d,
        )
    )
