# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import reduce
from math import gcd
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.quantized_ops import *  # noqa

from torch.ao.quantization.fx._decomposed import (
    _quant_min_max_bounds_check,
    quantized_decomposed_lib,
)
from torch.library import impl


try:
    # pyre-ignore[21]: Undefined import.
    from fairseq2.nn.embedding import (
        Embedding as fsEmbedding,
        StandardEmbedding as fsStandardEmbedding,
    )

    # pyre-ignore[21]: Undefined import.
    from fairseq2.nn.projection import Linear as fsLinear
except:
    print("Could not import fairseq2 modules.")
    fsEmbedding = nn.Embedding
    fsStandardEmbedding = nn.Embedding
    fsLinear = nn.Linear


def dynamically_quantize_per_channel(
    x,
    quant_min,
    quant_max,
    target_dtype,
    group_size: Optional[int] = None,
    *,
    scales_dtype=torch.float16,
    enable_non_multiple_groups=True,
):
    """
    Dynamically quantize per channel.  This function is used for quantizing weights,
    for linear and embedding layers.

    Arguments:
        x: input tensor,
        quant_min: minimum value after quantization,
        quant_max: maximum value after quantization,
        target_dtype: target data type for weights after quantization,
        group_size: number of elements of the channel to quantize together

    Keyword arguments:
        scales_dtype: data type of scale,
        enable_non_multiple_groups: if True, allow the rowsize to not be a multiple of group size,
                        with a final group of a size less than group size.

    Assumptions:
        This function assumes symmetric quantization, axis ==0 and a dense memory format.
    """

    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    x_shape_1 = x.shape[1]

    if group_size is None or group_size == 0:
        items = x_shape_1
    elif not enable_non_multiple_groups:
        assert group_size > 0, "group size must be positive"
        assert (
            x_shape_1 % group_size
        ) == 0, f"weights dimension 1 = {x_shape_1} must be a multiple of group size {group_size}"
        items = group_size
    else:
        assert group_size > 0, "group size must be positive"
        print(
            f"row-size of weight matrix {x_shape_1} is not divisible by group size {group_size}, using nearest neighbor rounding"
        )
        assert (
            x_shape_1 % group_size != 0
        ), f"expected x.shape[1] to not be a multiple of group size {group_size}, but got {x_shape_1}"
        padding = group_size - (x_shape_1 % group_size)
        x = F.pad(x, (0, padding))
        items = group_size

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    x = x.view(x.shape[0], x.shape[1] // items, items)
    # get min and max
    min_val, max_val = torch.aminmax(x, dim=2)
    # print(f"min_val {min_val}")
    # print(f"max_val {max_val}")

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = (
        torch.clamp(x_zp, quant_min, quant_max).to(target_dtype).view(x.shape[0], -1)
    )

    scales = scales.to(dtype=scales_dtype)
    quant = quant[:, :x_shape_1]

    return quant, scales, zero_points


# TODO: move this to https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(f"unsupported dtype in choose_qparams_per_token: {dtype}")

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


# TODO: move this to https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val, max_val = torch.aminmax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float32), zero_point.to(torch.float32)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    assert (
        num_tokens == scales.numel()
    ), f"num_tokens: {num_tokens} scales: {scales.size()}"
    assert (
        num_tokens == zero_points.numel()
    ), f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        torch.round(input / scales + zero_points).clamp(quant_min, quant_max).to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input.to(output_dtype) * scales
    return input


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return torch.empty_like(input, dtype=output_dtype)


def get_group_qparams_symmetric(w, n_bit=4, groupsize=128, precision=torch.float16):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    scales = max_val_abs / (float(max_int - min_int) / 2)
    scales = torch.max(scales, torch.full_like(scales, torch.finfo(torch.float32).eps))
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 0)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )


def pack_scales_and_zeros(scales, zeros, precision=torch.float16):
    assert scales.shape == zeros.shape
    assert scales.dtype == precision
    assert zeros.dtype == precision
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    # why is this float?
    # assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2

    # TODO: check for dtype, currently we can't express torch.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.div(scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    return torch.empty_like(input, dtype=dtype)


def group_quantize_tensor_symmetric(w, n_bit=4, group_size=128):
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size)
    n_bit = 4
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    # TODO: currently we don't know how to express torch.int4, we'll
    # add torch.int4 to core later
    w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
        w, scales, zeros, min_int, max_int, torch.int8, group_size
    )

    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int8, scales_and_zeros


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    assert group_size > 1
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    assert w_int8.shape[-1] % group_size == 0
    assert w_int8.dim() == 2

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)
    w_dq = (
        w_int8_grouped.sub(zero_points).mul(scales).reshape_as(w_int8).to(output_dtype)
    )
    return w_dq


def group_dequantize_tensor_symmetric(
    w_int8, scales_and_zeros, n_bit=4, group_size=128
):
    # TODO: remove this
    scales, zero_points = unpack_scales_and_zeros(scales_and_zeros)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    return torch.ops.quantized_decomposed.quantize_per_channel_group(
        w_int8, scales, zero_points, quant_min, quant_max, torch.int8, group_size
    )


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


quantized_decomposed_lib.define("pack_int4_from_int8(Tensor int8_data) -> Tensor")


@impl(quantized_decomposed_lib, "pack_int4_from_int8", "CompositeExplicitAutograd")
def pack_int4_from_int8(int8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = int8_data.shape
    assert shape[-1] % 2 == 0
    int8_data = int8_data.contiguous().view(-1)
    return (int8_data[::2] << 4 | int8_data[1::2]).view(down_size(shape))


quantized_decomposed_lib.define("unpack_int4_to_int8(Tensor int8_data) -> Tensor")


@impl(quantized_decomposed_lib, "unpack_int4_to_int8", "CompositeExplicitAutograd")
def unpack_int4_to_int8(int8_data: torch.Tensor) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using int8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    shape = int8_data.shape
    first_elements = (int8_data >> 4).to(torch.int8)
    second_elements = (int8_data & 0b1111).to(torch.int8)
    return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))


class QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(module, node_type):
    for name, child in module.named_children():
        print(f"name: {name}")
        if isinstance(child, nn.Linear):
            if (
                (node_type == "*")
                or (node_type == "output" and name == "output")
                or (node_type == "!output" and name != "output")
            ):
                print(f"{name, child}")
                print(f"in_features: {child.in_features}")
                print(f"out_features: {child.out_features}")
                setattr(
                    module,
                    name,
                    WeightOnlyInt8Linear(child.in_features, child.out_features),
                )
        else:
            replace_linear_weight_only_int8_per_channel(child, node_type)


class WeightOnlyInt8QuantHandler:
    def __init__(
        self,
        mod,
        *,
        node_type: str = "*",
        bitwidth: Optional[int] = None,
        group_size: Optional[int] = None,
    ):
        self.mod = mod
        self.group_size = group_size
        self.node_type = node_type
        if bitwidth is None:
            self.bitwidth = 8
        else:
            self.bitwidth = bitwidth

    @torch.no_grad()
    def create_quantized_state_dict(self) -> Dict:
        cur_state_dict = self.mod.state_dict()

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for fqn, mod in self.mod.named_modules():
            # print(f"maybe? quantize {fqn}...{type(mod)}")
            if isinstance(mod, torch.nn.Linear) or isinstance(mod, fsLinear):
                # print(f"candidate {fqn}, nodetype {self.node_type}")
                if (
                    (self.node_type == "*")
                    or (self.node_type == "output" and fqn in ["output", "final_proj"])
                    or (
                        self.node_type == "!output"
                        and fqn not in ["output", "final_proj"]
                    )
                ):
                    print(
                        f"quantize {self.node_type} {fqn, mod} with groupsize {self.group_size}, bitwidth {self.bitwidth}"
                    )

                    # print(f"initial weight shape {mod.weight.shape}")
                    input_weight = mod.weight.float()

                    # print(f"expanded weight shape {input_weight.shape}")
                    weight, scales, _ = dynamically_quantize_per_channel(
                        input_weight,
                        range_min,
                        range_max,
                        torch.int8,
                        self.group_size,
                        scales_dtype=mod.weight.dtype,
                    )

                    cur_state_dict[f"{fqn}.weight"] = weight
                    # squeeze makes groupsize=rowsize unidimensional
                    cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_linear_weight_only_int8_per_channel(self.mod, self.node_type)
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
        # return F.linear(input, self.weight.to(dtype=input.dtype)) * se...


##### embedding table quantization ######


def replace_embedding_weight_only_grouped_int8_per_channel(
    module, bitwidth: int = 8, group_size: Optional[int] = None
):
    for name, child in module.named_children():
        print(f"name: {name}")
        if isinstance(child, nn.Embedding):
            print(f"{name, child}")
            print(f"weights size: {child.weight.size()}")
            setattr(
                module,
                name,
                QuantizedGroupEmbedding(
                    vocab_size=child.weight.shape[0],
                    embedding_dim=child.weight.shape[1],
                    group_size=group_size,
                ),
            )
        else:
            replace_embedding_weight_only_grouped_int8_per_channel(
                child, bitwidth, group_size
            )


class EmbeddingOnlyInt8QuantHandler:
    def __init__(self, mod, *, bitwidth: int = 8, group_size: Optional[int] = None):
        self.mod = mod
        self.group_size = group_size
        self.bitwidth = bitwidth

    @torch.no_grad()
    def create_quantized_state_dict(self) -> Dict:
        cur_state_dict = self.mod.state_dict()

        if self.bitwidth == 4:
            range_min = -8
            range_max = 7
        elif self.bitwidth == 8:
            range_min = -128
            range_max = 127
        else:
            raise ValueError(f"Unsupported bitwidth {self.bitwidth}")

        for fqn, mod in self.mod.named_modules():
            if (
                isinstance(mod, nn.Embedding)
                or isinstance(mod, fsEmbedding)
                or isinstance(mod, fsStandardEmbedding)
            ):
                print("****")
                print(f"Embedding identified: {fqn, mod}")
                print(f"weights size: {mod.weight.size()}")
                # print(f"quantize {fqn}...")

                print(
                    f"quantize {fqn, mod} with groupsize {self.group_size}, bitwidth {self.bitwidth}"
                )
                weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(),
                    range_min,
                    range_max,
                    torch.int8,
                    self.group_size,
                    scales_dtype=mod.weight.dtype,
                )

                # Update state dict
                cur_state_dict[f"{fqn}.weight"] = weight
                # squeeze makes groupsize=rowsize unidimensional
                cur_state_dict[f"{fqn}.scales"] = scales.squeeze(dim=-1)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_embedding_weight_only_grouped_int8_per_channel(
            self.mod, self.bitwidth, self.group_size
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class QuantizedGroupEmbedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        group_size: Optional[int] = None,
        device=None,
        dtype=torch.half,
    ) -> None:
        super().__init__()
        if group_size is None:
            group_size = embedding_dim
        self.group_size = group_size
        self.dtype = dtype
        self.register_buffer(
            "weight", torch.empty((vocab_size, embedding_dim), dtype=torch.int8)
        )
        groups_per_row = (embedding_dim + group_size - 1) // group_size
        if groups_per_row > 1:
            self.register_buffer(
                "scales", torch.ones((vocab_size, groups_per_row), dtype=torch.float16)
            )
        else:
            self.register_buffer(
                "scales", torch.ones((vocab_size,), dtype=torch.float16)
            )

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.llama_quantized.embedding_byte.dtype(
            self.weight, self.scales, None, 0, 0, indices, dtype=self.dtype
        )


#        result_weights = self.weight.index_select(0, indices.view(-1))
#        result_scales = self.scales.index_select(0, indices.view(-1))
#
#        r = result_weights.to(dtype=result_scales.dtype) * result_scales
#        return r.view(indices.size() + (-1,))
##### weight only int4 per channel groupwise quantized code ######


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, group_size, inner_k_tiles):
    weight_int8, scales_and_zeros = group_quantize_tensor_symmetric(
        weight_bf16, n_bit=4, group_size=group_size
    )
    # TODO: better API
    # weight_int4packed = torch.ops.quantized_decomposed.pack_int4_from_int8(weight_int8)
    return weight_int8, scales_and_zeros


def linear_forward_int4(
    x, weight_int8, scales_and_zeros, out_features, group_size, precision
):

    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    # TODO: better API
    # TODO: remove?
    scales_and_zeros = scales_and_zeros.to(torch.float)
    n_bit = 4
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    scales, zeros = unpack_scales_and_zeros(scales_and_zeros)
    # weight_int8 = torch.ops.quantized_decomposed.unpack_int4_to_int8(weight_int4packed)
    w_dq = torch.ops.quantized_decomposed.dequantize_per_channel_group(
        weight_int8,
        scales,
        zeros,
        quant_min,
        quant_max,
        torch.int8,
        group_size,
        precision,
    )

    # x = x.to(torch.float16)
    # w_dq = w_dq.to(torch.float16)
    c = torch.ops.aten.linear(x, w_dq)

    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)


def _check_linear_int4_k(k, group_size=1, inner_k_tiles=1):
    return k % group_size == 0 and k % (inner_k_tiles * 16) == 0


def _calc_padded_size_linear_int4(k, groupsize=1, inner_k_tiles=1):
    return find_multiple(k, groupsize, inner_k_tiles * 16)


def replace_linear_8da4w(
    module,
    group_size,
    inner_k_tiles,
    padding_allowed,
    activation_precision,
    weight_precision,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if (
                _check_linear_int4_k(child.in_features, group_size, inner_k_tiles)
                or padding_allowed
            ):
                setattr(
                    module,
                    name,
                    Int8DynActInt4WeightLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        group_size=group_size,
                        inner_k_tiles=inner_k_tiles,
                        activation_precision=activation_precision,
                        weight_precision=weight_precision,
                    ),
                )
        else:
            replace_linear_8da4w(
                child,
                group_size,
                inner_k_tiles,
                padding_allowed,
                activation_precision,
                weight_precision,
            )


class Int8DynActInt4WeightQuantHandler:
    def __init__(
        self,
        mod,
        group_size=128,
        inner_k_tiles=8,
        padding_allowed=True,
        activation_precision=torch.float16,
        weight_precision=torch.float16,
    ):
        self.mod = mod
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.activation_precision = activation_precision
        self.weight_precision = weight_precision
        assert group_size in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                print("in features:", in_features, " out features:", out_features)
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.group_size, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        print(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = _calc_padded_size_linear_int4(
                            in_features, 1024
                        )
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that group_size and inner_k_tiles*16 evenly divide into it"
                        )

                        continue
                (
                    weight_int4pack,
                    scales_and_zeros,
                ) = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(self.weight_precision),
                    self.group_size,
                    self.inner_k_tiles,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_8da4w(
            self.mod,
            self.group_size,
            self.inner_k_tiles,
            self.padding_allowed,
            self.activation_precision,
            self.weight_precision,
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        group_size: int = 128,
        inner_k_tiles: int = 8,
        activation_precision: torch.dtype = torch.float16,
        weight_precision: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        self.origin_in_features = in_features
        in_features = _calc_padded_size_linear_int4(
            in_features, group_size, inner_k_tiles
        )
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.weight_precision = weight_precision
        self.activation_precision = activation_precision

        # assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert (
            in_features % (inner_k_tiles * 16) == 0
        ), "require in_features % (innerKTiles * 16) == 0"
        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty(
                (in_features // group_size, out_features, 2),
                dtype=self.weight_precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.activation_precision)
        input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))

        (
            scales,
            zero_points,
        ) = torch.ops.quantized_decomposed.choose_qparams_per_token(input, torch.int8)

        # TODO: get these from torch.int8
        quant_min = -128
        quant_max = 127
        input = torch.ops.quantized_decomposed.quantize_per_token(
            input, scales, zero_points, quant_min, quant_max, torch.int8
        )
        input = torch.ops.quantized_decomposed.dequantize_per_token(
            input,
            scales,
            zero_points,
            quant_min,
            quant_max,
            torch.int8,
            self.activation_precision,
        )

        input = input.to(self.activation_precision)
        return linear_forward_int4(
            input,
            self.weight,
            self.scales_and_zeros,
            self.out_features,
            self.group_size,
            self.weight_precision,
        )
