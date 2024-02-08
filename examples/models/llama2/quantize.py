# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.quantized_ops import *  # noqa


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

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
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


def get_group_qparams(w, n_bit=4, groupsize=128):
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
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
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
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def group_dequantize_tensor_from_qparams(
    w_int32, scales, zeros, n_bit=4, groupsize=128
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int32.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int32.shape[-1]
    assert w_int32.shape[-1] % groupsize == 0
    assert w_int32.dim() == 2

    w_int32_grouped = w_int32.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = (
        w_int32_grouped.sub(2 ** (n_bit - 1)).mul(scales).add(zeros).reshape_as(w_int32)
    )
    return w_dq


def group_dequantize_tensor(w_int32, scales_and_zeros, n_bit=4, groupsize=128):
    scales, zeros = unpack_scales_and_zeros(scales_and_zeros)
    return group_dequantize_tensor_from_qparams(
        w_int32, scales, zeros, n_bit, groupsize
    )


class QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    def create_quantized_state_dict(self) -> Dict:  # "StateDict"
        pass

    def convert_for_runtime(self) -> nn.Module:
        pass


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(
    module, group_size: Optional[int] = None
):
    assert group_size is None, "Linear does not support group-wise quantization"
    for name, child in module.named_children():
        print(f"name: {name}")
        if name == "XXXXoutputXXXXXXX":
            print("skipping quantizing output")
        elif isinstance(child, nn.Linear):
            print(f"{name, child}")
            print(f"in_features: {child.in_features}")
            print(f"out_features: {child.out_features}")
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self) -> Dict:
        cur_state_dict = self.mod.state_dict()

        for fqn, mod in self.mod.named_modules():
            print(f"quantized {fqn}")
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)

        return cur_state_dict

    def convert_for_runtime(self) -> nn.Module:
        replace_linear_weight_only_int8_per_channel(self.mod)
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


def embedding_quant(
    weight, group_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embedding quantization with per-row group scaling
    """
    if group_size is None:
        group_size = weight.size(1)
    weight_fp16 = weight.to(torch.half)
    weight_rows = weight_fp16.shape[0]
    weight_cols = weight_fp16.shape[1]
    assert weight_cols % group_size == 0, "colums must be a multiple of group size"
    groups_per_row = weight_cols // group_size
    weights_int8 = torch.zeros(weight_fp16.shape, dtype=torch.int8)
    scales_fp16 = torch.empty((weight_rows, groups_per_row), dtype=torch.float16)
    print("weights shape: ", weights_int8.shape)
    print("scales shape: ", scales_fp16.shape)
    for r in range(0, weight_cols, group_size):
        weights_group, scales, _ = dynamically_quantize_per_channel(
            weight_fp16[:, r : r + group_size], -128, 127, torch.int8
        )
        weights_int8[:, r : r + group_size] = weights_group
        scales_fp16[:, r // group_size] = scales
        scales_fp16 = scales_fp16.squeeze()

    return weights_int8, scales_fp16


def replace_embedding_weight_only_grouped_int8_per_channel(
    module, group_size: Optional[int] = None
):
    for name, child in module.named_children():
        print(f"name: {name}")
        if name == "XXXXoutputXXXXXXX":
            print("skipping quantizing output")
        elif isinstance(child, nn.Embedding):
            print(f"{name, child}")
            print(f"weights size: {child.weight.size()}")
            weight_int8, scales_fp16 = embedding_quant(child.weight, group_size)
            setattr(
                module,
                name,
                QuantizedGroupEmbedding(
                    weight_int8, scales_fp16, group_size=group_size
                ),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child, group_size)


class EmbeddingOnlyInt8QuantHandler:
    def __init__(self, mod, group_size: Optional[int] = None):
        self.mod = mod
        self.group_size = group_size

    def convert_for_runtime(self) -> nn.Module:
        replace_embedding_weight_only_grouped_int8_per_channel(
            self.mod, self.group_size
        )
        return self.mod


class QuantizedGroupEmbedding(torch.nn.Module):
    def __init__(
        self,
        weight_int8: torch.Tensor,
        scales_fp16: torch.Tensor,
        group_size: Optional[int] = None,
        device=None,
        dtype=torch.half,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.dtype = dtype
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scales_fp16", scales_fp16)

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.llama_quantized.embedding_byte.default(
            self.weight_int8, self.scales_fp16, None, 0, 0, indices
        )


#        result_weights = self.weight_int8.index_select(0, indices.view(-1))
#        result_scales = self.scales_fp16.index_select(0, indices.view(-1))
#
#        r = result_weights.to(dtype=result_scales.dtype) * result_scales
#        return r.view(indices.size() + (-1,))
