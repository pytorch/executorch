#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 export-compatible quantization.

Upstream NVFP4Tensor's dequantize() uses raw Python ops that don't survive
run_decompositions. This module registers a torch.library custom op
(torchao::dequantize_nvfp4) so the dequant node persists through export,
similar to how dequantize_affine works for int4.

Usage:
    from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Config
    from torchao.quantization import quantize_

    quantize_(model, ExportableNVFP4Config())
"""

import types
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.quantization.quant_api import _quantization_type
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


from typing import Optional


@torch.library.custom_op("torchao::dequantize_nvfp4", mutates_args=())
def nvfp4_dequantize(
    qdata: Tensor,
    scale: Tensor,
    per_tensor_scale: Tensor,
    block_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Dequantize NVFP4 packed data."""
    data_unpacked = unpack_uint4(qdata.view(torch.uint8).contiguous())
    data_f32 = f4_unpacked_to_f32(data_unpacked)

    M = data_f32.shape[0]
    K = data_f32.shape[1]

    data_f32 = data_f32.view(M, K // block_size, block_size)
    scale_fp8 = scale.view(torch.float8_e4m3fn)
    scale_f32 = scale_fp8.to(torch.float32).view(M, K // block_size, 1)
    scale_f32 = per_tensor_scale * scale_f32
    result = (data_f32 * scale_f32).view(M, K)
    return result.to(output_dtype)


@nvfp4_dequantize.register_fake
def _(qdata, scale, per_tensor_scale, block_size, output_dtype=torch.float32):
    M = qdata.shape[0]
    K = qdata.shape[1] * 8  # 8 FP4 values per uint32
    return torch.empty(M, K, dtype=output_dtype, device=qdata.device)


class ExportableNVFP4Tensor(TorchAOBaseTensor):
    """NVFP4 tensor subclass that dequantizes via a registered custom op."""

    tensor_data_names = ["qdata", "scale", "per_tensor_scale"]
    tensor_attribute_names = ["block_size", "orig_dtype"]

    def __new__(cls, qdata, scale, per_tensor_scale, block_size, orig_dtype):
        K = qdata.shape[-1] * 8  # 8 FP4 values per uint32
        shape = (qdata.shape[0], K)
        self = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=orig_dtype, device=qdata.device, requires_grad=False
        )
        self.qdata = qdata
        self.scale = scale
        self.per_tensor_scale = per_tensor_scale
        self.block_size = block_size
        self.orig_dtype = orig_dtype
        return self

    def dequantize(self, output_dtype=None):
        dtype = output_dtype or self.orig_dtype
        return torch.ops.torchao.dequantize_nvfp4(
            self.qdata,
            self.scale,
            self.per_tensor_scale,
            self.block_size,
            output_dtype=dtype,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


implements = ExportableNVFP4Tensor.implements


@implements([aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None
    weight_dequant = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_dequant, bias)


@implements([aten.embedding.default])
def _(func, types, args, kwargs):
    weight_tensor = args[0]
    indices = args[1]
    weight_dequant = weight_tensor.dequantize()
    return torch.nn.functional.embedding(indices, weight_dequant)


@implements([aten.t.default])
def _(func, types, args, kwargs):
    return args[0].dequantize().t()


@implements([aten.detach.default])
def _(func, types, args, kwargs):
    return args[0]


@implements([aten._to_copy.default])
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].orig_dtype)
    return args[0].dequantize(output_dtype=dtype)


@dataclass
class ExportableNVFP4Config(AOBaseConfig):
    """NVFP4 weight-only quantization config for torch.export."""

    use_per_tensor_scale: bool = True


def _linear_extra_repr(self):
    return (
        f"in_features={self.weight.shape[1]}, "
        f"out_features={self.weight.shape[0]}, "
        f"weight={_quantization_type(self.weight)}"
    )


@register_quantize_module_handler(ExportableNVFP4Config)
def _exportable_nvfp4_transform(module: nn.Module, config: ExportableNVFP4Config):
    weight = module.weight

    if weight.shape[-2] % 16 != 0 or weight.shape[-1] % 16 != 0:
        raise RuntimeError(
            f"NVFP4 requires weight dims divisible by 16, got {weight.shape}"
        )

    per_tensor_scale = 1.0
    if config.use_per_tensor_scale:
        tensor_amax = torch.max(torch.abs(weight))
        per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

    scales_fp8, qdata_packed = nvfp4_quantize(
        weight, block_size=16, per_tensor_scale=per_tensor_scale
    )

    qdata_u32 = qdata_packed.view(torch.uint32)
    scales_u8 = scales_fp8.view(torch.uint8)

    pts = torch.tensor(per_tensor_scale, dtype=torch.float32)
    quantized_weight = ExportableNVFP4Tensor(
        qdata_u32,
        scales_u8,
        pts,
        block_size=16,
        orig_dtype=weight.dtype,
    )
    module.weight = nn.Parameter(quantized_weight, requires_grad=False)
    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
