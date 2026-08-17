#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MX (microscaling) export-compatible quantization.

General OCP Microscaling **MX** block-scaled float tensor (mxfp8, and — once
their pack/unpack is wired — mxfp4/mxfp6). An MX weight is a block of ``elem_dtype``
elements sharing one ``float8_e8m0`` (power-of-two) scale per ``block_size``
group; there is no per-tensor global scale and no affine zero-point.

Storage mirrors torchao's ``MXTensor`` (backend-agnostic), NOT any one backend's
packing:
  * ``qdata``  the elements in their native dtype -- e.g. ``float8_e4m3fn`` for
               mxfp8 (one value per element, unpacked).
  * ``scale``  ``float8_e8m0fnu`` (biased power-of-two exponent), one per group.

A ``torchao::dequantize_mx`` custom op carries the dequant so the node survives
``torch.export`` (mirroring ``dequantize_nvfp4`` / ``dequantize_int4_tensor``).
Backends pattern-match the op and repack ``qdata``/``scale`` into whatever layout
their kernel wants (e.g. the MLX handler packs FP8 into uint32 words) and reject
element formats they do not support.
"""

import types
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import _quantization_type
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

_MX_DEFAULT_BLOCK_SIZE = 32

# Element encodings this carrier can represent/dequantize. All MX formats share
# the E8M0 (float8_e8m0fnu) block scale; only the element dtype varies. FP8
# elements are stored one-per-byte (unpacked); sub-byte formats (fp4) would add
# their own packing and are intentionally not enabled yet.
_MX_FP8_ELEM_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MX_SUPPORTED_ELEM_DTYPES = _MX_FP8_ELEM_DTYPES


@torch.library.custom_op("torchao::dequantize_mx", mutates_args=())
def mx_dequantize(
    qdata: Tensor,
    scale: Tensor,
    elem_dtype: torch.dtype,
    block_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Reference dequant of an MX weight to ``(M, K)``.

    ``qdata`` holds the elements in their native ``elem_dtype`` (FP8, one value
    per element); ``scale`` is ``float8_e8m0fnu`` (a power-of-two), one per
    ``block_size`` group along ``K``. This eager body is a portable reference --
    a backend (e.g. MLX) replaces the op with its fused kernel at lower time.
    """
    if elem_dtype not in _MX_FP8_ELEM_DTYPES:
        raise NotImplementedError(
            f"dequantize_mx reference only supports FP8 element dtypes "
            f"{_MX_FP8_ELEM_DTYPES}, got {elem_dtype}"
        )
    data_f32 = qdata.to(torch.float32)
    M = data_f32.shape[0]
    K = data_f32.shape[1]
    data_f32 = data_f32.view(M, K // block_size, block_size)

    # float8_e8m0fnu decodes directly to the fp32 power-of-two scale factor.
    scale_f32 = scale.to(torch.float32).view(M, K // block_size, 1)

    result = (data_f32 * scale_f32).view(M, K)
    return result.to(output_dtype)


@mx_dequantize.register_fake
def _(qdata, scale, elem_dtype, block_size, output_dtype=torch.float32):
    # FP8 elements are stored unpacked, so the logical last dim == qdata's.
    return torch.empty(
        qdata.shape[0], qdata.shape[1], dtype=output_dtype, device=qdata.device
    )


class ExportableMXTensor(TorchAOBaseTensor):
    """General MX tensor subclass that dequantizes via a registered custom op.

    Parametrized by ``elem_dtype`` (the block-float element encoding); the block
    scale is always ``float8_e8m0fnu``. Storage is backend-agnostic (native
    element dtype for ``qdata``) -- backends repack as needed in their handlers.
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["elem_dtype", "block_size", "orig_dtype"]

    def __new__(cls, qdata, scale, elem_dtype, block_size, orig_dtype):
        if elem_dtype not in _MX_SUPPORTED_ELEM_DTYPES:
            raise NotImplementedError(
                f"ExportableMXTensor supports element dtypes "
                f"{_MX_SUPPORTED_ELEM_DTYPES}, got {elem_dtype}"
            )
        # FP8 elements: qdata is unpacked (one value per element), so its shape
        # is already the logical (N, K).
        shape = (qdata.shape[0], qdata.shape[-1])
        self = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=orig_dtype, device=qdata.device, requires_grad=False
        )
        self.qdata = qdata
        self.scale = scale
        self.elem_dtype = elem_dtype
        self.block_size = block_size
        self.orig_dtype = orig_dtype
        return self

    def dequantize(self, output_dtype=None):
        return torch.ops.torchao.dequantize_mx(
            self.qdata,
            self.scale,
            self.elem_dtype,
            self.block_size,
            output_dtype=output_dtype or self.orig_dtype,
        )

    def to(self, *args, **kwargs) -> "ExportableMXTensor":
        """Move device and/or set the output dtype *without* dequantizing.

        Mirrors ``ExportableInt4Tensor.to``: ``qdata`` (native FP8) and ``scale``
        (E8M0) only move across devices -- neither is a floating scale, so
        casting to the activation dtype must not touch their bit patterns. Only
        ``orig_dtype`` (the dequantized output dtype) follows ``dtype``.
        """
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        assert dtype.is_floating_point, f"expected a floating dtype; got {dtype}"
        return ExportableMXTensor(
            self.qdata.to(device),
            self.scale.to(device),
            self.elem_dtype,
            self.block_size,
            dtype,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


implements = ExportableMXTensor.implements


@implements([aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None
    weight_dequant = weight_tensor.dequantize(input_tensor.dtype)
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


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return args[0]


@implements([aten._to_copy.default])
def _(func, types, args, kwargs):
    return args[0].dequantize(output_dtype=kwargs.get("dtype", args[0].orig_dtype))


@dataclass
class ExportableMXConfig(AOBaseConfig):
    """MX weight-only quantization config for torch.export.

    ``elem_dtype`` selects the block-float element encoding (default mxfp8 =
    ``float8_e4m3fn``); the scale is always E8M0.
    """

    elem_dtype: torch.dtype = torch.float8_e4m3fn
    block_size: int = _MX_DEFAULT_BLOCK_SIZE


def _linear_extra_repr(self):
    return (
        f"in_features={self.weight.shape[1]}, "
        f"out_features={self.weight.shape[0]}, "
        f"weight={_quantization_type(self.weight)}"
    )


@register_quantize_module_handler(ExportableMXConfig)
def _exportable_mx_transform(module: nn.Module, config: ExportableMXConfig):
    """Quantize a linear/embedding weight to MX via torchao ``to_mx``.

    ``to_mx`` returns the elements in ``elem_dtype`` and an E8M0 block scale --
    exactly the backend-agnostic form ``ExportableMXTensor`` stores. No backend
    packing happens here (backends pack in their own handlers).
    """
    from torchao.prototype.mx_formats.mx_tensor import to_mx

    weight = module.weight
    block_size = config.block_size
    if weight.shape[-1] % block_size != 0:
        raise RuntimeError(
            f"MX requires in_features divisible by {block_size}, "
            f"got {tuple(weight.shape)}"
        )

    # to_mx supports bf16/fp32 inputs; promote otherwise.
    hp = weight.contiguous()
    if hp.dtype not in (torch.bfloat16, torch.float32):
        hp = hp.to(torch.float32)

    scale_e8m0, data_lp = to_mx(hp, config.elem_dtype, block_size)
    # data_lp: (N, K) in elem_dtype; scale_e8m0: (N, n_groups) float8_e8m0fnu.
    # Stored as-is (torchao-native); the backend repacks in its handler.
    quantized_weight = ExportableMXTensor(
        data_lp.contiguous(),
        scale_e8m0.contiguous(),
        elem_dtype=config.elem_dtype,
        block_size=block_size,
        orig_dtype=weight.dtype,
    )
    module.weight = nn.Parameter(quantized_weight, requires_grad=False)
    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
