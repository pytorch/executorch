#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Int4 export-compatible quantization.

Wraps a torchao ``Int4Tensor`` (nibble-packed 4-bit groupwise weight) so it
survives ``torch.export`` / ``run_decompositions``: a ``torchao::dequantize_int4_tensor``
custom op carries the dequant, and ``aten.linear`` / ``aten.embedding`` desugar to
``dequantize_int4_tensor -> linear/embedding`` (mirroring ``dequantize_nvfp4`` /
``dequantize_gguf``). A backend may pattern-match the op to a low-bit kernel; the
eager body is a plain affine dequant so the representation is portable.

The tensor stores the ``Int4Tensor`` layout verbatim:
  * ``qdata``      ``(N, K // 2)`` uint8, two nibbles/byte (even index -> low nibble),
                   unsigned values in [0, 15].
  * ``scale``      ``(K // group_size, N)``.
  * ``zero_point`` ``(K // group_size, N)``, unsigned values in [0, 15].
Dequant is ``scale * (q - zero_point)`` per group.
"""

import torch
from torch import Tensor
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


def _dequantize_int4(
    qdata: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> Tensor:
    """Eager affine dequant of an ``Int4Tensor``-layout weight to ``(N, K)``."""
    p = qdata.view(torch.uint8)
    low = (p & 0x0F).to(torch.int32)
    high = ((p >> 4) & 0x0F).to(torch.int32)
    # Two nibbles/byte: even index -> low, odd -> high.
    q = torch.stack([low, high], dim=-1).reshape(p.shape[0], -1).to(torch.float32)

    # scale / zero_point are (K // gs, N) -> transpose to (N, K // gs) and expand.
    s = scale.t().to(torch.float32).repeat_interleave(group_size, dim=-1)
    z = zero_point.t().to(torch.float32).repeat_interleave(group_size, dim=-1)
    return ((q - z) * s).to(output_dtype)


@torch.library.custom_op("torchao::dequantize_int4_tensor", mutates_args=())
def dequantize_int4_tensor(
    qdata: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    group_size: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize a nibble-packed Int4 weight (``(N, K//2)`` uint8) to ``(N, K)``."""
    return _dequantize_int4(qdata, scale, zero_point, group_size, output_dtype)


@dequantize_int4_tensor.register_fake
def _(qdata, scale, zero_point, group_size, output_dtype=torch.bfloat16):
    K = qdata.shape[1] * 2  # two 4-bit values per byte
    return torch.empty(qdata.shape[0], K, dtype=output_dtype, device=qdata.device)


class ExportableInt4Tensor(TorchAOBaseTensor):
    """Int4 tensor subclass that dequantizes via a registered custom op."""

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["group_size", "orig_dtype"]

    def __new__(cls, qdata, scale, zero_point, group_size, orig_dtype):
        K = qdata.shape[-1] * 2  # two 4-bit values per byte
        shape = (qdata.shape[0], K)
        self = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=orig_dtype, device=qdata.device, requires_grad=False
        )
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.orig_dtype = orig_dtype
        return self

    @classmethod
    def from_int4_tensor(cls, w: Tensor) -> "ExportableInt4Tensor":
        """Build from a torchao ``Int4Tensor`` (copies its packed fields)."""
        return cls(
            w.qdata,
            w.scale,
            w.zero_point,
            int(w.block_size[-1]),
            w.dtype,
        )

    def dequantize(self, output_dtype=None):
        return torch.ops.torchao.dequantize_int4_tensor(
            self.qdata,
            self.scale,
            self.zero_point,
            self.group_size,
            output_dtype=output_dtype or self.orig_dtype,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


implements = ExportableInt4Tensor.implements


@implements([aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    return torch.nn.functional.linear(
        input_tensor, weight.dequantize(input_tensor.dtype), bias
    )


@implements([aten.embedding.default])
def _(func, types, args, kwargs):
    weight, indices = args[0], args[1]
    return torch.nn.functional.embedding(indices, weight.dequantize())


@implements([aten.t.default])
def _(func, types, args, kwargs):
    return args[0].dequantize().t()


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return args[0]


@implements([aten._to_copy.default])
def _(func, types, args, kwargs):
    return args[0].dequantize(output_dtype=kwargs.get("dtype", args[0].orig_dtype))
