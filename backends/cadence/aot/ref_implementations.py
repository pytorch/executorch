# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library


m = Library("cadence", "IMPL", "CompositeExplicitAutograd")

qdtype_map: dict[ScalarType, torch.dtype] = {
    ScalarType.QINT8: torch.qint8,
    ScalarType.QUINT8: torch.quint8,
    ScalarType.QINT32: torch.qint32,
}


@impl(m, "requantize")
def requantize(
    input: torch.Tensor,
    in_scale: torch.Tensor,
    in_zero_point: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    dtype: ScalarType,
) -> torch.Tensor:
    if dtype in qdtype_map:
        # Old quantization mechanism
        return torch.quantize_per_tensor(
            torch.dequantize(input), out_scale, out_zero_point, qdtype_map[dtype]
        )

    # For in_scale or out_scale other than scalar, it requires quant/dequant
    # per channel, but the channel dimension value is missing
    if in_scale.numel() > 1 or out_scale.numel() > 1:
        raise NotImplementedError("Only scalar scales are supported")

    quant_min = torch.iinfo(input.dtype).min
    quant_max = torch.iinfo(input.dtype).max
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_min = torch.iinfo(dtype).min
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_max = torch.iinfo(dtype).max
    return torch.ops.quantized_decomposed.quantize_per_tensor(
        torch.ops.quantized_decomposed.dequantize_per_tensor(
            input,
            in_scale.flatten()[0],
            in_zero_point.flatten()[0],
            quant_min,
            quant_max,
            input.dtype,
        ),
        out_scale.flatten()[0],
        out_zero_point.flatten()[0],
        out_quant_min,
        out_quant_max,
        dtype,
    )
