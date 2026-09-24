# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.pass_base import ProxyValue
from torch.export.graph_signature import InputKind


def create_per_tensor_qparams(
    builder: ProgramBuilder,
    scale: float = 1.0,
    zero_point: int = 0,
    dtype: torch.dtype = torch.int8,
) -> tuple[ProxyValue, ProxyValue, torch.dtype, int, int]:
    """Create per-tensor QuantParams as lifted scalar constants."""
    idx = len(builder.input_specs)
    scale_node = builder.placeholder(
        f"qp_scale_{idx}",
        torch.tensor(scale, dtype=torch.float32),
        input_kind=InputKind.CONSTANT_TENSOR,
    )
    zp_node = builder.placeholder(
        f"qp_zero_point_{idx}",
        torch.tensor(zero_point, dtype=torch.int64),
        input_kind=InputKind.CONSTANT_TENSOR,
    )
    return (scale_node, zp_node, dtype, -128, 127)


def create_per_channel_qparams(
    builder: ProgramBuilder,
    num_channels: int,
    idx: int = 0,
    scale: float = 1.0,
    zero_point: int = 0,
    dtype: torch.dtype = torch.int8,
    weight_ndim: int = 2,
) -> tuple[ProxyValue, ProxyValue, torch.dtype, int, int]:
    """Create per-channel QuantParams using BUFFER placeholders.

    Scales/zero_points are full-rank ``[num_channels, 1, ...]`` so their shape
    encodes the affine block layout (channel axis 0).
    """
    full_shape = (num_channels,) + (1,) * (weight_ndim - 1)
    scale_node = builder.placeholder(
        f"wt_scale_{idx}",
        torch.full(full_shape, scale, dtype=torch.float32),
        input_kind=InputKind.BUFFER,
    )
    zp_node = builder.placeholder(
        f"wt_zp_{idx}",
        torch.full(full_shape, zero_point, dtype=torch.int64),
        input_kind=InputKind.BUFFER,
    )
    return (scale_node, zp_node, dtype, -128, 127)


def create_per_axis_qparams(
    builder: ProgramBuilder,
    num_channels: int,
    ndim: int,
    axis: int,
    dtype: torch.dtype = torch.int8,
    vary_values: bool = False,
) -> tuple[ProxyValue, ProxyValue, torch.dtype, int, int]:
    """Create full-rank qparams whose non-unitary dimension is ``axis``."""
    idx = len(builder.input_specs)
    shape = [1] * ndim
    shape[axis] = num_channels
    scales = (
        0.5 + torch.arange(num_channels, dtype=torch.float32) / num_channels
        if vary_values
        else torch.ones(num_channels, dtype=torch.float32)
    )
    zero_points = (
        torch.arange(num_channels, dtype=torch.int64) - num_channels // 2
        if vary_values
        else torch.zeros(num_channels, dtype=torch.int64)
    )
    scale_node = builder.placeholder(
        f"axis_scale_{idx}",
        scales.reshape(shape),
        input_kind=InputKind.BUFFER,
    )
    zp_node = builder.placeholder(
        f"axis_zp_{idx}",
        zero_points.reshape(shape),
        input_kind=InputKind.BUFFER,
    )
    return (scale_node, zp_node, dtype, -128, 127)
