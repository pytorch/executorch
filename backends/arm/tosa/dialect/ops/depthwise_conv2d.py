# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.ops.conv2d import validate_conv2d_args_dtypes
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops


@register_fake_tosa_op(
    "DEPTHWISE_CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[2] stride, "
    "int[4] pad, "
    "int[2] dialation, "
    "bool transposed, "
    "int[2] output_padding, "
    "int groups) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def DEPTHWISE_CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int],
    dialation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(
        tosa_spec, x, weight, bias, op="DEPTHWISE_CONV2D"
    )

    torch_pad = [pad[0], pad[2]]
    H, W = weight.shape[0], weight.shape[2]
    in_channels_group = x.shape[1] // groups
    out_channels = weight.shape[1] * x.shape[1]
    torch_weight = weight.reshape(out_channels, in_channels_group, H, W)
    aten_fake_tensor = exir_ops.edge.aten.convolution.default(
        x,
        torch_weight,
        bias,
        stride,
        torch_pad,
        dialation,
        transposed,
        output_padding,
        groups,
    )
    return aten_fake_tensor.to(dtype=output_dtype)
