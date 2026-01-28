# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.conv2d import validate_conv2d_args_dtypes
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "TRANSPOSE_CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[4] out_pad, "
    "int[2] stride) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def TRANSPOSE_CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_pad: list[int],
    stride: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(
        tosa_spec, x, weight, bias, op="TRANSPOSE_CONV2D"
    )

    if len(out_pad) != 4:
        raise TosaValueError(
            f"TRANSPOSE_CONV2D expects out_pad with 4 values, got {out_pad}",
            op="TRANSPOSE_CONV2D",
        )
    if len(stride) != 2:
        raise TosaValueError(
            f"TRANSPOSE_CONV2D expects stride with 2 values, got {stride}",
            op="TRANSPOSE_CONV2D",
        )

    N = x.shape[0]
    C_out = weight.shape[1]
    H_in, W_in = x.shape[2:]
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]

    H_out = (H_in - 1) * stride[0] + out_pad[0] + out_pad[1] + kernel_h
    W_out = (W_in - 1) * stride[1] + out_pad[2] + out_pad[3] + kernel_w
    output_shape = [N, C_out, H_out, W_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
