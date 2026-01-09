# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def validate_conv2d_args_dtypes(
    tosa_spec: TosaSpecification,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    op: str = "CONV2D",
) -> torch.dtype:
    output_dtype = None
    supported_int_types = (torch.int8, torch.int16)
    supported_float_types = (
        torch.float16,
        torch.float32,
    )
    if x.dtype in supported_int_types:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {x.dtype} but found input type {x.dtype}",
                op=op,
            )
        if weight.dtype not in (torch.int8,):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} only supports {torch.int8} weights for {x.dtype} input but found {weight.dtype}",
                op=op,
            )
        if bias is not None and bias.dtype not in (torch.int32,):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} only supports {torch.int32} bias for {x.dtype} input but found {bias.dtype}",
                op=op,
            )
        output_dtype = torch.int32

    elif x.dtype in supported_float_types:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {x.dtype} but found input type {x.dtype}",
                op=op,
            )
        if weight.dtype != x.dtype:
            raise TosaValueError(
                f"TOSA spec {tosa_spec} requires weights {weight.dtype} to be of the same type as input {x.dtype}",
                op=op,
            )
        if bias is not None and bias.dtype != x.dtype:
            raise TosaValueError(
                f"TOSA spec {tosa_spec} requires bias {bias.dtype} to be of the same type as input {x.dtype}",
                op=op,
            )
        output_dtype = x.dtype
    else:
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype}, supported types are {supported_int_types + supported_float_types} ",
            op=op,
        )
    return output_dtype


@register_fake_tosa_op(
    "CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[2] stride, "
    "int[4] pad, "
    "int[2] dilation) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(tosa_spec, x, weight, bias, op="CONV2D")

    torch_pad = [pad[0], pad[2]]
    N = x.shape[0]
    C_out = weight.shape[0]
    H_in, W_in = x.shape[2:]
    H_out = math.floor(
        (H_in + 2 * torch_pad[0] - dilation[0] * (weight.shape[2] - 1) - 1) / stride[0]
        + 1
    )
    W_out = math.floor(
        (W_in + 2 * torch_pad[1] - dilation[1] * (weight.shape[3] - 1) - 1) / stride[1]
        + 1
    )
    output_shape = [N, C_out, H_out, W_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
