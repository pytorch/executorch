# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops


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
        # TODO update to int32 for int8 inputs
        output_dtype = torch.int8 if x.dtype == torch.int8 else torch.int16

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
    "int[2] dialation, "
    "bool transposed, "
    "int[2] output_padding, "
    "int groups) -> Tensor",  # schema
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
    dialation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(tosa_spec, x, weight, bias, op="CONV2D")

    torch_pad = [pad[0], pad[2]]
    aten_fake_tensor = exir_ops.edge.aten.convolution.default(
        x,
        weight,
        bias,
        stride,
        torch_pad,
        dialation,
        transposed,
        output_padding,
        groups,
    )
    return aten_fake_tensor.to(dtype=output_dtype)
