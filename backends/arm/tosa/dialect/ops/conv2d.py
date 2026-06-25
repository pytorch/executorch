# Copyright 2025-2026 Arm Limited and/or its affiliates.
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


def validate_conv2d_args_dtypes(  # noqa: C901
    tosa_spec: TosaSpecification,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    op: str = "CONV2D",
) -> torch.dtype:
    output_dtype = None
    supported_int_types = (torch.int8, torch.int16)
    supported_float_types = [
        torch.float16,
        torch.float32,
    ]
    if tosa_spec.support_extension("bf16"):
        supported_float_types.append(torch.bfloat16)
    if tosa_spec.support_extension("fp8e4m3"):
        supported_float_types.append(torch.float8_e4m3fn)
    if tosa_spec.support_extension("fp8e5m2"):
        supported_float_types.append(torch.float8_e5m2)
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
        if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            output_dtype = torch.float16
        else:
            output_dtype = x.dtype
        if bias is not None and bias.dtype != output_dtype:
            if output_dtype != x.dtype:
                raise TosaValueError(
                    f"TOSA spec {tosa_spec} requires bias {bias.dtype} to be of the same type as output {output_dtype}",
                    op=op,
                )
            raise TosaValueError(
                f"TOSA spec {tosa_spec} requires bias {bias.dtype} to be of the same type as input {x.dtype}",
                op=op,
            )
    else:
        supported_types = (
            *(supported_int_types if tosa_spec.support_integer() else ()),
            *(supported_float_types if tosa_spec.support_float() else ()),
        )
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype}, supported types are {supported_types} ",
            op=op,
        )
    return output_dtype


def conv_output_dim(
    input_dim: int | torch.SymInt,
    kernel_dim: int,
    stride: int,
    pad_before: int | torch.SymInt,
    pad_after: int | torch.SymInt,
    dilation: int,
) -> int | torch.SymInt:
    receptive_field = dilation * (kernel_dim - 1) + 1
    total_pad = pad_before + pad_after

    if stride == 1:
        return input_dim + total_pad - receptive_field + 1

    return (input_dim + total_pad - receptive_field) // stride + 1


@register_fake_tosa_op(
    "CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[2] stride, "
    "SymInt[4] pad, "
    "int[2] dilation) -> Tensor",  # schema
    TosaSpecification.all_versions_and_profiles(),  # target TOSA specifications
)
def CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int | torch.SymInt],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(tosa_spec, x, weight, bias, op="CONV2D")

    N = x.shape[0]
    H_in, W_in = x.shape[1:3]
    C_out = weight.shape[0]
    H_out = conv_output_dim(
        H_in, weight.shape[1], stride[0], pad[0], pad[1], dilation[0]
    )
    W_out = conv_output_dim(
        W_in, weight.shape[2], stride[1], pad[2], pad[3], dilation[1]
    )
    output_shape = [N, H_out, W_out, C_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
