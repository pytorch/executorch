# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.conv2d import validate_conv2d_args_dtypes
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops


def validate_conv3d_args_dtypes(
    tosa_spec: TosaSpecification,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.dtype:
    if len(x.shape) != 5 or len(weight.shape) != 5:
        raise TosaValueError(
            f"Expected 5D input/weight tensors for CONV3D, got {x.shape} and {weight.shape}",
            op="CONV3D",
        )
    return validate_conv2d_args_dtypes(tosa_spec, x, weight, bias, op="CONV3D")


@register_fake_tosa_op(
    "CONV3D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[3] stride, "
    "int[6] pad, "
    "int[3] dialation, "
    "bool transposed, "
    "int[3] output_padding, "
    "int groups) -> Tensor",
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),
)
def CONV3D(
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

    output_dtype = validate_conv3d_args_dtypes(tosa_spec, x, weight, bias)

    torch_pad = [pad[0], pad[2], pad[4]]
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
