# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.conv2d import validate_conv2d_args_dtypes
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


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
    "int[3] dilation) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def CONV3D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv3d_args_dtypes(tosa_spec, x, weight, bias)

    torch_pad = [pad[0], pad[2], pad[4]]
    N = x.shape[0]
    C_out = weight.shape[0]
    D_in, H_in, W_in = x.shape[2:]
    D_out = math.floor(
        (D_in + 2 * torch_pad[0] - dilation[0] * (weight.shape[2] - 1) - 1) / stride[0]
        + 1
    )
    H_out = math.floor(
        (H_in + 2 * torch_pad[1] - dilation[1] * (weight.shape[3] - 1) - 1) / stride[1]
        + 1
    )
    W_out = math.floor(
        (W_in + 2 * torch_pad[2] - dilation[2] * (weight.shape[4] - 1) - 1) / stride[2]
        + 1
    )
    output_shape = [N, C_out, D_out, H_out, W_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
