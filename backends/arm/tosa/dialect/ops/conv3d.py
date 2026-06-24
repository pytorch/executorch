# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.conv2d import (
    conv_output_dim,
    validate_conv2d_args_dtypes,
)
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
    "SymInt[6] pad, "
    "int[3] dilation) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def CONV3D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int | torch.SymInt],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv3d_args_dtypes(tosa_spec, x, weight, bias)

    N = x.shape[0]
    C_out = weight.shape[0]
    D_in, H_in, W_in = x.shape[1:4]
    D_out = conv_output_dim(
        D_in, weight.shape[1], stride[0], pad[0], pad[1], dilation[0]
    )
    H_out = conv_output_dim(
        H_in, weight.shape[2], stride[1], pad[2], pad[3], dilation[1]
    )
    W_out = conv_output_dim(
        W_in, weight.shape[3], stride[2], pad[4], pad[5], dilation[2]
    )
    output_shape = [N, D_out, H_out, W_out, C_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
