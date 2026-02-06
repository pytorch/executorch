# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.backends.arm.tosa.dialect.ops.conv2d import validate_conv2d_args_dtypes
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "DEPTHWISE_CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[2] stride, "
    "int[4] pad, "
    "int[2] dialation) -> Tensor",  # schema
    TosaSpecification.all_versions_and_profiles(),
)
def DEPTHWISE_CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(
        tosa_spec, x, weight, bias, op="DEPTHWISE_CONV2D"
    )

    torch_pad = [pad[0], pad[2]]
    kernel_h, kernel_w = weight.shape[0], weight.shape[2]
    C_out = weight.shape[1] * x.shape[1]
    N = x.shape[0]
    H_in, W_in = x.shape[2:]
    H_out = math.floor(
        (H_in + 2 * torch_pad[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1
    )
    W_out = math.floor(
        (W_in + 2 * torch_pad[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1
    )
    output_shape = [N, C_out, H_out, W_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
