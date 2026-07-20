# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.ops.conv2d import (
    conv_output_dim,
    validate_conv2d_args_dtypes,
)
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_tosa_op(
    "DEPTHWISE_CONV2D(Tensor input, "
    "Tensor weight, "
    "Tensor bias, "
    "int[2] stride, "
    "SymInt[4] pad, "
    "int[2] dialation) -> Tensor",  # schema
    TosaSpecification.all_versions_and_profiles(),
)
def DEPTHWISE_CONV2D(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    pad: list[int | torch.SymInt],
    dilation: list[int],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    output_dtype = validate_conv2d_args_dtypes(
        tosa_spec, x, weight, bias, op="DEPTHWISE_CONV2D"
    )

    kernel_h, kernel_w = weight.shape[0], weight.shape[1]
    C_out = weight.shape[2] * weight.shape[3]
    N = x.shape[0]
    H_in, W_in = x.shape[1:3]
    H_out = conv_output_dim(H_in, kernel_h, stride[0], pad[0], pad[1], dilation[0])
    W_out = conv_output_dim(W_in, kernel_w, stride[1], pad[2], pad[3], dilation[1])
    output_shape = [N, H_out, W_out, C_out]
    return torch.empty(size=output_shape, dtype=output_dtype)
