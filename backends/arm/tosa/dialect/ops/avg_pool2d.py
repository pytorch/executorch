# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Union

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "AVG_POOL2D(Tensor input, Tensor input_zp, Tensor output_zp, int[2] kernel, int[2] stride, SymInt[4] pad, ScalarType acc_type) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def AVG_POOL2D(
    x: torch.Tensor,
    input_zp: torch.Tensor,
    output_zp: torch.Tensor,
    kernel: List[int],
    stride: List[int],
    pad: List[Union[int, torch.SymInt]],
    acc_type: torch.dtype,
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    supported_dtypes = []
    if tosa_spec.support_integer():
        supported_dtypes.extend([torch.int8])
    if tosa_spec.support_float():
        supported_dtypes.extend([torch.float16, torch.float32])
    if tosa_spec.support_extension("bf16"):
        supported_dtypes.append(torch.bfloat16)
    if tosa_spec.support_extension("int16"):
        supported_dtypes.append(torch.int16)

    if x.dtype not in supported_dtypes:
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype}, supported types are {supported_dtypes}",
            op="AVG_POOL2D",
        )

    # Input is NHWC: [N, H, W, C]
    N = x.shape[0]
    H_in = x.shape[1]
    W_in = x.shape[2]
    C = x.shape[3]

    # pad is [top, bottom, left, right]
    H_out = math.floor((H_in + pad[0] + pad[1] - kernel[0]) / stride[0]) + 1
    W_out = math.floor((W_in + pad[2] + pad[3] - kernel[1]) / stride[1]) + 1

    output_shape = [N, H_out, W_out, C]
    return x.new_empty(output_shape, dtype=x.dtype)
