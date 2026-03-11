# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "SLICE(Tensor input1, SymInt[] start, SymInt[] size) -> Tensor",  # schema
    TosaSpecification.all_versions_and_profiles(),  # target TOSA specifications
)
def SLICE(a, start, size):
    tosa_spec = get_context_spec()

    # Rank validation
    input_rank = a.dim()
    if input_rank != len(start):
        raise TosaValueError(
            f"start list does not have the same rank {len(start)} as input {input_rank}"
        )
    if len(start) != len(size):
        raise TosaValueError(
            f"size list does not have the same rank {len(size)} as start list {len(start)}"
        )

    # Shape validation
    for i in range(len(start)):
        dim_start = start[i]
        if dim_start < 0 or dim_start > a.shape[i]:
            raise TosaValueError(
                f"Expected start values between [0, {a.shape[i]}] but got {dim_start}"
            )
        dim_size = size[i]
        if dim_size < 0 or dim_start + dim_size > a.shape[i]:
            raise TosaValueError(
                f"Expected start + size values between [0, {a.shape[i]}] but got {dim_start + dim_size}"
            )

    # Dtype validation
    supported_dtypes = [torch.bool]
    if tosa_spec.support_integer():
        supported_dtypes += [torch.int8, torch.int16, torch.int32]
    if tosa_spec.support_float():
        supported_dtypes += [torch.float16, torch.float32]
    if tosa_spec.support_extension("bf16"):
        supported_dtypes += [torch.bfloat16]

    if a.dtype not in supported_dtypes:
        raise TosaValueError(
            f"Unsupported dtype {a.dtype} for SLICE. Supported dtypes are {supported_dtypes}"
        )

    return torch.empty(size=size, dtype=a.dtype)
