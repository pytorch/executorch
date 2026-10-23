# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Explicit tensor constants in the TOSA dialect."""

import torch

from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_tosa_op(
    "CONST(Scalar[] values, int[] shape, ScalarType dtype) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
    include_real_impl=False,
)
def CONST(
    values: list[int | float], shape: list[int], dtype: torch.dtype
) -> torch.Tensor:
    if len(values) != torch.Size(shape).numel():
        raise ValueError("CONST values must match its shape")
    return torch.empty(shape, dtype=dtype)
