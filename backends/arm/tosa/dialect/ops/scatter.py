# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import TosaSpecification


@register_fake_tosa_op(
    "SCATTER(Tensor values_in, Tensor indices, Tensor input) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def SCATTER(
    x: torch.Tensor,
    indices: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:

    return torch.empty(x.shape, dtype=x.dtype)
