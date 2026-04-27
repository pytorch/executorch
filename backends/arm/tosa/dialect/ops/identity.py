# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_fake_tosa_op(
    "IDENTITY(Tensor input) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def IDENTITY(a):
    return torch.empty_like(a, dtype=a.dtype)
