# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_identity_tosa_FP() -> None:
    sample_input = torch.randn((1, 2, 3, 4), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.IDENTITY.default(mode.from_tensor(sample_input))

        assert output.dtype == sample_input.dtype
        assert tuple(output.shape) == tuple(sample_input.shape)
