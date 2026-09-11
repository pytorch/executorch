# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_argmax_tosa_fp() -> None:
    sample_input = torch.randn((2, 3, 4), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.ARGMAX.default(
            mode.from_tensor(sample_input),
            axis=1,
        )

    assert output.dtype == torch.int32
    assert tuple(output.shape) == (2, 4)


def test_argmax_rejects_bfloat16_without_extension() -> None:
    sample_input = torch.randn((2, 3, 4), dtype=torch.bfloat16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support bfloat16"):
            exir_ops.backend.tosa.ARGMAX.default(
                mode.from_tensor(sample_input),
                axis=1,
            )
