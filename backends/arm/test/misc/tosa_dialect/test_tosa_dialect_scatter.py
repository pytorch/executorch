# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize(
    "dtype, extension",
    [
        (torch.float8_e4m3fn, "fp8e4m3"),
        (torch.float8_e5m2, "fp8e5m2"),
    ],
)
def test_scatter_tosa_FP_fp8(dtype: torch.dtype, extension: str):
    with TosaLoweringContext(
        TosaSpecification.create_from_string(f"TOSA-1.0+FP+{extension}")
    ), FakeTensorMode() as mode:
        values_in = mode.from_tensor(
            torch.rand((1, 5, 3), dtype=torch.float32).to(dtype)
        )
        indices = mode.from_tensor(torch.tensor([[1, 3]], dtype=torch.int32))
        input_tensor = mode.from_tensor(
            torch.rand((1, 2, 3), dtype=torch.float32).to(dtype)
        )
        output = exir_ops.backend.tosa.SCATTER.default(values_in, indices, input_tensor)

    assert output.dtype == dtype
    assert tuple(output.shape) == (1, 5, 3)
