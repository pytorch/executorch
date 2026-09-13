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


def _to_fake(mode: FakeTensorMode, *values):
    return [
        mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
        for value in values
    ]


@pytest.mark.parametrize(
    (
        "op_name",
        "spec",
        "input1",
        "input2",
        "expected_shape",
    ),
    [
        pytest.param(
            "EQUAL",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            (2, 4, 3),
        ),
        pytest.param(
            "GREATER",
            "TOSA-1.1+FP",
            torch.randn((2, 1, 3), dtype=torch.float32),
            torch.randn((1, 4, 3), dtype=torch.float32),
            (2, 4, 3),
        ),
        pytest.param(
            "GREATER_EQUAL",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            (2, 4, 3),
        ),
    ],
)
def test_tosa_comparison_ops(
    op_name: str,
    spec: str,
    input1: torch.Tensor,
    input2: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            *_to_fake(mode, input1, input2)
        )

    assert output.dtype == torch.bool
    assert tuple(output.shape) == expected_shape


def test_equal_rejects_int8() -> None:
    input1 = torch.randint(-8, 8, (2, 3), dtype=torch.int8)
    input2 = torch.randint(-8, 8, (2, 3), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            exir_ops.backend.tosa.EQUAL.default(*_to_fake(mode, input1, input2))


@pytest.mark.parametrize("op_name", ["EQUAL", "GREATER", "GREATER_EQUAL"])
def test_compare_ops_reject_int32_on_fp_profile(op_name: str) -> None:
    input1 = torch.randint(1, 16, (2, 3), dtype=torch.int32)
    input2 = torch.randint(1, 8, (2, 3), dtype=torch.int32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support int32"):
            getattr(exir_ops.backend.tosa, op_name).default(
                *_to_fake(mode, input1, input2)
            )
