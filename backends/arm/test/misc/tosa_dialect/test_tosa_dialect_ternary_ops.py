# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import (
    get_registered_tosa_ops,
)
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
        "spec",
        "condition",
        "input1",
        "input2",
        "expected_shape",
        "expected_dtype",
    ),
    [
        pytest.param(
            "TOSA-1.1+INT",
            torch.randint(0, 2, (2, 1, 3), dtype=torch.bool),
            torch.randint(-8, 8, (1, 4, 3), dtype=torch.int32),
            torch.randint(-8, 8, (1, 4, 3), dtype=torch.int32),
            (2, 4, 3),
            torch.int32,
            id="int32",
        ),
        pytest.param(
            "TOSA-1.1+FP",
            torch.randint(0, 2, (2, 1, 3), dtype=torch.bool),
            torch.randn((1, 4, 3), dtype=torch.float32),
            torch.randn((1, 4, 3), dtype=torch.float32),
            (2, 4, 3),
            torch.float32,
            id="fp32",
        ),
        pytest.param(
            "TOSA-1.1+FP",
            torch.randint(0, 2, (2, 1, 3), dtype=torch.bool),
            torch.randint(0, 2, (1, 4, 3), dtype=torch.bool),
            torch.randint(0, 2, (1, 4, 3), dtype=torch.bool),
            (2, 4, 3),
            torch.bool,
            id="bool",
        ),
    ],
)
def test_tosa_select(
    spec: str,
    condition: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.SELECT.default(
            *_to_fake(mode, condition, input1, input2)
        )

    assert output.dtype == expected_dtype
    assert tuple(output.shape) == expected_shape


@pytest.mark.parametrize("spec", ["TOSA-1.1+INT", "TOSA-1.1+FP"])
def test_tosa_select_registered_for_all_profiles(spec: str) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert exir_ops.backend.tosa.SELECT.default in registered_ops


def test_tosa_select_accepts_bfloat16_with_bf16_extension() -> None:
    condition = torch.randint(0, 2, (2, 3), dtype=torch.bool)
    input1 = torch.randn((2, 3), dtype=torch.bfloat16)
    input2 = torch.randn((2, 3), dtype=torch.bfloat16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+bf16")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.SELECT.default(
            *_to_fake(mode, condition, input1, input2)
        )

    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == tuple(input1.shape)


def test_tosa_select_rejects_non_bool_condition() -> None:
    condition = torch.ones((2, 3), dtype=torch.int32)
    input1 = torch.randn((2, 3), dtype=torch.float32)
    input2 = torch.randn((2, 3), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="requires bool condition"):
            exir_ops.backend.tosa.SELECT.default(
                *_to_fake(mode, condition, input1, input2)
            )


def test_tosa_select_rejects_mismatched_value_dtypes() -> None:
    condition = torch.randint(0, 2, (2, 3), dtype=torch.bool)
    input1 = torch.randn((2, 3), dtype=torch.float32)
    input2 = torch.randn((2, 3), dtype=torch.float16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Expected matching dtypes"):
            exir_ops.backend.tosa.SELECT.default(
                *_to_fake(mode, condition, input1, input2)
            )


def test_tosa_select_rejects_bfloat16_without_bf16_extension() -> None:
    condition = torch.randint(0, 2, (2, 3), dtype=torch.bool)
    input1 = torch.randn((2, 3), dtype=torch.bfloat16)
    input2 = torch.randn((2, 3), dtype=torch.bfloat16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support"):
            exir_ops.backend.tosa.SELECT.default(
                *_to_fake(mode, condition, input1, input2)
            )
