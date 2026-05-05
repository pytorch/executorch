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
        "op_name",
        "spec",
        "input1",
        "input2",
        "kwargs",
        "expected_shape",
        "expected_dtype",
    ),
    [
        pytest.param(
            "ADD",
            "TOSA-1.1+FP",
            torch.randn((2, 1, 3), dtype=torch.float32),
            torch.randn((1, 4, 3), dtype=torch.float32),
            {},
            (2, 4, 3),
            torch.float32,
        ),
        pytest.param(
            "ARITHMETIC_RIGHT_SHIFT",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            torch.ones((2, 3), dtype=torch.int8),
            {"round": True},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "BITWISE_AND",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "BITWISE_OR",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "BITWISE_XOR",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "EQUAL",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            {},
            (2, 4, 3),
            torch.bool,
        ),
        pytest.param(
            "GREATER",
            "TOSA-1.1+FP",
            torch.randn((2, 1, 3), dtype=torch.float32),
            torch.randn((1, 4, 3), dtype=torch.float32),
            {},
            (2, 4, 3),
            torch.bool,
        ),
        pytest.param(
            "GREATER_EQUAL",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            {},
            (2, 4, 3),
            torch.bool,
        ),
        pytest.param(
            "INTDIV",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 3), dtype=torch.int32),
            torch.randint(1, 8, (2, 3), dtype=torch.int32),
            {},
            (2, 3),
            torch.int32,
        ),
        pytest.param(
            "LOGICAL_AND",
            "TOSA-1.1+FP",
            torch.tensor([[True, False], [True, True]], dtype=torch.bool),
            torch.tensor([[True, True], [False, True]], dtype=torch.bool),
            {},
            (2, 2),
            torch.bool,
        ),
        pytest.param(
            "LOGICAL_LEFT_SHIFT",
            "TOSA-1.1+INT",
            torch.randint(0, 8, (2, 3), dtype=torch.int8),
            torch.ones((2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "LOGICAL_RIGHT_SHIFT",
            "TOSA-1.1+INT",
            torch.randint(0, 8, (2, 3), dtype=torch.int8),
            torch.ones((2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int8,
        ),
        pytest.param(
            "LOGICAL_OR",
            "TOSA-1.1+FP",
            torch.tensor([[True, False], [True, True]], dtype=torch.bool),
            torch.tensor([[True, True], [False, True]], dtype=torch.bool),
            {},
            (2, 2),
            torch.bool,
        ),
        pytest.param(
            "LOGICAL_XOR",
            "TOSA-1.1+FP",
            torch.tensor([[True, False], [True, True]], dtype=torch.bool),
            torch.tensor([[True, True], [False, True]], dtype=torch.bool),
            {},
            (2, 2),
            torch.bool,
        ),
        pytest.param(
            "MAXIMUM",
            "TOSA-1.1+FP",
            torch.randn((2, 1, 3), dtype=torch.float32),
            torch.randn((1, 4, 3), dtype=torch.float32),
            {},
            (2, 4, 3),
            torch.float32,
        ),
        pytest.param(
            "MINIMUM",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            {},
            (2, 4, 3),
            torch.int32,
        ),
        pytest.param(
            "MUL",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            {},
            (2, 3),
            torch.int32,
        ),
        pytest.param(
            "POW",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            torch.randn((2, 3), dtype=torch.float32),
            {},
            (2, 3),
            torch.float32,
        ),
        pytest.param(
            "SUB",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 1, 3), dtype=torch.int32),
            torch.randint(1, 8, (1, 4, 3), dtype=torch.int32),
            {},
            (2, 4, 3),
            torch.int32,
        ),
    ],
)
def test_tosa_binary_ops(
    op_name: str,
    spec: str,
    input1: torch.Tensor,
    input2: torch.Tensor,
    kwargs: dict[str, object],
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            *_to_fake(mode, input1, input2),
            **kwargs,
        )

    assert output.dtype == expected_dtype
    assert tuple(output.shape) == expected_shape


@pytest.mark.parametrize("op_name", ["LOGICAL_LEFT_SHIFT", "LOGICAL_RIGHT_SHIFT"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_logical_shift_supports_int_dtype_on_fp_profile(
    op_name: str,
    dtype: torch.dtype,
) -> None:
    input1 = torch.randint(0, 8, (2, 3), dtype=dtype)
    input2 = torch.ones((2, 3), dtype=dtype)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            *_to_fake(mode, input1, input2)
        )

    assert output.dtype == dtype
    assert tuple(output.shape) == tuple(input1.shape)


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("TOSA-1.1+INT+int64", id="int_profile"),
        pytest.param("TOSA-1.1+FP+int64", id="fp_profile"),
    ],
)
def test_bitwise_and_supports_int64_extension(spec: str) -> None:
    input1 = torch.randint(0, 8, (2, 3), dtype=torch.int64)
    input2 = torch.ones((2, 3), dtype=torch.int64)

    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.BITWISE_AND.default(
            *_to_fake(mode, input1, input2)
        )

    assert output.dtype == torch.int64
    assert tuple(output.shape) == tuple(input1.shape)


def test_bitwise_and_rejects_int64_without_extension() -> None:
    input1 = torch.randint(0, 8, (2, 3), dtype=torch.int64)
    input2 = torch.ones((2, 3), dtype=torch.int64)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support int64"):
            exir_ops.backend.tosa.BITWISE_AND.default(*_to_fake(mode, input1, input2))


@pytest.mark.parametrize(
    ("op", "spec", "expected"),
    [
        pytest.param(
            exir_ops.backend.tosa.ARITHMETIC_RIGHT_SHIFT.default,
            "TOSA-1.1+INT",
            True,
            id="arithmetic_right_shift_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.ARITHMETIC_RIGHT_SHIFT.default,
            "TOSA-1.1+FP",
            False,
            id="arithmetic_right_shift_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.BITWISE_OR.default,
            "TOSA-1.1+INT",
            True,
            id="bitwise_or_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.BITWISE_OR.default,
            "TOSA-1.1+FP",
            False,
            id="bitwise_or_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.BITWISE_XOR.default,
            "TOSA-1.1+INT",
            True,
            id="bitwise_xor_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.BITWISE_XOR.default,
            "TOSA-1.1+FP",
            False,
            id="bitwise_xor_fp",
        ),
    ],
)
def test_tosa_integer_shift_and_bitwise_ops_registered_for_int_profile_only(
    op,
    spec: str,
    expected: bool,
) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (op in registered_ops) is expected


@pytest.mark.parametrize("spec", ["TOSA-1.1+INT", "TOSA-1.1+FP"])
def test_tosa_bitwise_and_registered_for_all_profiles(spec: str) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert exir_ops.backend.tosa.BITWISE_AND.default in registered_ops


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param("TOSA-1.1+INT", False, id="pow_int"),
        pytest.param("TOSA-1.1+FP", True, id="pow_fp"),
    ],
)
def test_tosa_pow_registered_for_fp_profile_only(spec: str, expected: bool) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (exir_ops.backend.tosa.POW.default in registered_ops) is expected


def test_pow_accepts_bfloat16_with_bf16_extension() -> None:
    input1 = torch.randn((2, 3), dtype=torch.bfloat16)
    input2 = torch.randn((2, 3), dtype=torch.bfloat16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+bf16")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.POW.default(*_to_fake(mode, input1, input2))

    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == tuple(input1.shape)


def test_mul_rejects_non_zero_shift_for_non_int32() -> None:
    input1 = torch.randint(-8, 8, (2, 3), dtype=torch.int8)
    input2 = torch.randint(-8, 8, (2, 3), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="Only int32 MUL supports a non-zero shift value",
        ):
            exir_ops.backend.tosa.MUL.default(
                *_to_fake(mode, input1, input2),
                shift=3,
            )


def test_intdiv_supports_int32_on_fp_profile() -> None:
    input1 = torch.randint(1, 16, (2, 3), dtype=torch.int32)
    input2 = torch.randint(1, 8, (2, 3), dtype=torch.int32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.INTDIV.default(*_to_fake(mode, input1, input2))

    assert output.dtype == torch.int32
    assert tuple(output.shape) == tuple(input1.shape)


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


@pytest.mark.parametrize("op_name", ["MAXIMUM", "MINIMUM"])
def test_extrema_ops_reject_int32_on_fp_profile(op_name: str) -> None:
    input1 = torch.randint(1, 16, (2, 3), dtype=torch.int32)
    input2 = torch.randint(1, 8, (2, 3), dtype=torch.int32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support int32"):
            getattr(exir_ops.backend.tosa, op_name).default(
                *_to_fake(mode, input1, input2)
            )
