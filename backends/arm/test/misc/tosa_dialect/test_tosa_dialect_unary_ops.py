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


@pytest.mark.parametrize(
    ("op_name", "spec", "input_tensor"),
    [
        pytest.param(
            "ABS",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 3), dtype=torch.int32),
            id="ABS",
        ),
        pytest.param(
            "BITWISE_NOT",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int8),
            id="BITWISE_NOT",
        ),
        pytest.param(
            "BITWISE_NOT",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int16),
            id="BITWISE_NOT_INT16",
        ),
        pytest.param(
            "CEIL",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="CEIL",
        ),
        pytest.param(
            "CLZ",
            "TOSA-1.1+INT",
            torch.randint(1, 16, (2, 3), dtype=torch.int32),
            id="CLZ",
        ),
        pytest.param(
            "COS",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="COS",
        ),
        pytest.param(
            "EXP",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="EXP",
        ),
        pytest.param(
            "FLOOR",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="FLOOR",
        ),
        pytest.param(
            "LOG",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32).abs() + 1.0,
            id="LOG",
        ),
        pytest.param(
            "LOGICAL_NOT",
            "TOSA-1.1+FP",
            torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            id="LOGICAL_NOT",
        ),
        pytest.param(
            "NEGATE",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int32),
            id="NEGATE",
        ),
        pytest.param(
            "NEGATE",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3), dtype=torch.int16),
            id="NEGATE_INT16",
        ),
        pytest.param(
            "NEGATE",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="NEGATE_FP32",
        ),
        pytest.param(
            "RECIPROCAL",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32).abs() + 1.0,
            id="RECIPROCAL",
        ),
        pytest.param(
            "RSQRT",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32).abs() + 1.0,
            id="RSQRT",
        ),
        pytest.param(
            "SIN",
            "TOSA-1.1+FP",
            torch.randn((2, 3), dtype=torch.float32),
            id="SIN",
        ),
    ],
)
def test_tosa_unary_ops(
    op_name: str,
    spec: str,
    input_tensor: torch.Tensor,
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            mode.from_tensor(input_tensor)
        )

    assert output.dtype == input_tensor.dtype
    assert tuple(output.shape) == tuple(input_tensor.shape)


@pytest.mark.parametrize(
    ("op", "spec", "expected"),
    [
        pytest.param(
            exir_ops.backend.tosa.BITWISE_NOT.default,
            "TOSA-1.1+INT",
            True,
            id="bitwise_not_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.BITWISE_NOT.default,
            "TOSA-1.1+FP",
            False,
            id="bitwise_not_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.CLZ.default,
            "TOSA-1.1+INT",
            True,
            id="clz_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.CLZ.default,
            "TOSA-1.1+FP",
            False,
            id="clz_fp",
        ),
    ],
)
def test_tosa_integer_unary_ops_registered_for_int_profile_only(
    op,
    spec: str,
    expected: bool,
) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (op in registered_ops) is expected


@pytest.mark.parametrize(
    ("op", "spec", "expected"),
    [
        pytest.param(
            exir_ops.backend.tosa.CEIL.default,
            "TOSA-1.1+INT",
            False,
            id="ceil_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.CEIL.default,
            "TOSA-1.1+FP",
            True,
            id="ceil_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.COS.default,
            "TOSA-1.1+INT",
            False,
            id="cos_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.COS.default,
            "TOSA-1.1+FP",
            True,
            id="cos_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.EXP.default,
            "TOSA-1.1+INT",
            False,
            id="exp_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.EXP.default,
            "TOSA-1.1+FP",
            True,
            id="exp_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.FLOOR.default,
            "TOSA-1.1+INT",
            False,
            id="floor_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.FLOOR.default,
            "TOSA-1.1+FP",
            True,
            id="floor_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.LOG.default,
            "TOSA-1.1+INT",
            False,
            id="log_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.LOG.default,
            "TOSA-1.1+FP",
            True,
            id="log_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.RECIPROCAL.default,
            "TOSA-1.1+INT",
            False,
            id="reciprocal_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.RECIPROCAL.default,
            "TOSA-1.1+FP",
            True,
            id="reciprocal_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.RSQRT.default,
            "TOSA-1.1+INT",
            False,
            id="rsqrt_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.RSQRT.default,
            "TOSA-1.1+FP",
            True,
            id="rsqrt_fp",
        ),
        pytest.param(
            exir_ops.backend.tosa.SIN.default,
            "TOSA-1.1+INT",
            False,
            id="sin_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.SIN.default,
            "TOSA-1.1+FP",
            True,
            id="sin_fp",
        ),
    ],
)
def test_tosa_float_unary_ops_registered_for_fp_profile_only(
    op,
    spec: str,
    expected: bool,
) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (op in registered_ops) is expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param("TOSA-1.1+INT", True, id="negate_int"),
        pytest.param("TOSA-1.1+FP", True, id="negate_fp"),
    ],
)
def test_tosa_negate_registered_for_int_and_fp_profiles(
    spec: str,
    expected: bool,
) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (exir_ops.backend.tosa.NEGATE.default in registered_ops) is expected


@pytest.mark.parametrize(
    ("op_name", "input_tensor"),
    [
        pytest.param(
            "CEIL",
            torch.randn((2, 3), dtype=torch.bfloat16),
            id="CEIL",
        ),
        pytest.param(
            "COS",
            torch.randn((2, 3), dtype=torch.bfloat16),
            id="COS",
        ),
        pytest.param(
            "EXP",
            torch.randn((2, 3), dtype=torch.bfloat16),
            id="EXP",
        ),
        pytest.param(
            "FLOOR",
            torch.randn((2, 3), dtype=torch.bfloat16),
            id="FLOOR",
        ),
        pytest.param(
            "LOG",
            torch.randn((2, 3), dtype=torch.bfloat16).abs() + 1.0,
            id="LOG",
        ),
        pytest.param(
            "NEGATE",
            torch.randn((2, 3), dtype=torch.bfloat16),
            id="NEGATE",
        ),
    ],
)
def test_tosa_float_unary_ops_accept_bfloat16_with_bf16_extension(
    op_name: str,
    input_tensor: torch.Tensor,
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+bf16")
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            mode.from_tensor(input_tensor)
        )

    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == tuple(input_tensor.shape)


def test_negate_rejects_bfloat16_without_bf16_extension() -> None:
    sample_input = torch.randn((2, 3), dtype=torch.bfloat16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support bfloat16"):
            exir_ops.backend.tosa.NEGATE.default(mode.from_tensor(sample_input))


def test_abs_rejects_int8() -> None:
    sample_input = torch.randint(-8, 8, (2, 3), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            exir_ops.backend.tosa.ABS.default(mode.from_tensor(sample_input))


def test_floor_requires_float_profile() -> None:
    sample_input = torch.randn((2, 3), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support"):
            exir_ops.backend.tosa.FLOOR.default(mode.from_tensor(sample_input))


def test_logical_not_rejects_non_bool() -> None:
    sample_input = torch.randint(-8, 8, (2, 3), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="requires bool inputs"):
            exir_ops.backend.tosa.LOGICAL_NOT.default(mode.from_tensor(sample_input))
