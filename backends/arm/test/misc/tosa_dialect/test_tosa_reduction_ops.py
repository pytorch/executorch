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


@pytest.mark.parametrize(
    "op_name,input_tensor,kwargs,expected_shape",
    [
        (
            "REDUCE_ALL",
            torch.tensor([[[True, False], [True, True]]]),
            {"axis": 1},
            (1, 1, 2),
        ),
        (
            "REDUCE_ANY",
            torch.tensor([[[True, False], [False, False]]]),
            {"axis": 2},
            (1, 2, 1),
        ),
        (
            "REDUCE_MAX",
            torch.randint(-8, 8, (2, 3, 4), dtype=torch.int32),
            {"axis": 0, "nan_mode": "PROPAGATE"},
            (1, 3, 4),
        ),
        (
            "REDUCE_MIN",
            torch.randn((2, 3, 4), dtype=torch.float32),
            {"axis": 2, "nan_mode": "IGNORE"},
            (2, 3, 1),
        ),
        (
            "REDUCE_PRODUCT",
            torch.randn((2, 3, 4), dtype=torch.float32),
            {"axis": 1},
            (2, 1, 4),
        ),
        (
            "REDUCE_SUM",
            torch.randint(-8, 8, (2, 3, 4), dtype=torch.int32),
            {"axis": 1},
            (2, 1, 4),
        ),
    ],
)
def test_reduction_ops(op_name, input_tensor, kwargs, expected_shape):
    spec = (
        "TOSA-1.1+FP+bf16+int64"
        if input_tensor.dtype.is_floating_point
        else "TOSA-1.1+INT+int16+int64"
    )
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        op = getattr(exir_ops.backend.tosa, op_name).default
        output = op(mode.from_tensor(input_tensor), **kwargs)

    assert output.dtype == input_tensor.dtype
    assert tuple(output.shape) == expected_shape


def test_reduce_all_rejects_non_bool():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="requires bool input"):
            exir_ops.backend.tosa.REDUCE_ALL.default(
                mode.from_tensor(torch.ones((2, 2), dtype=torch.int32)), axis=1
            )


def test_reduce_product_rejects_integer_input():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="floating-point input"):
            exir_ops.backend.tosa.REDUCE_PRODUCT.default(
                mode.from_tensor(torch.ones((2, 2), dtype=torch.int32)), axis=1
            )


@pytest.mark.parametrize(
    "op_name,dtype", [("REDUCE_MAX", torch.float32), ("REDUCE_MIN", torch.int32)]
)
def test_reduce_minmax_default_nan_mode(op_name: str, dtype: torch.dtype):
    spec = "TOSA-1.1+FP" if dtype.is_floating_point else "TOSA-1.1+INT"
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        op = getattr(exir_ops.backend.tosa, op_name).default
        output = op(mode.from_tensor(torch.ones((2, 2), dtype=dtype)), axis=1)

    assert output.dtype == dtype
    assert tuple(output.shape) == (2, 1)


@pytest.mark.parametrize("op_name", ["REDUCE_MAX", "REDUCE_MIN"])
def test_reduce_minmax_rejects_invalid_nan_mode(op_name: str):
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        op = getattr(exir_ops.backend.tosa, op_name).default
        with pytest.raises(TosaValueError, match="Invalid nan_mode"):
            op(
                mode.from_tensor(torch.ones((2, 2), dtype=torch.float32)),
                axis=1,
                nan_mode="INVALID_MODE",
            )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16])
def test_reduce_sum_rejects_narrow_integer_inputs(dtype: torch.dtype):
    spec = "TOSA-1.1+INT+int16" if dtype == torch.int16 else "TOSA-1.1+INT"
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            exir_ops.backend.tosa.REDUCE_SUM.default(
                mode.from_tensor(torch.ones((2, 2), dtype=dtype)),
                axis=1,
            )
