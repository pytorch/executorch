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


def _fake_tensor(dtype: torch.dtype, mode: FakeTensorMode) -> torch.Tensor:
    return mode.from_tensor(torch.empty((2, 3), dtype=dtype))


_DATA_LAYOUT_OPS = [
    pytest.param(
        lambda x: exir_ops.backend.tosa.CONCAT.default([x, x], axis=0),
        (4, 3),
        id="concat",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.PAD.default(x, [1, 2, 3, 4], value=0),
        (5, 10),
        id="pad",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.RESHAPE.default(x, [3, 2]),
        (3, 2),
        id="reshape",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.REVERSE.default(x, axis=0),
        (2, 3),
        id="reverse",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.SLICE.default(x, [0, 1], [2, 2]),
        (2, 2),
        id="slice",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.TILE.default(x, [1, 2]),
        (2, 6),
        id="tile",
    ),
    pytest.param(
        lambda x: exir_ops.backend.tosa.TRANSPOSE.default(x, [1, 0]),
        (3, 2),
        id="transpose",
    ),
]

_POSITIVE_DTYPES = [
    pytest.param("TOSA-1.1+FP", torch.float32, id="fp32"),
    pytest.param("TOSA-1.1+INT", torch.int32, id="int32"),
    pytest.param("TOSA-1.1+FP", torch.bool, id="bool"),
    pytest.param("TOSA-1.1+INT+int64", torch.int64, id="int64"),
    pytest.param("TOSA-1.1+FP+bf16", torch.bfloat16, id="bf16"),
    pytest.param("TOSA-1.1+FP+fp8e4m3", torch.float8_e4m3fn, id="fp8e4m3"),
    pytest.param("TOSA-1.1+FP+fp8e5m2", torch.float8_e5m2, id="fp8e5m2"),
]


@pytest.mark.parametrize("spec,dtype", _POSITIVE_DTYPES)
@pytest.mark.parametrize("op,expected_shape", _DATA_LAYOUT_OPS)
def test_data_layout_ops_positive(op, expected_shape, spec, dtype) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = op(_fake_tensor(dtype, mode))

    assert output.dtype == dtype
    assert tuple(output.shape) == expected_shape


@pytest.mark.parametrize(
    "op,error_match",
    [
        pytest.param(
            lambda x: exir_ops.backend.tosa.CONCAT.default([x, x], axis=2),
            "out of range",
            id="concat",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.PAD.default(x, [0, -1, 0, 0], value=0),
            "non-negative",
            id="pad",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.RESHAPE.default(x, [-2, -3]),
            "Negative dimension",
            id="reshape",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.REVERSE.default(x, axis=2),
            "out of range",
            id="reverse",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.SLICE.default(x, [0, 0], [2, 0]),
            r"Expected start \+ size",
            id="slice",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.TILE.default(x, [0, 1]),
            "TILE multiples must be positive",
            id="tile",
        ),
        pytest.param(
            lambda x: exir_ops.backend.tosa.TRANSPOSE.default(x, [0, 0]),
            "Invalid permutation",
            id="transpose",
        ),
    ],
)
def test_data_layout_ops_reject_invalid_arguments(op, error_match) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match=error_match):
            op(_fake_tensor(torch.float32, mode))


@pytest.mark.parametrize("op,expected_shape", _DATA_LAYOUT_OPS)
def test_data_layout_ops_reject_int64_without_extension(op, expected_shape) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            op(_fake_tensor(torch.int64, mode))


def test_int16_data_layout_dtype_support_follows_tosa_spec() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        x = _fake_tensor(torch.int16, mode)

        assert exir_ops.backend.tosa.RESHAPE.default(x, [3, 2]).dtype == torch.int16
        assert exir_ops.backend.tosa.REVERSE.default(x, axis=0).dtype == torch.int16
        assert exir_ops.backend.tosa.TILE.default(x, [1, 1]).dtype == torch.int16

        with pytest.raises(TosaValueError, match="Unsupported dtype"):
            exir_ops.backend.tosa.CONCAT.default([x, x], axis=0)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
    ), FakeTensorMode() as mode:
        x = _fake_tensor(torch.int16, mode)
        assert exir_ops.backend.tosa.CONCAT.default([x, x], axis=0).dtype == torch.int16


def test_pad_rejects_wrong_padding_length() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Padding length"):
            exir_ops.backend.tosa.PAD.default(
                mode.from_tensor(torch.randn((2, 3), dtype=torch.float32)),
                [1, 2],
                value=0.0,
            )


def test_reshape_rejects_size_change():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="same number of elements"):
            exir_ops.backend.tosa.RESHAPE.default(
                mode.from_tensor(torch.randn((2, 3), dtype=torch.float32)),
                [5],
            )
