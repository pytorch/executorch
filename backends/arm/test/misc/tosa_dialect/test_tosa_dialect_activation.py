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
    ("op_name", "spec", "input_tensor", "args", "kwargs"),
    [
        pytest.param(
            "CLAMP",
            "TOSA-1.1+INT",
            torch.randint(-8, 8, (2, 3, 4), dtype=torch.int8),
            (-3, 3),
            {},
            id="CLAMP",
        ),
        pytest.param(
            "ERF",
            "TOSA-1.1+FP",
            torch.randn((2, 3, 4), dtype=torch.float32),
            (),
            {},
            id="ERF",
        ),
        pytest.param(
            "SIGMOID",
            "TOSA-1.1+FP",
            torch.randn((2, 3, 4), dtype=torch.float32),
            (),
            {},
            id="SIGMOID",
        ),
        pytest.param(
            "TANH",
            "TOSA-1.1+FP",
            torch.randn((2, 3, 4), dtype=torch.float32),
            (),
            {},
            id="TANH",
        ),
    ],
)
def test_tosa_activation_ops(
    op_name: str,
    spec: str,
    input_tensor: torch.Tensor,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string(spec)
    ), FakeTensorMode() as mode:
        output = getattr(exir_ops.backend.tosa, op_name).default(
            *_to_fake(mode, input_tensor, *args),
            **kwargs,
        )

    assert output.dtype == input_tensor.dtype
    assert tuple(output.shape) == tuple(input_tensor.shape)


@pytest.mark.parametrize(
    ("op", "spec", "expected"),
    [
        pytest.param(
            exir_ops.backend.tosa.ERF.default, "TOSA-1.1+INT", False, id="erf_int"
        ),
        pytest.param(
            exir_ops.backend.tosa.SIGMOID.default,
            "TOSA-1.1+INT",
            False,
            id="sigmoid_int",
        ),
        pytest.param(
            exir_ops.backend.tosa.TANH.default, "TOSA-1.1+INT", False, id="tanh_int"
        ),
        pytest.param(
            exir_ops.backend.tosa.ERF.default, "TOSA-1.1+FP", True, id="erf_fp"
        ),
        pytest.param(
            exir_ops.backend.tosa.SIGMOID.default, "TOSA-1.1+FP", True, id="sigmoid_fp"
        ),
        pytest.param(
            exir_ops.backend.tosa.TANH.default, "TOSA-1.1+FP", True, id="tanh_fp"
        ),
    ],
)
def test_tosa_transcendentals_registered_for_fp_profile_only(
    op,
    spec: str,
    expected: bool,
) -> None:
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        registered_ops = get_registered_tosa_ops()

    assert (op in registered_ops) is expected


@pytest.mark.parametrize(
    ("op_name", "input_tensor"),
    [
        pytest.param(
            "ERF",
            torch.randn((2, 3, 4), dtype=torch.bfloat16),
            id="ERF",
        ),
        pytest.param(
            "SIGMOID",
            torch.randn((2, 3, 4), dtype=torch.bfloat16),
            id="SIGMOID",
        ),
        pytest.param(
            "TANH",
            torch.randn((2, 3, 4), dtype=torch.bfloat16),
            id="TANH",
        ),
    ],
)
def test_tosa_transcendentals_accept_bfloat16_with_bf16_extension(
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


def test_clamp_rejects_invalid_range() -> None:
    sample_input = torch.randint(-8, 8, (2, 3, 4), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="max_val must be greater than or equal to min_val",
        ):
            exir_ops.backend.tosa.CLAMP.default(
                mode.from_tensor(sample_input),
                4,
                -4,
            )


@pytest.mark.parametrize(
    ("min_val", "max_val", "match"),
    [
        pytest.param(-1.5, 1.5, "must be an integer", id="non_integral"),
        pytest.param(-200, 200, "must be in \\[-128, 127\\]", id="out_of_range"),
    ],
)
def test_clamp_rejects_invalid_integer_bounds(
    min_val: int | float,
    max_val: int | float,
    match: str,
) -> None:
    sample_input = torch.randint(-8, 8, (2, 3, 4), dtype=torch.int8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match=match):
            exir_ops.backend.tosa.CLAMP.default(
                mode.from_tensor(sample_input),
                min_val,
                max_val,
            )
