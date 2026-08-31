# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.avg_pool2d import (
    validate_avg_pool2d_dtype,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_avg_pool2d_tosa_non_square_kernel_output_shape():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 20, 20, 8), dtype=torch.float32))
        input_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        output_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))

        output = exir_ops.backend.tosa.AVG_POOL2D.default(
            x,
            input_zp,
            output_zp,
            [2, 3],
            [2, 1],
            [1, 1, 0, 0],
            torch.float32,
        )

        assert tuple(output.shape) == (1, 11, 18, 8)


def test_avg_pool2d_adaptive_tosa_INT():
    sample_inputs = [
        (
            (
                torch.randint(-128, 127, (1, 20, 20, 8), dtype=torch.int8),
                torch.zeros((1,), dtype=torch.int8),
                torch.zeros((1,), dtype=torch.int8),
                [3, 3],
                [2, 2],
                [1, 1, 1, 1],
                torch.int32,
            ),
            (1, 10, 10, 8),
            torch.int8,
        ),
        (
            (
                torch.randint(-32768, 32767, (1, 9, 13, 4), dtype=torch.int16),
                torch.zeros((1,), dtype=torch.int16),
                torch.zeros((1,), dtype=torch.int16),
                [2, 4],
                [1, 3],
                [0, 0, 1, 1],
                torch.int32,
            ),
            (1, 8, 4, 4),
            torch.int16,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT+int16")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert output.dtype == expected_output_type
            assert tuple(output.shape) == expected_output_shape


def test_avg_pool2d_adaptive_tosa_FP():
    sample_inputs = [
        (
            (
                torch.randn((1, 20, 20, 8), dtype=torch.float32),
                torch.zeros((1,), dtype=torch.float32),
                torch.zeros((1,), dtype=torch.float32),
                [3, 3],
                [2, 2],
                [1, 1, 1, 1],
                torch.float32,
            ),
            (1, 10, 10, 8),
            torch.float32,
        ),
        (
            (
                torch.randn((1, 9, 13, 4), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.bfloat16),
                torch.zeros((1,), dtype=torch.bfloat16),
                [2, 4],
                [1, 3],
                [0, 0, 1, 1],
                torch.float32,
            ),
            (1, 8, 4, 4),
            torch.bfloat16,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+bf16")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert output.dtype == expected_output_type
            assert tuple(output.shape) == expected_output_shape


def test_avg_pool2d_adaptive_accepts_remainder_one_mapping():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 5, 5, 4), dtype=torch.float32))
        input_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        output_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))

        output = exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default(
            x,
            input_zp,
            output_zp,
            [3, 3],
            [2, 2],
            [0, 0, 0, 0],
            torch.float32,
        )

        assert tuple(output.shape) == (1, 2, 2, 4)


def test_avg_pool2d_adaptive_rejects_irregular_single_op_mapping():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 8, 8, 4), dtype=torch.float32))
        input_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        output_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))

        with pytest.raises(
            TosaValueError, match=r"input_size % output_size in \{0, 1\}"
        ):
            exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default(
                x,
                input_zp,
                output_zp,
                [3, 3],
                [2, 2],
                [0, 0, 0, 0],
                torch.float32,
            )


@pytest.mark.parametrize(
    "spec_str,input_dtype,zero_point_dtype,acc_type",
    [
        ("TOSA-1.0+INT", torch.int8, torch.int8, torch.int32),
        ("TOSA-1.1+INT+int16", torch.int16, torch.int16, torch.int32),
        ("TOSA-1.0+FP", torch.float16, torch.float16, torch.float16),
        ("TOSA-1.0+FP", torch.float16, torch.float16, torch.float32),
        ("TOSA-1.0+FP", torch.float32, torch.float32, torch.float32),
        ("TOSA-1.1+FP+bf16", torch.bfloat16, torch.bfloat16, torch.float32),
    ],
)
def test_validate_avg_pool2d_dtype_accepts_spec_supported_combinations(
    spec_str: str,
    input_dtype: torch.dtype,
    zero_point_dtype: torch.dtype,
    acc_type: torch.dtype,
):
    spec = TosaSpecification.create_from_string(spec_str)
    x = torch.zeros((1, 2, 8, 8), dtype=input_dtype)
    input_zp = torch.zeros((1,), dtype=zero_point_dtype)
    output_zp = torch.zeros((1,), dtype=zero_point_dtype)

    validate_avg_pool2d_dtype(spec, x, input_zp, output_zp, acc_type, op="AVG_POOL2D")


@pytest.mark.parametrize(
    "spec_str,input_dtype,zero_point_dtype,acc_type,match",
    [
        (
            "TOSA-1.0+FP",
            torch.float32,
            torch.int32,
            torch.float32,
            "input zero-point dtype",
        ),
        (
            "TOSA-1.0+FP",
            torch.float32,
            torch.float32,
            torch.int32,
            "accumulator type must be one of",
        ),
        (
            "TOSA-1.0+INT",
            torch.int16,
            torch.int16,
            torch.int32,
            "Unsupported input dtype",
        ),
        (
            "TOSA-1.0+INT",
            torch.uint8,
            torch.uint8,
            torch.int32,
            "Unsupported input dtype",
        ),
    ],
)
def test_validate_avg_pool2d_dtype_rejects_invalid_combinations(
    spec_str: str,
    input_dtype: torch.dtype,
    zero_point_dtype: torch.dtype,
    acc_type: torch.dtype,
    match: str,
):
    spec = TosaSpecification.create_from_string(spec_str)
    x = torch.zeros((1, 2, 8, 8), dtype=input_dtype)
    input_zp = torch.zeros((1,), dtype=zero_point_dtype)
    output_zp = torch.zeros((1,), dtype=zero_point_dtype)

    with pytest.raises(TosaValueError, match=match):
        validate_avg_pool2d_dtype(
            spec,
            x,
            input_zp,
            output_zp,
            acc_type,
            op="AVG_POOL2D",
        )


@pytest.mark.parametrize(
    "op_target",
    [
        exir_ops.backend.tosa.AVG_POOL2D.default,
        exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default,
    ],
)
def test_avg_pool2d_ops_reject_invalid_parameter_lengths(op_target):
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 8, 8, 4), dtype=torch.float32))
        input_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        output_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))

        with pytest.raises(TosaValueError, match="expects kernel of length 2"):
            op_target(
                x,
                input_zp,
                output_zp,
                [2],
                [2, 2],
                [0, 0, 0, 0],
                torch.float32,
            )

        with pytest.raises(TosaValueError, match="stride of length 2"):
            op_target(
                x,
                input_zp,
                output_zp,
                [2, 2],
                [2],
                [0, 0, 0, 0],
                torch.float32,
            )

        with pytest.raises(TosaValueError, match="pad of length 4"):
            op_target(
                x,
                input_zp,
                output_zp,
                [2, 2],
                [2, 2],
                [0, 0, 0],
                torch.float32,
            )


def test_avg_pool2d_adaptive_no_target_requires_tosa_1_1():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 8, 8, 4), dtype=torch.float32))
        input_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        output_zp = mode.from_tensor(torch.zeros((1,), dtype=torch.float32))
        with pytest.raises(TosaValueError, match="support AVG_POOL2D_ADAPTIVE"):
            exir_ops.backend.tosa.AVG_POOL2D_ADAPTIVE.default(
                x,
                input_zp,
                output_zp,
                [2, 2],
                [2, 2],
                [0, 0, 0, 0],
                torch.float32,
            )
