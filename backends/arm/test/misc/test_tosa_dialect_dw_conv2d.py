# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.arm.tosa.dialect  # noqa: unused
import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_depthwise_conv2d_tosa_INT():
    sample_inputs = [
        (
            (
                torch.randint(-128, 127, (1, 8, 20, 20), dtype=torch.int8),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randint(-127, 127, (5, 2, 5, 8), dtype=torch.int8),
                torch.randint(-(2**31), 2**31, (16,), dtype=torch.int32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            (1, 16, 20, 20),
            torch.int32,
        ),
        (
            (
                torch.randint(-128, 127, (1, 8, 20, 20), dtype=torch.int8),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randint(-127, 127, (5, 4, 5, 8), dtype=torch.int8),
                None,
                [2, 2],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            (1, 32, 10, 10),
            torch.int32,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert (
                output.dtype == expected_output_type
            ), f"Expected output dtype {expected_output_type} but got {output.dtype}"
            assert (
                tuple(output.shape) == expected_output_shape
            ), f"Expected output shape {expected_output_shape} but got {tuple(output.shape)}"


def test_depthwise_conv2d_invalid_tosa_INT():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    sample_inputs = [
        (
            (
                torch.randn((1, 8, 20, 20), dtype=torch.float32),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randn((5, 2, 5, 8), dtype=torch.float32),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"doesn't support {torch.float32} but found input type {torch.float32}",
        ),
        (
            (
                torch.randint(-128, 127, (1, 8, 20, 20), dtype=torch.int8),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randn((5, 2, 5, 8), dtype=torch.float32),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"only supports {torch.int8} weights for {torch.int8} input but found {torch.float32}",
        ),
        (
            (
                torch.randint(-128, 127, (1, 8, 20, 20), dtype=torch.int8),
                torch.randint(-127, 127, (5, 2, 5, 8), dtype=torch.int8),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"only supports {torch.int32} bias for {torch.int8} input but found {torch.float32}",
        ),
    ]

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        for sample_input, expected_error, expected_error_str in sample_inputs:
            with pytest.raises(expected_error, match=expected_error_str):
                exir_ops.backend.tosa.DEPTHWISE_CONV2D.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )


def test_depthwise_conv2d_tosa_FP():
    sample_inputs = [
        (
            (
                torch.randn((1, 8, 20, 20), dtype=torch.float32),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randn((5, 2, 5, 8), dtype=torch.float32),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            (1, 16, 20, 20),
            torch.float32,
        ),
        (
            (
                torch.randn((1, 8, 20, 20), dtype=torch.float32),
                # weight shape is [H, m_length, W, in_channels], where m_length = out_channels // in_channels
                torch.randn((5, 4, 5, 8), dtype=torch.float32),
                None,
                [2, 2],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            (1, 32, 10, 10),
            torch.float32,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert (
                output.dtype == expected_output_type
            ), f"Expected output dtype {expected_output_type} but got {output.dtype}"
            assert (
                tuple(output.shape) == expected_output_shape
            ), f"Expected output shape {expected_output_shape} but got {tuple(output.shape)}"


def test_depthwise_conv2d_invalid_tosa_FP():

    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")

    sample_inputs = [
        (
            (
                torch.randint(-127, 127, (1, 8, 20, 20), dtype=torch.int8),
                torch.randn((5, 2, 5, 8), dtype=torch.float32),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"doesn't support {torch.int8} but found input type {torch.int8}",
        ),
        (
            (
                torch.randn((1, 8, 20, 20), dtype=torch.float32),
                torch.randn((5, 2, 5, 8), dtype=torch.float16),
                torch.randn((16,), dtype=torch.float32),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"requires weights {torch.float16} to be of the same type as input {torch.float32}",
        ),
        (
            (
                torch.randn((1, 8, 20, 20), dtype=torch.float32),
                torch.randn((5, 2, 5, 8), dtype=torch.float32),
                torch.randn((16,), dtype=torch.float16),
                [1, 1],
                [2, 2, 2, 2],
                [1, 1],
                False,
                [0, 0],
                8,
            ),
            TosaValueError,
            f"requires bias {torch.float16} to be of the same type as input {torch.float32}",
        ),
    ]
    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        for sample_input, expected_error, expected_error_str in sample_inputs:
            with pytest.raises(expected_error, match=expected_error_str):
                exir_ops.backend.tosa.CONV2D.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )
