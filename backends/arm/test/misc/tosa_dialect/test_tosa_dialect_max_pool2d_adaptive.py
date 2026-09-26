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


def test_max_pool2d_adaptive_tosa_INT():
    sample_inputs = [
        (
            (
                torch.randint(-128, 127, (1, 20, 20, 8), dtype=torch.int8),
                [3, 3],
                [2, 2],
                [1, 1, 1, 1],
            ),
            (1, 10, 10, 8),
            torch.int8,
        ),
        (
            (
                torch.randint(-32768, 32767, (1, 9, 13, 4), dtype=torch.int16),
                [2, 4],
                [1, 3],
                [0, 0, 1, 1],
            ),
            (1, 8, 4, 4),
            torch.int16,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT+int16")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert output.dtype == expected_output_type
            assert tuple(output.shape) == expected_output_shape


def test_max_pool2d_adaptive_tosa_FP():
    sample_inputs = [
        (
            (
                torch.randn((1, 20, 20, 8), dtype=torch.float32),
                [3, 3],
                [2, 2],
                [1, 1, 1, 1],
            ),
            (1, 10, 10, 8),
            torch.float32,
        ),
        (
            (
                torch.randn((1, 9, 13, 4), dtype=torch.bfloat16),
                [2, 4],
                [1, 3],
                [0, 0, 1, 1],
            ),
            (1, 8, 4, 4),
            torch.bfloat16,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+bf16")
    ), FakeTensorMode() as mode:
        for sample_input, expected_output_shape, expected_output_type in sample_inputs:
            output = exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )
            assert output.dtype == expected_output_type
            assert tuple(output.shape) == expected_output_shape


def test_max_pool2d_adaptive_accepts_remainder_one_mapping():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 5, 5, 4), dtype=torch.float32))

        output = exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default(
            x,
            [3, 3],
            [2, 2],
            [0, 0, 0, 0],
        )

        assert tuple(output.shape) == (1, 2, 2, 4)


def test_max_pool2d_adaptive_rejects_irregular_single_op_mapping():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn((1, 8, 8, 4), dtype=torch.float32))

        with pytest.raises(
            TosaValueError, match=r"input_size % output_size in \{0, 1\}"
        ):
            exir_ops.backend.tosa.MAX_POOL2D_ADAPTIVE.default(
                x,
                [3, 3],
                [2, 2],
                [0, 0, 0, 0],
            )
