# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch

from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)

from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x


def test_rescale_op_tosa_INT():
    sample_inputs = [
        # (data, out_dtype, scale, in_zp, out_zp)
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            [0.2],
            2,
            0,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            [0.2],
            0,
            -128,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int8,
            [0.8],
            10,
            127,
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            exir_ops.backend.tosa.RESCALE.default(
                *tuple(
                    [
                        mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                        for i in sample_input
                    ]
                )
            )


def test_nonzero_zp_for_int32_tosa_INT():

    sample_inputs = [
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            [0.2],
            2,  # Should be 0, expect error
            1,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            [0.2],
            1,
            1,  # Should be 0, expect error
        ),
    ]

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            with pytest.raises(
                ValueError, match="TOSA requires (output|input)_zp to be zero"
            ):
                exir_ops.backend.tosa.RESCALE.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )


def test_zp_outside_range_tosa_INT():

    sample_inputs = [
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int8),
            torch.int32,
            [0.2],
            128,  # Should be <128, expect error
            0,
        ),
        (
            torch.randint(low=0, high=100, size=(4, 4, 4), dtype=torch.int32),
            torch.int8,
            [0.2],
            0,
            -129,  # Should be >-129 expect error
        ),
    ]
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        for sample_input in sample_inputs:
            with pytest.raises(
                Exception, match="(in_zp|out_zp)=-?[0-9]* outside valid range"
            ):
                exir_ops.backend.tosa.RESCALE.default(
                    *tuple(
                        [
                            mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                            for i in sample_input
                        ]
                    )
                )
