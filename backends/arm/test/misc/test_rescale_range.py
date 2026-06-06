# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

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


def test_unsigned_zp_range_tosa_INT_valid():
    # Validate unsigned zero-point ranges via explicit unsigned semantics.
    # First case: uint8 input (input_unsigned=True) uses in_zp in [0,255].
    # Second case: signed int8 input but unsigned output semantics (output_unsigned=True)
    # allow out_zp in [0,255].
    sample_inputs = [
        # (data, out_dtype, scale, in_zp, out_zp, input_unsigned, output_unsigned)
        (
            torch.randint(low=0, high=255, size=(4, 4, 4), dtype=torch.uint8),
            torch.int8,
            [0.5],
            255,
            0,
            True,
            False,
        ),
        (
            torch.randint(low=-128, high=127, size=(4, 4, 4), dtype=torch.int8),
            torch.int8,
            [0.5],
            0,
            255,
            False,
            True,
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
                        for i in sample_input[:5]
                    ]
                ),
                input_unsigned=sample_input[5],
                output_unsigned=sample_input[6],
            )


def test_unsigned_zp_range_tosa_INT_invalid():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="(in_zp|input_zp).*range"):
            exir_ops.backend.tosa.RESCALE.default(
                mode.from_tensor(
                    torch.randint(low=0, high=255, size=(4, 4, 4), dtype=torch.uint8)
                ),
                torch.int8,
                [0.5],
                256,
                0,
                input_unsigned=True,
                output_unsigned=False,
            )
        with pytest.raises(TosaValueError, match="(out_zp|output_zp).*range"):
            exir_ops.backend.tosa.RESCALE.default(
                mode.from_tensor(
                    torch.randint(low=0, high=255, size=(4, 4, 4), dtype=torch.uint8)
                ),
                torch.int8,
                [0.5],
                0,
                256,
                input_unsigned=False,
                output_unsigned=True,
            )
