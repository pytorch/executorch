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


def test_conv2d_block_scaled_tosa_fp_mxfp() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default(
            mode.from_tensor(torch.randn(1, 8, 8, 32, dtype=torch.float8_e4m3fn)),
            mode.from_tensor(torch.randn(1, 8, 8, 1, dtype=torch.float8_e8m0fnu)),
            mode.from_tensor(torch.randn(4, 3, 3, 32, dtype=torch.float8_e4m3fn)),
            mode.from_tensor(torch.randn(4, 3, 3, 1, dtype=torch.float8_e8m0fnu)),
            mode.from_tensor(torch.randn(4, dtype=torch.float32)),
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            32,
        )

    assert output.dtype == torch.float32
    assert tuple(output.shape) == (1, 8, 8, 4)


def test_conv2d_block_scaled_tosa_fp_mxfp4() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    ), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default(
            mode.from_tensor(torch.zeros(1, 8, 8, 16, dtype=torch.uint8)),
            mode.from_tensor(torch.randn(1, 8, 8, 1, dtype=torch.float8_e8m0fnu)),
            mode.from_tensor(torch.zeros(4, 3, 3, 16, dtype=torch.uint8)),
            mode.from_tensor(torch.randn(4, 3, 3, 1, dtype=torch.float8_e8m0fnu)),
            mode.from_tensor(torch.randn(4, dtype=torch.float32)),
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            32,
        )

    assert output.dtype == torch.float32
    assert tuple(output.shape) == (1, 8, 8, 4)


def test_conv2d_block_scaled_invalid_scale_shape() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="weight_scale shape"):
            exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default(
                mode.from_tensor(torch.randn(1, 8, 8, 32, dtype=torch.float8_e4m3fn)),
                mode.from_tensor(torch.randn(1, 8, 8, 1, dtype=torch.float8_e8m0fnu)),
                mode.from_tensor(torch.randn(4, 3, 3, 32, dtype=torch.float8_e4m3fn)),
                mode.from_tensor(torch.randn(4, 3, 3, 2, dtype=torch.float8_e8m0fnu)),
                mode.from_tensor(torch.randn(4, dtype=torch.float32)),
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                32,
            )


def test_conv2d_block_scaled_invalid_block_size() -> None:
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="Unsupported block_size 16"):
            exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default(
                mode.from_tensor(torch.randn(1, 8, 8, 32, dtype=torch.float8_e4m3fn)),
                mode.from_tensor(torch.randn(1, 8, 8, 2, dtype=torch.float8_e8m0fnu)),
                mode.from_tensor(torch.randn(4, 3, 3, 32, dtype=torch.float8_e4m3fn)),
                mode.from_tensor(torch.randn(4, 3, 3, 2, dtype=torch.float8_e8m0fnu)),
                mode.from_tensor(torch.randn(4, dtype=torch.float32)),
                [1, 1],
                [1, 1, 1, 1],
                [1, 1],
                16,
            )
