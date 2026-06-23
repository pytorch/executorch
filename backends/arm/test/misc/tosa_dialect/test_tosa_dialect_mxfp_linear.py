# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops import matmul_t_block_scaled  # noqa: F401
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def test_matmul_t_block_scaled_tosa_fp_mxfp() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    a_data = torch.randn((1, 4, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.empty((1, 4, 1), dtype=torch.float8_e8m0fnu)
    b_data = torch.randn((1, 8, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
    b_scale = torch.empty((1, 8, 1), dtype=torch.float8_e8m0fnu)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        output = exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default(
            mode.from_tensor(a_data),
            mode.from_tensor(a_scale),
            mode.from_tensor(b_data),
            mode.from_tensor(b_scale),
            32,
        )

    assert output.dtype == torch.float32
    assert tuple(output.shape) == (1, 4, 8)


def test_matmul_t_block_scaled_invalid_scale_shape() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    a_data = torch.randn((1, 4, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.empty((1, 4, 2), dtype=torch.float8_e8m0fnu)
    b_data = torch.randn((1, 8, 32), dtype=torch.float32).to(torch.float8_e4m3fn)
    b_scale = torch.empty((1, 8, 1), dtype=torch.float8_e8m0fnu)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="A_scale shape \\(1, 4, 2\\) must match \\(1, 4, 1\\)",
        ):
            exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default(
                mode.from_tensor(a_data),
                mode.from_tensor(a_scale),
                mode.from_tensor(b_data),
                mode.from_tensor(b_scale),
                32,
            )
