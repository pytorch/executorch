# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.ao_ext.mxfp import mxfp_dtype_to_str
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops import cast_to_block_scaled  # noqa: F401
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2


def test_cast_to_block_scaled_requires_mxfp_extension() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP")
    sample_input = torch.randn((2, 32), dtype=torch.float32)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="doesn't support MXFP block-scaled casts",
        ):
            exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
                mode.from_tensor(sample_input),
                32,
                output_dtype=mxfp_dtype_to_str(torch.float8_e4m3fn),
            )


def test_cast_to_block_scaled_tosa_fp_mxfp() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    sample_input = torch.randn((2, 32), dtype=torch.float32)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        output_data, output_scale = exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
            mode.from_tensor(sample_input),
            32,
            output_dtype=mxfp_dtype_to_str(torch.float8_e4m3fn),
        )

    assert output_data.dtype == torch.float8_e4m3fn
    assert tuple(output_data.shape) == (2, 32)
    assert output_scale.dtype == torch.float8_e8m0fnu
    assert tuple(output_scale.shape) == (2, 1)


def test_cast_to_block_scaled_tosa_fp_mxfp4() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    sample_input = torch.randn((2, 32), dtype=torch.float32)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        output_data, output_scale = exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
            mode.from_tensor(sample_input),
            32,
            output_dtype=mxfp_dtype_to_str(torch.float4_e2m1fn_x2),
        )

    assert output_data.dtype == torch.uint8
    assert tuple(output_data.shape) == (2, 16)
    assert output_scale.dtype == torch.float8_e8m0fnu
    assert tuple(output_scale.shape) == (2, 1)


def _test_cast_to_block_scaled_tosa_fp_mxfp6(dtype: str) -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")
    sample_input = torch.randn((2, 32), dtype=torch.float32)

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        output_data, output_scale = exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
            mode.from_tensor(sample_input),
            32,
            output_dtype=mxfp_dtype_to_str(dtype),
        )

    assert output_data.dtype == torch.uint8
    assert tuple(output_data.shape) == (2, 32)
    assert output_scale.dtype == torch.float8_e8m0fnu
    assert tuple(output_scale.shape) == (2, 1)


def test_cast_to_block_scaled_tosa_fp_mxfp6e2m3() -> None:
    _test_cast_to_block_scaled_tosa_fp_mxfp6(DTYPE_FP6_E2M3)


def test_cast_to_block_scaled_tosa_fp_mxfp6e3m2() -> None:
    _test_cast_to_block_scaled_tosa_fp_mxfp6(DTYPE_FP6_E3M2)


def test_cast_to_block_scaled_invalid_shape() -> None:
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="Last dim 30 must be divisible by block_size 32",
        ):
            exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
                mode.from_tensor(torch.randn((2, 30), dtype=torch.float32)),
                32,
                output_dtype=mxfp_dtype_to_str(torch.float8_e4m3fn),
            )
