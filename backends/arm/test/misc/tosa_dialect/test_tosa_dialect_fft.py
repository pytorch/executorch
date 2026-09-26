# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


def _expr(sym: torch.SymInt) -> sympy.Expr:
    return sympy.sympify(str(sym.node._expr))


def test_fft2d_tosa_fp_fft() -> None:
    input_real = torch.randn((2, 8, 16), dtype=torch.float32)
    input_imag = torch.randn((2, 8, 16), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+fft")
    ), FakeTensorMode() as mode:
        output_real, output_imag = exir_ops.backend.tosa.FFT2D.default(
            mode.from_tensor(input_real),
            mode.from_tensor(input_imag),
        )

    assert output_real.dtype == torch.float32
    assert output_imag.dtype == torch.float32
    assert tuple(output_real.shape) == (2, 8, 16)
    assert tuple(output_imag.shape) == (2, 8, 16)


def test_fft2d_accepts_matching_symbolic_shape() -> None:
    shape_env = ShapeEnv()
    width = _make_symint(shape_env, "w", hint=16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+fft"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        input_real = torch.empty((2, 8, width), dtype=torch.float32)
        input_imag = torch.empty((2, 8, width), dtype=torch.float32)
        output_real, output_imag = exir_ops.backend.tosa.FFT2D.default(
            mode.from_tensor(input_real),
            mode.from_tensor(input_imag),
        )

    assert isinstance(output_real.shape[2], torch.SymInt)
    assert isinstance(output_imag.shape[2], torch.SymInt)
    assert sympy.simplify(_expr(output_real.shape[2]) - sympy.Symbol("w")) == 0
    assert sympy.simplify(_expr(output_imag.shape[2]) - sympy.Symbol("w")) == 0


def test_rfft2d_tosa_fp_fft() -> None:
    input_real = torch.randn((2, 8, 16), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+fft")
    ), FakeTensorMode() as mode:
        output_real, output_imag = exir_ops.backend.tosa.RFFT2D.default(
            mode.from_tensor(input_real),
        )

    assert output_real.dtype == torch.float32
    assert output_imag.dtype == torch.float32
    assert tuple(output_real.shape) == (2, 8, 9)
    assert tuple(output_imag.shape) == (2, 8, 9)


def test_fft_requires_extension() -> None:
    input_real = torch.randn((2, 8, 16), dtype=torch.float32)
    input_imag = torch.randn((2, 8, 16), dtype=torch.float32)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP")
    ), FakeTensorMode() as mode:
        with pytest.raises(TosaValueError, match="doesn't support FFT2D"):
            exir_ops.backend.tosa.FFT2D.default(
                mode.from_tensor(input_real),
                mode.from_tensor(input_imag),
            )


def test_rfft2d_preserves_symbolic_width() -> None:
    shape_env = ShapeEnv()
    width = _make_symint(shape_env, "w", hint=16)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+fft"),
        shape_env,
    ), FakeTensorMode(shape_env=shape_env) as mode:
        input_real = torch.empty((2, 8, width), dtype=torch.float32)
        output_real, output_imag = exir_ops.backend.tosa.RFFT2D.default(
            mode.from_tensor(input_real)
        )

    expected = sympy.floor(sympy.Symbol("w") / 2) + sympy.Integer(1)
    assert isinstance(output_real.shape[2], torch.SymInt)
    assert isinstance(output_imag.shape[2], torch.SymInt)
    assert sympy.simplify(_expr(output_real.shape[2]) - expected) == 0
    assert sympy.simplify(_expr(output_imag.shape[2]) - expected) == 0
