# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import sympy  # type: ignore
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
    return sympy.sympify(getattr(sym.node, "expr", sym.node._expr))


def test_bilinear_resize_rejects_exact_one_sixteenth_downscale():
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT")
    ), FakeTensorMode() as mode:
        with pytest.raises(
            TosaValueError,
            match="Bilinear RESIZE downscale must be strictly greater than 1/16",
        ):
            exir_ops.backend.tosa.RESIZE.default(
                mode.from_tensor(
                    torch.randint(0, 10, (1, 3, 256, 448), dtype=torch.int8)
                ),
                [2, 32, 2, 32],
                [15, 15],
                [-15, -15],
                resize_mode="bilinear",
            )


def test_resize_accepts_symbolic_scale_and_border_values():
    shape_env = ShapeEnv()
    scale_y_n = _make_symint(shape_env, "scale_y_n", hint=2, min=1, max=8)
    border_y = _make_symint(shape_env, "border_y", hint=1, min=0, max=8)

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ), FakeTensorMode(shape_env=shape_env) as mode:
        x = mode.from_tensor(torch.empty(size=(1, 3, 4, 2), dtype=torch.float32))
        output = exir_ops.backend.tosa.RESIZE.default(
            x,
            [scale_y_n, 1, 4, 2],
            [0, 0],
            [border_y, 0],
            resize_mode="nearest",
        )

    assert output.dtype == torch.float32
    assert (output.shape[0], output.shape[-1]) == (1, 2)
    assert isinstance(output.shape[1], torch.SymInt)
    assert output.shape[2] == 7
    # The output height is computed as: (input_height - 1) * scale_y_n + border_y + 1.
    # As the hegiht is a symbolic expression, we check that the expression is correct by
    # comparing it to the expected expression.
    assert str(_expr(output.shape[1])) == "(((border_y + 2*scale_y_n)//1)) + 1"
