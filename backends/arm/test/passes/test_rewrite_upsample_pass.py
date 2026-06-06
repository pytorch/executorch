# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import sympy  # type: ignore
import torch
from executorch.backends.arm._passes.rewrite_upsample import RewriteUpsamplePass
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


def test_get_resize_parameters_1d_supports_symbolic_shapes_with_constant_ratio():
    shape_env = ShapeEnv()
    input_size = _make_symint(shape_env, "input_size", hint=4)
    output_size = input_size * 2

    scale_n, scale_d, offset, border = RewriteUpsamplePass.get_resize_parameters_1d(
        input_size, output_size, align_corners=False
    )

    assert (scale_n, scale_d, offset, border) == (4, 2, -1, 1)


def test_get_resize_parameters_1d_rejects_non_constant_symbolic_ratio():
    shape_env = ShapeEnv()
    input_size = _make_symint(shape_env, "input_size", hint=4)
    output_size = input_size + 1

    with pytest.raises(RuntimeError, match="constant ratio"):
        RewriteUpsamplePass.get_resize_parameters_1d(
            input_size, output_size, align_corners=False
        )
