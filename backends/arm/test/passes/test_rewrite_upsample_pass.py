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


def test_get_resize_parameters_1d_nearest_matches_pytorch_scale_factor():
    # PyTorch nearest computes input_index = floor(out_index / scale).
    # TOSA nearest applies a half-scale sampling bias, so the Arm lowering uses
    # a negative half-scale offset to cancel that bias:
    #
    #   out:    0 1 2 3 4 5 6 7 8
    #   PyTorch 0 0 1 2 2 3 4 4 5     (input=6, output=9)
    #   TOSA   floor((out * scale_d + offset + scale_n/2) / scale_n)
    scale_n, scale_d, offset, border = RewriteUpsamplePass.get_resize_parameters_1d(
        6, 9, align_corners=False, resize_mode="nearest"
    )

    assert (scale_n, scale_d, offset, border) == (6, 4, -3, -1)


def test_get_resize_parameters_1d_nearest_uses_explicit_scale_factor():
    # Output size is rounded down from 4 * 1.6 to 6. PyTorch still samples with
    # the explicit 1.6 scale, not the 6 / 4 ratio implied by the tensor sizes.
    actual = RewriteUpsamplePass.get_resize_parameters_1d(
        4,
        6,
        align_corners=False,
        resize_mode="nearest",
        scale_factor=1.6,
    )

    assert actual == (16, 10, -8, -6)

    # Quantization can store the same factor as a float32 value. It must map to
    # the same compact TOSA rational instead of a very large fraction.
    float32_scale = float(torch.tensor(1.6, dtype=torch.float32))
    actual = RewriteUpsamplePass.get_resize_parameters_1d(
        4,
        6,
        align_corners=False,
        resize_mode="nearest",
        scale_factor=float32_scale,
    )

    assert actual == (16, 10, -8, -6)


@pytest.mark.parametrize(
    "output_size, expected",
    [
        (6, (4, 2, -1, 1)),
        (12, (8, 2, -3, 3)),
        (24, (16, 2, -7, 7)),
    ],
)
def test_get_resize_parameters_1d_nearest_preserves_integer_upscale_encoding(
    output_size, expected
):
    # For exact integer upscales, the half-pixel and PyTorch-nearest encodings
    # select the same input pixels. Keep the half-pixel form because Vela can
    # lower it to the Ethos-U55 2x, 4x, and 8x nearest-neighbor implementation.
    actual = RewriteUpsamplePass.get_resize_parameters_1d(
        3, output_size, align_corners=False, resize_mode="nearest"
    )

    assert actual == expected


def test_get_resize_parameters_1d_rejects_non_constant_symbolic_ratio():
    shape_env = ShapeEnv()
    input_size = _make_symint(shape_env, "input_size", hint=4)
    output_size = input_size + 1

    with pytest.raises(RuntimeError, match="constant ratio"):
        RewriteUpsamplePass.get_resize_parameters_1d(
            input_size, output_size, align_corners=False
        )
