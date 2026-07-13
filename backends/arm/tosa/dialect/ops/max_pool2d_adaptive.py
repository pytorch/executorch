# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.max_pool2d import (
    compute_max_pool2d_output_shape,
    validate_max_pool2d_dtype,
)
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
    TosaSpecification,
)
from torch.types import IntLikeType


def _is_directly_representable(
    input_size: IntLikeType, output_size: IntLikeType
) -> bool:
    remainder = sympy.Mod(input_size, output_size)
    if isinstance(remainder, torch.SymInt):
        shape_env = get_context_shape_env()
        try:
            remainder_range = shape_env.bound_sympy(remainder.node.expr)
        except Exception:
            return False

        return remainder_range.is_singleton() and int(remainder_range.upper) in (0, 1)

    return remainder in (0, 1)


@register_tosa_op(
    "MAX_POOL2D_ADAPTIVE(Tensor input, SymInt[2] kernel, SymInt[2] stride, SymInt[4] pad) -> Tensor",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MAX_POOL2D_ADAPTIVE(
    x: torch.Tensor,
    kernel: list[IntLikeType],
    stride: list[IntLikeType],
    pad: list[IntLikeType],
) -> torch.Tensor:
    """Fake MAX_POOL2D_ADAPTIVE stub: computes output shape and returns empty tensor."""

    tosa_spec = get_context_spec()
    validate_max_pool2d_dtype(tosa_spec, x, op="MAX_POOL2D_ADAPTIVE")
    output_shape = compute_max_pool2d_output_shape(
        x,
        kernel,
        stride,
        pad,
        op="MAX_POOL2D_ADAPTIVE",
    )

    input_h, input_w = x.shape[1], x.shape[2]
    output_h, output_w = output_shape[1], output_shape[2]
    if not _is_directly_representable(
        input_h, output_h
    ) or not _is_directly_representable(input_w, output_w):
        raise TosaValueError(
            "MAX_POOL2D_ADAPTIVE requires input_size % output_size in {0, 1}",
            op="MAX_POOL2D_ADAPTIVE",
        )

    return torch.empty(size=output_shape, dtype=x.dtype)
