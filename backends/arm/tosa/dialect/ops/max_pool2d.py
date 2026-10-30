# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import compute_pool2d_output_shape
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from torch.types import IntLikeType


def validate_max_pool2d_dtype(
    tosa_spec: TosaSpecification,
    x: torch.Tensor,
    op: str,
) -> None:

    # Validate dtype support
    supported_int_types = [torch.int8]
    supported_float_types = [
        torch.float16,
        torch.float32,
    ]
    if tosa_spec.support_extension("bf16"):
        supported_float_types.append(torch.bfloat16)
    if tosa_spec.support_extension("int16"):
        supported_int_types.append(torch.int16)
    if tosa_spec.support_extension("fp8e4m3"):
        supported_float_types.append(torch.float8_e4m3fn)
    if tosa_spec.support_extension("fp8e5m2"):
        supported_float_types.append(torch.float8_e5m2)

    if x.dtype in supported_int_types:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integer pools", op=op
            )
    elif x.dtype in supported_float_types:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float pools", op=op
            )
    else:
        raise TosaValueError(f"Unsupported input dtype {x.dtype} pools", op=op)


@register_tosa_op(
    "MAX_POOL2D(Tensor input, int[2] kernel, int[2] stride, SymInt[4] pad) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def MAX_POOL2D(
    x: torch.Tensor,
    kernel: List[int],
    stride: List[int],
    pad: List[Union[int, torch.SymInt]],
) -> torch.Tensor:
    """Compute output meta for a TOSA MAX_POOL2D operation."""
    tosa_spec = get_context_spec()
    validate_max_pool2d_dtype(tosa_spec, x, op="MAX_POOL2D")
    output_shape = compute_max_pool2d_output_shape(
        x,
        kernel,
        stride,
        pad,
        op="MAX_POOL2D",
    )
    return torch.empty(size=output_shape, dtype=x.dtype)


def compute_max_pool2d_output_shape(
    x: torch.Tensor,
    kernel: List[IntLikeType] | List[int],
    stride: List[IntLikeType] | List[int],
    pad: List[IntLikeType] | List[int],
    op: str = "MAX_POOL2D",
) -> List[IntLikeType]:
    """Compute the output shape for NHWC max-pool."""

    if x.dim() != 4:
        raise TosaValueError(f"{op} requires a 4D tensor, got {x.dim()}D", op=op)

    return compute_pool2d_output_shape(x, kernel, stride, pad)
