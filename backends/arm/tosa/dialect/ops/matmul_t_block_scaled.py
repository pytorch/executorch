# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise TosaValueError(
            f"block_size must be positive, got {block_size}",
            op="MATMUL_T_BLOCK_SCALED",
        )
    if block_size != 32:
        raise TosaValueError(
            f"Unsupported block_size {block_size}",
            op="MATMUL_T_BLOCK_SCALED",
        )


def _validate_dtypes(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    if A_data.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TosaValueError(
            f"Unsupported A_data dtype {A_data.dtype}",
            op="MATMUL_T_BLOCK_SCALED",
        )
    if B_data.dtype != A_data.dtype:
        raise TosaValueError(
            f"B_data dtype {B_data.dtype} must match A_data dtype {A_data.dtype}",
            op="MATMUL_T_BLOCK_SCALED",
        )
    if A_scale.dtype != torch.float8_e8m0fnu or B_scale.dtype != torch.float8_e8m0fnu:
        raise TosaValueError(
            "Scale tensors must use torch.float8_e8m0fnu",
            op="MATMUL_T_BLOCK_SCALED",
        )


def _validate_shapes(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    block_size: int,
) -> tuple[int, int, int]:
    if A_data.ndim != 3 or A_scale.ndim != 3 or B_data.ndim != 3 or B_scale.ndim != 3:
        raise TosaValueError(
            "MATMUL_T_BLOCK_SCALED expects rank-3 tensors for values and scales",
            op="MATMUL_T_BLOCK_SCALED",
        )

    N, H, C = A_data.shape
    D, W, Cb = B_data.shape
    if C != Cb:
        raise TosaValueError(
            f"A_data last dim {C} must match B_data last dim {Cb}",
            op="MATMUL_T_BLOCK_SCALED",
        )
    if C % block_size != 0:
        raise TosaValueError(
            f"Last dim {C} must be divisible by block_size {block_size}",
            op="MATMUL_T_BLOCK_SCALED",
        )

    expected_a_scale_shape = (N, H, C // block_size)
    expected_b_scale_shape = (D, W, C // block_size)
    if tuple(A_scale.shape) != expected_a_scale_shape:
        raise TosaValueError(
            f"A_scale shape {tuple(A_scale.shape)} must match {expected_a_scale_shape}",
            op="MATMUL_T_BLOCK_SCALED",
        )
    if tuple(B_scale.shape) != expected_b_scale_shape:
        raise TosaValueError(
            f"B_scale shape {tuple(B_scale.shape)} must match {expected_b_scale_shape}",
            op="MATMUL_T_BLOCK_SCALED",
        )

    if D not in (1, N):
        raise TosaValueError(
            f"B_data batch dim {D} must be 1 or match A_data batch dim {N}",
            op="MATMUL_T_BLOCK_SCALED",
        )

    return N, H, W


@register_tosa_op(
    "MATMUL_T_BLOCK_SCALED(Tensor A_data, Tensor A_scale, Tensor B_data, Tensor B_scale, SymInt block_size) -> Tensor",
    [TosaSpecification.create_from_string("TOSA-1.1+FP")],
)
def MATMUL_T_BLOCK_SCALED(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    if not tosa_spec.support_float() or not tosa_spec.support_extension("mxfp"):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support MXFP block-scaled matmul",
            op="MATMUL_T_BLOCK_SCALED",
        )

    _validate_block_size(block_size)
    _validate_dtypes(A_data, A_scale, B_data, B_scale)
    output_shape = _validate_shapes(
        A_data,
        A_scale,
        B_data,
        B_scale,
        block_size,
    )
    return A_data.new_empty(output_shape, dtype=torch.float32)
