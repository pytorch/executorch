# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "CAST_TO_BLOCK_SCALED(Tensor input, SymInt block_size, ScalarType output_dtype) -> (Tensor, Tensor)",
    [TosaSpecification.create_from_string("TOSA-1.1+FP")],
)
def CAST_TO_BLOCK_SCALED(
    input: torch.Tensor,
    block_size: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tosa_spec = get_context_spec()

    if not tosa_spec.support_float() or not tosa_spec.support_extension("mxfp"):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support MXFP block-scaled casts",
            op="CAST_TO_BLOCK_SCALED",
        )

    if input.dtype not in (torch.float32, torch.bfloat16):
        raise TosaValueError(
            f"Unsupported input dtype {input.dtype} for CAST_TO_BLOCK_SCALED",
            op="CAST_TO_BLOCK_SCALED",
        )
    if input.dtype == torch.bfloat16 and not (
        tosa_spec.support_extension("bf16") or tosa_spec.support_extension("mxfp")
    ):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support bf16",
            op="CAST_TO_BLOCK_SCALED",
        )

    if input.ndim < 1:
        raise TosaValueError(
            "CAST_TO_BLOCK_SCALED requires rank >= 1",
            op="CAST_TO_BLOCK_SCALED",
        )
    if block_size != 32:
        raise TosaValueError(
            f"Unsupported block_size {block_size} (must be 32)",
            op="CAST_TO_BLOCK_SCALED",
        )
    if input.shape[-1] % block_size != 0:
        raise TosaValueError(
            f"Last dim {input.shape[-1]} must be divisible by block_size {block_size}",
            op="CAST_TO_BLOCK_SCALED",
        )

    scale_tensor_dtype = torch.float8_e8m0fnu
    if output_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TosaValueError(
            f"Unsupported block-scaled output dtype {output_dtype}",
            op="CAST_TO_BLOCK_SCALED",
        )
    scale_shape = (*input.shape[:-1], input.shape[-1] // block_size)
    output_data = torch.empty_like(input, dtype=output_dtype)
    output_scale = input.new_empty(scale_shape, dtype=scale_tensor_dtype)
    return output_data, output_scale
