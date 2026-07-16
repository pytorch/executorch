# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List

import torch

from executorch.backends.arm.ao_ext.mxfp import mxfp_str_to_dtype
from executorch.backends.arm.tosa.cast_support import supported_tosa_casts
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2

INT_SPECS = TosaSpecification.all_versions_for_profile("INT")
FP_SPECS = TosaSpecification.all_versions_for_profile("FP")
DUAL_PROFILE_SPECS = [*INT_SPECS, *FP_SPECS]


def _supported_casts() -> set[tuple[torch.dtype, torch.dtype]]:
    return supported_tosa_casts(get_context_spec())


@register_tosa_op(
    "CAST(Tensor input, ScalarType dtype) -> Tensor",
    DUAL_PROFILE_SPECS,
)
def CAST(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if (input.dtype, dtype) not in _supported_casts():
        raise TosaValueError(
            f"Unsupported CAST from {input.dtype} to {dtype} for TOSA spec {get_context_spec()}",
            op="CAST",
        )
    return torch.empty_like(input, dtype=dtype)


@register_tosa_op(
    "CAST_TO_BLOCK_SCALED(Tensor input, SymInt block_size, str output_dtype) -> (Tensor, Tensor)",
    [TosaSpecification.create_from_string("TOSA-1.1+FP")],
)
def CAST_TO_BLOCK_SCALED(
    input: torch.Tensor,
    block_size: int,
    output_dtype: str,
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
    elem_dtype = mxfp_str_to_dtype(output_dtype)
    if elem_dtype not in (
        torch.float4_e2m1fn_x2,
        DTYPE_FP6_E2M3,
        DTYPE_FP6_E3M2,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise TosaValueError(
            f"Unsupported block-scaled output dtype {output_dtype}",
            op="CAST_TO_BLOCK_SCALED",
        )
    scale_shape = (*input.shape[:-1], input.shape[-1] // block_size)
    if elem_dtype == torch.float4_e2m1fn_x2:
        output_shape = (*input.shape[:-1], input.shape[-1] // 2)
        output_data = input.new_empty(output_shape, dtype=torch.uint8)
    elif elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
        output_data = input.new_empty(input.shape, dtype=torch.uint8)
    else:
        output_data = torch.empty_like(input, dtype=cast(torch.dtype, elem_dtype))
    output_scale = input.new_empty(scale_shape, dtype=scale_tensor_dtype)
    return output_data, output_scale


@register_tosa_op(
    "RESCALE(Tensor input1, ScalarType dtype, float[] scale, int in_zp, int out_zp, *, bool input_unsigned=False, bool output_unsigned=False) -> Tensor",
    INT_SPECS,
)
def RESCALE(  # noqa: C901
    x: torch.Tensor,
    dtype: torch.dtype,
    scales: List[float],
    in_zp: int,
    out_zp: int,
    *,
    input_unsigned: bool = False,
    output_unsigned: bool = False,
) -> torch.Tensor:
    tosa_spec = get_context_spec()
    if not tosa_spec.support_integer():
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support integers", op="RESCALE"
        )

    if dtype not in (torch.int32, torch.int8, torch.int16):
        raise TosaValueError(
            f"tosa::rescale currently only supports int32, int16 and int8, not {dtype}",
            op="RESCALE",
        )
    if input_unsigned and output_unsigned:
        raise TosaValueError(
            "TOSA requires input_unsigned and output_unsigned not both be true.",
            op="RESCALE",
        )

    if input_unsigned:
        if x.dtype not in (torch.int8, torch.int16, torch.uint8):
            raise TosaValueError(
                f"input_unsigned requires int8/int16/uint8 input dtype, got {x.dtype}.",
                op="RESCALE",
            )
        if x.dtype == torch.int32:
            raise TosaValueError(
                "TOSA forbids input_unsigned for int32 inputs.", op="RESCALE"
            )
        if x.dtype == torch.int16:
            if in_zp not in (0, 32768):
                raise TosaValueError(
                    f"{in_zp=} outside valid range (0,32768) for uint16.",
                    op="RESCALE",
                )
        else:
            if not 0 <= in_zp <= 255:
                raise TosaValueError(
                    f"{in_zp=} outside valid range (0,255) for uint8.",
                    op="RESCALE",
                )
    else:
        if x.dtype in (torch.int32, torch.int16) and in_zp != 0:
            raise TosaValueError(
                f"TOSA requires input_zp to be zero when the input dtype is {x.dtype}.",
                op="RESCALE",
            )
        if x.dtype == torch.int8 and not -128 <= in_zp <= 127:
            raise TosaValueError(
                f"{in_zp=} outside valid range (-128,127) for int8.",
                op="RESCALE",
            )

    if output_unsigned:
        if dtype not in (torch.int8, torch.int16):
            raise TosaValueError(
                f"output_unsigned requires int8/int16 output dtype, got {dtype}.",
                op="RESCALE",
            )
        if dtype == torch.int32:
            raise TosaValueError(
                "TOSA forbids output_unsigned for int32 outputs.", op="RESCALE"
            )
        if dtype == torch.int16:
            if out_zp not in (0, 32768):
                raise TosaValueError(
                    f"{out_zp=} outside valid range (0,32768) for uint16.",
                    op="RESCALE",
                )
        else:
            if not 0 <= out_zp <= 255:
                raise TosaValueError(
                    f"{out_zp=} outside valid range (0,255) for uint8.",
                    op="RESCALE",
                )
    else:
        if dtype in (torch.int32, torch.int16) and out_zp != 0:
            raise TosaValueError(
                f"TOSA requires output_zp to be zero when the output dtype is {dtype}.",
                op="RESCALE",
            )
        if dtype == torch.int8 and not -128 <= out_zp <= 127:
            raise TosaValueError(
                f"{out_zp=} outside valid range (-128,127) for int8.",
                op="RESCALE",
            )

    return torch.empty_like(x, dtype=dtype)
