# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

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


def validate_rescale_zero_points(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    input_zp: int,
    output_zp: int,
    input_unsigned: bool,
    output_unsigned: bool,
) -> None:
    if input_unsigned and output_unsigned:
        raise TosaValueError(
            "RESCALE cannot have both input_unsigned and output_unsigned"
        )
    for name, dtype, zero_point, unsigned in (
        ("in", input_dtype, input_zp, input_unsigned),
        ("out", output_dtype, output_zp, output_unsigned),
    ):
        if unsigned:
            valid = (
                0 <= zero_point <= 255
                if dtype in (torch.int8, torch.uint8)
                else dtype == torch.int16 and zero_point in (0, 32768)
            )
            if not valid:
                raise TosaValueError(
                    f"{name}_zp={zero_point} outside valid range for unsigned {dtype}"
                )
        elif dtype == torch.int8 and not -128 <= zero_point <= 127:
            raise TosaValueError(
                f"{name}_zp={zero_point} outside valid range for {dtype}"
            )
        elif dtype in (torch.int16, torch.int32) and zero_point != 0:
            argument = "input" if name == "in" else "output"
            raise TosaValueError(f"TOSA requires {argument}_zp to be zero for {dtype}")


def _validate_rescale_unsigned(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    input_unsigned: bool,
    output_unsigned: bool,
) -> None:
    if input_unsigned and output_unsigned:
        raise TosaValueError(
            "RESCALE cannot have both unsigned flags set", op="RESCALE"
        )
    if input_unsigned and input_dtype not in (torch.int8, torch.uint8, torch.int16):
        raise TosaValueError(
            "Unsigned RESCALE input must be int8 or int16", op="RESCALE"
        )
    if output_unsigned and (
        output_dtype not in (torch.int8, torch.int16)
        or input_dtype not in (torch.int8, torch.uint8, torch.int16)
    ):
        raise TosaValueError(
            "Unsigned RESCALE output requires narrow input and output", op="RESCALE"
        )


@register_tosa_op(
    "RESCALE(Tensor input1, int[] multiplier, int[] shift, int input_zp, int output_zp, *, ScalarType dtype, bool scale32=True, str rounding_mode='SINGLE_ROUND', bool per_channel=False, bool input_unsigned=False, bool output_unsigned=False) -> Tensor",
    INT_SPECS,
)
def RESCALE(
    x: torch.Tensor,
    multiplier: list[int],
    shift: list[int],
    input_zp: int,
    output_zp: int,
    *,
    dtype: torch.dtype,
    scale32: bool = True,
    rounding_mode: str = "SINGLE_ROUND",
    per_channel: bool = False,
    input_unsigned: bool = False,
    output_unsigned: bool = False,
) -> torch.Tensor:
    if not get_context_spec().support_integer():
        raise TosaValueError("RESCALE requires integer support", op="RESCALE")
    if dtype not in (torch.int8, torch.int16, torch.int32):
        raise TosaValueError(f"Unsupported RESCALE output dtype {dtype}", op="RESCALE")
    if len(shift) != len(multiplier) or not multiplier:
        raise TosaValueError(
            "RESCALE shifts must match the multiplier count", op="RESCALE"
        )
    if not per_channel and len(multiplier) != 1:
        raise TosaValueError("Per-tensor RESCALE requires one multiplier", op="RESCALE")
    if per_channel and len(multiplier) != x.shape[-1]:
        raise TosaValueError(
            "Per-channel RESCALE must match the channel dimension", op="RESCALE"
        )
    width = 32 if scale32 else 16
    if any(not 0 <= value < 1 << (width - 1) for value in multiplier):
        raise TosaValueError("RESCALE multiplier exceeds scale width", op="RESCALE")
    if any(not 2 <= value <= 62 for value in shift):
        raise TosaValueError("RESCALE shift must be in [2, 62]", op="RESCALE")
    if rounding_mode not in ("SINGLE_ROUND", "DOUBLE_ROUND", "INEXACT_ROUND"):
        raise TosaValueError(
            f"Invalid RESCALE rounding mode {rounding_mode}", op="RESCALE"
        )
    validate_rescale_zero_points(
        x.dtype, dtype, input_zp, output_zp, input_unsigned, output_unsigned
    )
    _validate_rescale_unsigned(x.dtype, dtype, input_unsigned, output_unsigned)
    return torch.empty_like(x, dtype=dtype)
