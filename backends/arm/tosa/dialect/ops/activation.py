# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import validate_nan_mode
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

FP_SPECS = TosaSpecification.all_versions_for_profile("FP")


def _validate_clamp_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype == torch.int8:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support int8 for {op}",
                op=op,
            )
        return

    if dtype == torch.int16:
        if not (tosa_spec.support_integer() and tosa_spec.support_extension("int16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support int16 for {op}",
                op=op,
            )
        return

    _validate_float_dtype(dtype, op)
    return


def _validate_float_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {dtype} for {op}",
                op=op,
            )
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support bfloat16 for {op}",
                op=op,
            )
        return

    raise TosaValueError(f"Unsupported dtype {dtype} for {op}", op=op)


def _validate_integer_clamp_bounds(
    dtype: torch.dtype,
    min_val,
    max_val,
) -> None:
    if dtype not in (torch.int8, torch.int16):
        return

    dtype_info = torch.iinfo(dtype)
    for name, value in (("min_val", min_val), ("max_val", max_val)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TosaValueError(
                f"{name} must be an integer for {dtype} CLAMP",
                op="CLAMP",
            )
        if value < dtype_info.min or value > dtype_info.max:
            raise TosaValueError(
                f"{name} must be in [{dtype_info.min}, {dtype_info.max}] for {dtype} CLAMP",
                op="CLAMP",
            )


@register_tosa_op(
    'CLAMP(Tensor input, Scalar min_val, Scalar max_val, *, str nan_mode="PROPAGATE") -> Tensor',
    TosaSpecification.all_versions_and_profiles(),
)
def CLAMP(
    input: torch.Tensor,
    min_val,
    max_val,
    *,
    nan_mode: str = "PROPAGATE",
) -> torch.Tensor:
    validate_nan_mode(nan_mode, "CLAMP")
    _validate_clamp_dtype(input.dtype, "CLAMP")
    _validate_integer_clamp_bounds(input.dtype, min_val, max_val)

    if isinstance(min_val, float) and math.isnan(min_val):
        raise TosaValueError("min_val cannot be NaN", op="CLAMP")
    if isinstance(max_val, float) and math.isnan(max_val):
        raise TosaValueError("max_val cannot be NaN", op="CLAMP")
    if min_val > max_val:
        raise TosaValueError(
            "max_val must be greater than or equal to min_val", op="CLAMP"
        )

    return torch.empty_like(input, dtype=input.dtype)


@register_tosa_op(
    "ERF(Tensor input) -> Tensor",
    FP_SPECS,
)
def ERF(input: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input.dtype, "ERF")
    return torch.empty_like(input, dtype=input.dtype)


@register_tosa_op(
    "SIGMOID(Tensor input) -> Tensor",
    FP_SPECS,
)
def SIGMOID(input: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input.dtype, "SIGMOID")
    return torch.empty_like(input, dtype=input.dtype)


@register_tosa_op(
    "TANH(Tensor input) -> Tensor",
    FP_SPECS,
)
def TANH(input: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input.dtype, "TANH")
    return torch.empty_like(input, dtype=input.dtype)
