# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

FP_SPECS = TosaSpecification.all_versions_for_profile("FP")
INT_SPECS = TosaSpecification.all_versions_for_profile("INT")
DUAL_PROFILE_SPECS = [*INT_SPECS, *FP_SPECS]


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


def _validate_integer_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype in {torch.int8, torch.int16, torch.int32}:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {dtype} for {op}",
                op=op,
            )
        return

    raise TosaValueError(f"Unsupported dtype {dtype} for {op}", op=op)


def _validate_abs_dtype(dtype: torch.dtype) -> None:
    tosa_spec = get_context_spec()

    if dtype == torch.int32:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support int32 for ABS",
                op="ABS",
            )
        return

    if dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {dtype} for ABS",
                op="ABS",
            )
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support bfloat16 for ABS",
                op="ABS",
            )
        return

    raise TosaValueError(f"Unsupported dtype {dtype} for ABS", op="ABS")


def _validate_clz_dtype(dtype: torch.dtype) -> None:
    tosa_spec = get_context_spec()

    if dtype != torch.int32:
        raise TosaValueError(f"CLZ requires int32 inputs but got {dtype}", op="CLZ")
    if not tosa_spec.support_integer():
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support int32 for CLZ",
            op="CLZ",
        )


def _validate_bool_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype != torch.bool:
        raise TosaValueError(f"{op} requires bool inputs but got {dtype}", op=op)


def _validate_negate_dtype(dtype: torch.dtype) -> None:
    if dtype in (torch.int8, torch.int16, torch.int32):
        _validate_integer_dtype(dtype, "NEGATE")
        return

    _validate_float_dtype(dtype, "NEGATE")


@register_fake_tosa_op(
    "ABS(Tensor input1) -> Tensor",
    DUAL_PROFILE_SPECS,
)
def ABS(input1: torch.Tensor) -> torch.Tensor:
    _validate_abs_dtype(input1.dtype)
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "BITWISE_NOT(Tensor input1) -> Tensor",
    INT_SPECS,
)
def BITWISE_NOT(input1: torch.Tensor) -> torch.Tensor:
    _validate_integer_dtype(input1.dtype, "BITWISE_NOT")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "CEIL(Tensor input1) -> Tensor",
    FP_SPECS,
)
def CEIL(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "CEIL")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "CLZ(Tensor input1) -> Tensor",
    INT_SPECS,
)
def CLZ(input1: torch.Tensor) -> torch.Tensor:
    _validate_clz_dtype(input1.dtype)
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "COS(Tensor input1) -> Tensor",
    FP_SPECS,
)
def COS(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "COS")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "EXP(Tensor input1) -> Tensor",
    FP_SPECS,
)
def EXP(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "EXP")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "FLOOR(Tensor input1) -> Tensor",
    FP_SPECS,
)
def FLOOR(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "FLOOR")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "LOG(Tensor input1) -> Tensor",
    FP_SPECS,
)
def LOG(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "LOG")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "LOGICAL_NOT(Tensor input1) -> Tensor",
    DUAL_PROFILE_SPECS,
)
def LOGICAL_NOT(input1: torch.Tensor) -> torch.Tensor:
    _validate_bool_dtype(input1.dtype, "LOGICAL_NOT")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "NEGATE(Tensor input1) -> Tensor",
    DUAL_PROFILE_SPECS,
)
def NEGATE(input1: torch.Tensor) -> torch.Tensor:
    _validate_negate_dtype(input1.dtype)
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "RECIPROCAL(Tensor input1) -> Tensor",
    FP_SPECS,
)
def RECIPROCAL(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "RECIPROCAL")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "RSQRT(Tensor input1) -> Tensor",
    FP_SPECS,
)
def RSQRT(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "RSQRT")
    return torch.empty_like(input1, dtype=input1.dtype)


@register_fake_tosa_op(
    "SIN(Tensor input1) -> Tensor",
    FP_SPECS,
)
def SIN(input1: torch.Tensor) -> torch.Tensor:
    _validate_float_dtype(input1.dtype, "SIN")
    return torch.empty_like(input1, dtype=input1.dtype)
