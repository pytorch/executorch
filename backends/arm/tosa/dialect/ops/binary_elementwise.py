# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import (
    broadcast_shape,
    require_same_dtype,
    validate_nan_mode,
)
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

FP_SPECS = TosaSpecification.all_versions_for_profile("FP")
INT_SPECS = TosaSpecification.all_versions_for_profile("INT")
INT_DTYPES = (torch.int8, torch.int16, torch.int32)
FP_DTYPES = (torch.float16, torch.float32)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _raise_unsupported_dtype(dtype: torch.dtype, op: str) -> None:
    raise TosaValueError(f"Unsupported dtype {dtype} for {op}", op=op)


def _raise_unsupported_profile(dtype: torch.dtype, op: str) -> None:
    raise TosaValueError(
        f"TOSA spec {get_context_spec()} doesn't support {_dtype_name(dtype)} for {op}",
        op=op,
    )


def _binary_meta(
    input1: torch.Tensor,
    input2: torch.Tensor,
    op: str,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    require_same_dtype(input1, input2, op)
    output_shape = broadcast_shape(input1, input2, op)
    return torch.empty(output_shape, dtype=output_dtype or input1.dtype)


def _require_int_profile_support(dtype: torch.dtype, op: str) -> None:
    if not get_context_spec().support_integer():
        _raise_unsupported_profile(dtype, op)


def _validate_fp_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype in FP_DTYPES:
        if not tosa_spec.support_float():
            _raise_unsupported_profile(dtype, op)
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            _raise_unsupported_profile(dtype, op)
        return

    _raise_unsupported_dtype(dtype, op)


def _validate_int_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype in INT_DTYPES:
        _require_int_profile_support(dtype, op)
        return

    _raise_unsupported_dtype(dtype, op)


def _validate_any_profile_int_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype not in INT_DTYPES:
        _raise_unsupported_dtype(dtype, op)


def _validate_bitwise_and_dtype(dtype: torch.dtype) -> None:
    if dtype in INT_DTYPES:
        _require_int_profile_support(dtype, "BITWISE_AND")
        return

    if dtype == torch.int64:
        if not get_context_spec().support_extension("int64"):
            _raise_unsupported_profile(dtype, "BITWISE_AND")
        return

    _raise_unsupported_dtype(dtype, "BITWISE_AND")


def _validate_add_sub_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype == torch.int32:
        return

    _validate_fp_dtype(dtype, op)


def _validate_profile_int32_or_fp_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype == torch.int32:
        _require_int_profile_support(dtype, op)
        return

    _validate_fp_dtype(dtype, op)


def _validate_bool_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype != torch.bool:
        _raise_unsupported_dtype(dtype, op)


def _validate_int32_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype != torch.int32:
        _raise_unsupported_dtype(dtype, op)


def _validate_and_infer_mul_output_dtype(dtype: torch.dtype) -> torch.dtype:  # type: ignore[return]
    if dtype in FP_DTYPES or dtype == torch.bfloat16:
        _validate_fp_dtype(dtype, "MUL")
        return dtype

    if dtype in INT_DTYPES:
        if dtype != torch.int32:
            _require_int_profile_support(dtype, "MUL")
        return torch.int32

    _raise_unsupported_dtype(dtype, "MUL")


@register_fake_tosa_op(
    "ADD(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def ADD(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_add_sub_dtype(input1.dtype, "ADD")
    return _binary_meta(input1, input2, "ADD")


@register_fake_tosa_op(
    "ARITHMETIC_RIGHT_SHIFT(Tensor input1, Tensor input2, *, bool round=False) -> Tensor",
    INT_SPECS,
)
def ARITHMETIC_RIGHT_SHIFT(
    input1: torch.Tensor,
    input2: torch.Tensor,
    *,
    round: bool = False,
) -> torch.Tensor:
    _validate_int_dtype(input1.dtype, "ARITHMETIC_RIGHT_SHIFT")
    return _binary_meta(input1, input2, "ARITHMETIC_RIGHT_SHIFT")


@register_fake_tosa_op(
    "BITWISE_AND(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def BITWISE_AND(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_bitwise_and_dtype(input1.dtype)
    return _binary_meta(input1, input2, "BITWISE_AND")


@register_fake_tosa_op(
    "BITWISE_OR(Tensor input1, Tensor input2) -> Tensor",
    INT_SPECS,
)
def BITWISE_OR(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_int_dtype(input1.dtype, "BITWISE_OR")
    return _binary_meta(input1, input2, "BITWISE_OR")


@register_fake_tosa_op(
    "BITWISE_XOR(Tensor input1, Tensor input2) -> Tensor",
    INT_SPECS,
)
def BITWISE_XOR(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_int_dtype(input1.dtype, "BITWISE_XOR")
    return _binary_meta(input1, input2, "BITWISE_XOR")


@register_fake_tosa_op(
    "EQUAL(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def EQUAL(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_profile_int32_or_fp_dtype(input1.dtype, "EQUAL")
    return _binary_meta(input1, input2, "EQUAL", output_dtype=torch.bool)


@register_fake_tosa_op(
    "GREATER(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def GREATER(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_profile_int32_or_fp_dtype(input1.dtype, "GREATER")
    return _binary_meta(input1, input2, "GREATER", output_dtype=torch.bool)


@register_fake_tosa_op(
    "GREATER_EQUAL(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def GREATER_EQUAL(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_profile_int32_or_fp_dtype(input1.dtype, "GREATER_EQUAL")
    return _binary_meta(input1, input2, "GREATER_EQUAL", output_dtype=torch.bool)


@register_fake_tosa_op(
    "INTDIV(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def INTDIV(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_int32_dtype(input1.dtype, "INTDIV")
    return _binary_meta(input1, input2, "INTDIV")


@register_fake_tosa_op(
    "LOGICAL_AND(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def LOGICAL_AND(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_bool_dtype(input1.dtype, "LOGICAL_AND")
    return _binary_meta(input1, input2, "LOGICAL_AND")


@register_fake_tosa_op(
    "LOGICAL_LEFT_SHIFT(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def LOGICAL_LEFT_SHIFT(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_any_profile_int_dtype(input1.dtype, "LOGICAL_LEFT_SHIFT")
    return _binary_meta(input1, input2, "LOGICAL_LEFT_SHIFT")


@register_fake_tosa_op(
    "LOGICAL_RIGHT_SHIFT(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def LOGICAL_RIGHT_SHIFT(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_any_profile_int_dtype(input1.dtype, "LOGICAL_RIGHT_SHIFT")
    return _binary_meta(input1, input2, "LOGICAL_RIGHT_SHIFT")


@register_fake_tosa_op(
    "LOGICAL_OR(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def LOGICAL_OR(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_bool_dtype(input1.dtype, "LOGICAL_OR")
    return _binary_meta(input1, input2, "LOGICAL_OR")


@register_fake_tosa_op(
    "LOGICAL_XOR(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def LOGICAL_XOR(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_bool_dtype(input1.dtype, "LOGICAL_XOR")
    return _binary_meta(input1, input2, "LOGICAL_XOR")


@register_fake_tosa_op(
    "MAXIMUM(Tensor input1, Tensor input2, *, str nan_mode='PROPAGATE') -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def MAXIMUM(
    input1: torch.Tensor,
    input2: torch.Tensor,
    *,
    nan_mode: str = "PROPAGATE",
) -> torch.Tensor:
    validate_nan_mode(nan_mode, "MAXIMUM")
    _validate_profile_int32_or_fp_dtype(input1.dtype, "MAXIMUM")
    return _binary_meta(input1, input2, "MAXIMUM")


@register_fake_tosa_op(
    "MINIMUM(Tensor input1, Tensor input2, *, str nan_mode='PROPAGATE') -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def MINIMUM(
    input1: torch.Tensor,
    input2: torch.Tensor,
    *,
    nan_mode: str = "PROPAGATE",
) -> torch.Tensor:
    validate_nan_mode(nan_mode, "MINIMUM")
    _validate_profile_int32_or_fp_dtype(input1.dtype, "MINIMUM")
    return _binary_meta(input1, input2, "MINIMUM")


@register_fake_tosa_op(
    "MUL(Tensor input1, Tensor input2, *, int shift=0) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def MUL(
    input1: torch.Tensor,
    input2: torch.Tensor,
    *,
    shift: int = 0,
) -> torch.Tensor:
    output_dtype = _validate_and_infer_mul_output_dtype(input1.dtype)

    if shift < 0 or shift > 63:
        raise TosaValueError("shift must be in the range [0, 63]", op="MUL")
    if input1.dtype != torch.int32 and shift != 0:
        raise TosaValueError(
            "Only int32 MUL supports a non-zero shift value",
            op="MUL",
        )

    return _binary_meta(input1, input2, "MUL", output_dtype=output_dtype)


@register_fake_tosa_op(
    "POW(Tensor input1, Tensor input2) -> Tensor",
    FP_SPECS,
)
def POW(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_fp_dtype(input1.dtype, "POW")
    return _binary_meta(input1, input2, "POW")


@register_fake_tosa_op(
    "SUB(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def SUB(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_add_sub_dtype(input1.dtype, "SUB")
    return _binary_meta(input1, input2, "SUB")
