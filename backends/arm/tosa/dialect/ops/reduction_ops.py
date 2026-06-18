# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _validate_axis(x: torch.Tensor, axis: int, op: str) -> None:
    if x.dim() < 1:
        raise TosaValueError(f"{op} requires rank >= 1 input", op=op)
    if axis < 0 or axis >= x.dim():
        raise TosaValueError(
            f"{op} axis {axis} is out of range for rank {x.dim()}",
            op=op,
        )


def _reduce_shape(x: torch.Tensor, axis: int) -> list[int | torch.SymInt]:
    output_shape: list[int | torch.SymInt] = list(x.shape)
    output_shape[axis] = 1
    return output_shape


def _validate_bool_dtype(x: torch.Tensor, op: str) -> None:
    if x.dtype != torch.bool:
        raise TosaValueError(f"{op} requires bool input, got {x.dtype}", op=op)


def _validate_float_integer_dtype(x: torch.Tensor, op: str) -> None:
    tosa_spec = get_context_spec()
    supported_int_dtypes = {torch.int8, torch.int16, torch.int32}
    supported_float_dtypes = {torch.float16, torch.float32}

    if tosa_spec.support_extension("int64"):
        supported_int_dtypes.add(torch.int64)
    if tosa_spec.support_extension("bf16"):
        supported_float_dtypes.add(torch.bfloat16)

    if x.dtype in supported_int_dtypes:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integer reductions",
                op=op,
            )
        return

    if x.dtype in supported_float_dtypes:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support floating-point reductions",
                op=op,
            )
        return

    raise TosaValueError(f"Unsupported dtype {x.dtype} for {op}", op=op)


def _validate_reduce_sum_dtype(x: torch.Tensor) -> None:
    tosa_spec = get_context_spec()
    supported_int_dtypes = {torch.int32}
    supported_float_dtypes = {torch.float16, torch.float32}

    if tosa_spec.support_extension("int64"):
        supported_int_dtypes.add(torch.int64)
    if tosa_spec.support_extension("bf16"):
        supported_float_dtypes.add(torch.bfloat16)

    if x.dtype in supported_int_dtypes:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integer reductions",
                op="REDUCE_SUM",
            )
        return

    if x.dtype in supported_float_dtypes:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support floating-point reductions",
                op="REDUCE_SUM",
            )
        return

    raise TosaValueError(
        f"Unsupported dtype {x.dtype} for REDUCE_SUM",
        op="REDUCE_SUM",
    )


def _validate_product_dtype(x: torch.Tensor, op: str) -> None:
    tosa_spec = get_context_spec()
    supported_dtypes = {torch.float16, torch.float32}
    if tosa_spec.support_extension("bf16"):
        supported_dtypes.add(torch.bfloat16)

    if x.dtype not in supported_dtypes:
        raise TosaValueError(
            f"{op} requires floating-point input, got {x.dtype}", op=op
        )
    if not tosa_spec.support_float():
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support floating-point reductions",
            op=op,
        )


def _validate_nan_mode(nan_mode: str, op: str) -> None:
    if nan_mode not in ("PROPAGATE", "IGNORE"):
        raise TosaValueError(
            f"Invalid nan_mode {nan_mode}, must be PROPAGATE or IGNORE",
            op=op,
        )


@register_tosa_op(
    "REDUCE_ALL(Tensor input, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_ALL(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_ALL")
    _validate_bool_dtype(x, "REDUCE_ALL")
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)


@register_tosa_op(
    "REDUCE_ANY(Tensor input, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_ANY(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_ANY")
    _validate_bool_dtype(x, "REDUCE_ANY")
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)


@register_tosa_op(
    'REDUCE_MAX(Tensor input, *, int axis, str nan_mode="PROPAGATE") -> Tensor',
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_MAX(
    x: torch.Tensor, *, axis: int, nan_mode: str = "PROPAGATE"
) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_MAX")
    _validate_float_integer_dtype(x, "REDUCE_MAX")
    _validate_nan_mode(nan_mode, "REDUCE_MAX")
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)


@register_tosa_op(
    'REDUCE_MIN(Tensor input, *, int axis, str nan_mode="PROPAGATE") -> Tensor',
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_MIN(
    x: torch.Tensor, *, axis: int, nan_mode: str = "PROPAGATE"
) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_MIN")
    _validate_float_integer_dtype(x, "REDUCE_MIN")
    _validate_nan_mode(nan_mode, "REDUCE_MIN")
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)


@register_tosa_op(
    "REDUCE_PRODUCT(Tensor input, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_PRODUCT(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_PRODUCT")
    _validate_product_dtype(x, "REDUCE_PRODUCT")
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)


@register_tosa_op(
    "REDUCE_SUM(Tensor input, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def REDUCE_SUM(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    _validate_axis(x, axis, "REDUCE_SUM")
    _validate_reduce_sum_dtype(x)
    return torch.empty(size=_reduce_shape(x, axis), dtype=x.dtype)
