# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import (
    broadcast_shape,
    require_same_dtype,
)
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _raise_unsupported_dtype(dtype: torch.dtype, op: str) -> None:
    raise TosaValueError(f"Unsupported dtype {dtype} for {op}", op=op)


def _raise_unsupported_profile(dtype: torch.dtype, op: str) -> None:
    raise TosaValueError(
        f"TOSA spec {get_context_spec()} doesn't support {dtype} for {op}",
        op=op,
    )


def _validate_condition_dtype(dtype: torch.dtype, op: str) -> None:
    if dtype != torch.bool:
        raise TosaValueError(f"{op} requires bool condition but got {dtype}", op=op)


def _validate_select_value_dtype(dtype: torch.dtype) -> None:
    tosa_spec = get_context_spec()

    if dtype == torch.bool:
        return

    if dtype in {torch.int8, torch.int16, torch.int32}:
        if not tosa_spec.support_integer():
            _raise_unsupported_profile(dtype, "SELECT")
        return

    if dtype in {torch.float16, torch.float32}:
        if not tosa_spec.support_float():
            _raise_unsupported_profile(dtype, "SELECT")
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            _raise_unsupported_profile(dtype, "SELECT")
        return

    _raise_unsupported_dtype(dtype, "SELECT")


@register_fake_tosa_op(
    "SELECT(Tensor condition, Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def SELECT(
    condition: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> torch.Tensor:
    _validate_condition_dtype(condition.dtype, "SELECT")
    require_same_dtype(input1, input2, "SELECT")
    _validate_select_value_dtype(input1.dtype)
    output_shape = broadcast_shape(condition, input1, "SELECT")
    output_shape = broadcast_shape(
        torch.empty(output_shape, dtype=input1.dtype),
        input2,
        "SELECT",
    )
    return torch.empty(output_shape, dtype=input1.dtype)
