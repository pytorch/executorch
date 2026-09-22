# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import binary_meta
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)

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


def _validate_comparison_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype == torch.int32:
        if not tosa_spec.support_integer():
            _raise_unsupported_profile(dtype, op)
        return

    if dtype in FP_DTYPES:
        if not tosa_spec.support_float():
            _raise_unsupported_profile(dtype, op)
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            _raise_unsupported_profile(dtype, op)
        return

    _raise_unsupported_dtype(dtype, op)


@register_fake_tosa_op(
    "EQUAL(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def EQUAL(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_comparison_dtype(input1.dtype, "EQUAL")
    return binary_meta(input1, input2, "EQUAL", output_dtype=torch.bool)


@register_fake_tosa_op(
    "GREATER(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def GREATER(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_comparison_dtype(input1.dtype, "GREATER")
    return binary_meta(input1, input2, "GREATER", output_dtype=torch.bool)


@register_fake_tosa_op(
    "GREATER_EQUAL(Tensor input1, Tensor input2) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def GREATER_EQUAL(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    _validate_comparison_dtype(input1.dtype, "GREATER_EQUAL")
    return binary_meta(input1, input2, "GREATER_EQUAL", output_dtype=torch.bool)
