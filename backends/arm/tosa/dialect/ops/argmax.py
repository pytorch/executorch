# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops._common import validate_nan_mode
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _validate_argmax_dtype(dtype: torch.dtype) -> None:
    tosa_spec = get_context_spec()

    if dtype == torch.int8:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support int8 for ARGMAX",
                op="ARGMAX",
            )
        return

    if dtype == torch.int16:
        if not (tosa_spec.support_integer() and tosa_spec.support_extension("int16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support int16 for ARGMAX",
                op="ARGMAX",
            )
        return

    if dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {dtype} for ARGMAX",
                op="ARGMAX",
            )
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support bfloat16 for ARGMAX",
                op="ARGMAX",
            )
        return

    raise TosaValueError(f"Unsupported dtype {dtype} for ARGMAX", op="ARGMAX")


@register_tosa_op(
    'ARGMAX(Tensor input, int axis, *, str nan_mode="PROPAGATE") -> Tensor',
    TosaSpecification.all_versions_and_profiles(),
)
def ARGMAX(
    input: torch.Tensor,
    axis: int,
    *,
    nan_mode: str = "PROPAGATE",
) -> torch.Tensor:
    validate_nan_mode(nan_mode, "ARGMAX")
    _validate_argmax_dtype(input.dtype)

    if input.dim() == 0:
        raise TosaValueError(
            "ARGMAX requires an input with rank at least 1", op="ARGMAX"
        )
    if axis < 0 or axis >= input.dim():
        raise TosaValueError(
            f"axis must be in [0, {input.dim() - 1}] but got {axis}",
            op="ARGMAX",
        )

    output_shape = tuple(input.shape[:axis]) + tuple(input.shape[axis + 1 :])
    return torch.empty(output_shape, dtype=torch.int32)
