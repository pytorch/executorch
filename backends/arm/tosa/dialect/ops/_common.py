# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError

_VALID_NAN_MODES = {"PROPAGATE", "IGNORE"}


def broadcast_shape(
    input1: torch.Tensor, input2: torch.Tensor, op: str
) -> tuple[int | torch.SymInt, ...]:
    try:
        return tuple(torch.broadcast_shapes(input1.shape, input2.shape))
    except (RuntimeError, ValueError) as err:
        raise TosaValueError(
            f"Failed to broadcast shapes {tuple(input1.shape)} and {tuple(input2.shape)}",
            op=op,
        ) from err


def require_same_dtype(input1: torch.Tensor, input2: torch.Tensor, op: str) -> None:
    if input1.dtype != input2.dtype:
        raise TosaValueError(
            f"Expected matching dtypes but got {input1.dtype} and {input2.dtype}",
            op=op,
        )


def validate_nan_mode(nan_mode: str, op: str) -> None:
    if nan_mode not in _VALID_NAN_MODES:
        raise TosaValueError(
            f"Unsupported nan_mode {nan_mode}. Expected one of {_VALID_NAN_MODES}",
            op=op,
        )


def validate_power_of_two(size: int | torch.SymInt, name: str, op: str) -> None:
    if not isinstance(size, int):
        return
    if size < 1 or (size & (size - 1)) != 0:
        raise TosaValueError(
            f"{name} must be a positive power of two, got {size}", op=op
        )
