# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "AVG_POOL2D(Tensor input, Tensor input_zp, Tensor output_zp, int[2] kernel, int[2] stride, SymInt[4] pad, ScalarType acc_type) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def AVG_POOL2D(
    x: torch.Tensor,
    input_zp: torch.Tensor,
    output_zp: torch.Tensor,
    kernel: List[int],
    stride: List[int],
    pad: List[Union[int, torch.SymInt]],
    acc_type: torch.dtype,
) -> torch.Tensor:
    """Compute output meta for a TOSA AVG_POOL2D operation."""
    tosa_spec = get_context_spec()

    # Validate dtype support
    supported_int_types = [torch.int8]
    supported_float_types = [
        torch.float16,
        torch.float32,
    ]
    if tosa_spec.support_extension("bf16"):
        supported_float_types.append(torch.bfloat16)
    if tosa_spec.support_extension("int16"):
        supported_int_types.append(torch.int16)

    if x.dtype in supported_int_types:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integer pools", op="AVG_POOL2D"
            )
    elif x.dtype in supported_float_types:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float pools", op="AVG_POOL2D"
            )
    else:
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype} for TOSA AVG_POOL2D", op="AVG_POOL2D"
        )

    # Validate input dimensions
    if x.dim() != 4:
        raise TosaValueError(
            f"AVG_POOL2D requires a 4D tensor, got {x.dim()}D", op="AVG_POOL2D"
        )

    # Validate kernel, stride, pad lengths
    if len(kernel) != 2 or len(stride) != 2 or len(pad) != 4:
        raise TosaValueError(
            f"AVG_POOL2D expects kernel of length 2, stride of length 2, pad of length 4; got "
            f"kernel={kernel}, stride={stride}, pad={pad}",
            op="AVG_POOL2D",
        )

    # Validate and determine accumulator (output) dtype: only FP32 or INT32
    acc_allowed = [torch.float32, torch.int32]
    if acc_type not in acc_allowed:
        raise TosaValueError(
            f"Unsupported acc_type {acc_type} for TOSA AVG_POOL2D; "
            f"must be one of {acc_allowed}",
            op="AVG_POOL2D",
        )
    # Unpack dimensions and parameters; zero-points are not used for shape
    n, c, h, w = x.shape

    k_h, k_w = kernel
    s_h, s_w = stride
    p_top, p_left, p_bot, p_right = pad
    # Compute output spatial dimensions (floor division)
    h_out = (h + p_top + p_left - k_h) // s_h + 1
    w_out = (w + p_bot + p_right - k_w) // s_w + 1

    # Return a tensor with the computed shape and dtype
    return torch.empty(size=[n, c, h_out, w_out], dtype=x.dtype)
