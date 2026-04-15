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
    "MAX_POOL2D(Tensor input, int[2] kernel, int[2] stride, SymInt[4] pad) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def MAX_POOL2D(
    x: torch.Tensor,
    kernel: List[int],
    stride: List[int],
    pad: List[Union[int, torch.SymInt]],
) -> torch.Tensor:
    """Compute output meta for a TOSA MAX_POOL2D operation."""
    tosa_spec = get_context_spec()

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
                f"TOSA spec {tosa_spec} doesn't support integer pools", op="MAX_POOL2D"
            )
    elif x.dtype in supported_float_types:
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float pools", op="MAX_POOL2D"
            )
    else:
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype} for TOSA MAX_POOL2D", op="MAX_POOL2D"
        )

    if x.dim() != 4:
        raise TosaValueError(
            f"MAX_POOL2D requires a 4D tensor, got {x.dim()}D", op="MAX_POOL2D"
        )

    if len(kernel) != 2 or len(stride) != 2 or len(pad) != 4:
        raise TosaValueError(
            f"MAX_POOL2D expects kernel of length 2, stride of length 2, pad of "
            f"length 4; got kernel={kernel}, stride={stride}, pad={pad}",
            op="MAX_POOL2D",
        )

    n, c, h, w = x.shape
    k_h, k_w = kernel
    s_h, s_w = stride
    # TOSA MAX_POOL2D pad order is [top, bottom, left, right]
    p_top, p_bot, p_left, p_right = pad

    h_out = (h + p_top + p_bot - k_h) // s_h + 1
    w_out = (w + p_left + p_right - k_w) // s_w + 1
    return torch.empty(size=[n, c, h_out, w_out], dtype=x.dtype)
