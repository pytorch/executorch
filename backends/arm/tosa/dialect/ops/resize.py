# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Optional

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops


# Add kwarg instead?
@register_fake_tosa_op(
    "RESIZE(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors, *, str resize_mode) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ),  # target TOSA specifications
)
def RESIZE(
    x: torch.Tensor,
    output_size: list[int] | None = None,
    align_corners: Optional[bool] = False,
    scale_factors: list[float] | None = None,
    *,
    resize_mode: Literal["nearest", "bilinear"],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    if resize_mode not in ("nearest", "bilinear"):
        raise TosaValueError(f"Unsupported resize mode {resize_mode} for TOSA RESIZE")
    if x.dtype == torch.int8:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integers", op="RESIZE"
            )
        bilinear = resize_mode == "bilinear"
        output_dtype = torch.int32 if bilinear else torch.int8
    elif x.dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float", op="RESIZE"
            )
        output_dtype = x.dtype
    else:
        raise TosaValueError(f"Unsupported input dtype {x.dtype} for TOSA RESIZE")

    # Does it matter which one to use for fake tracing?
    fake_aten_tensor = exir_ops.edge.aten.upsample_nearest2d.vec(
        x, output_size, scale_factors
    )

    return fake_aten_tensor.to(output_dtype)
