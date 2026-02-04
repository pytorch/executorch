# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "RESCALE(Tensor input1, ScalarType dtype, float[] scale, int in_zp, int out_zp) -> Tensor",  # schema
    TosaSpecification.all_versions_for_profile("INT"),  # target TOSA specifications
)
def RESCALE(
    x: torch.Tensor, dtype: torch.dtype, scales: List[float], in_zp: int, out_zp: int
) -> torch.Tensor:
    tosa_spec = get_context_spec()
    """Casts the input tensor to dtype `dtype` to produce the correct tensor
    meta for a _rescale op.

    Additionally validates TOSA constraints of a RESCALE op.

    """
    if not tosa_spec.support_integer():
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support integers", op="RESCALE"
        )

    if dtype not in (torch.int32, torch.int8, torch.int16):
        raise NotImplementedError(
            f"tosa::rescale currently only supports int32, int16 and int8, not {dtype}"
        )
    if dtype in (torch.int32, torch.int16) and out_zp != 0:
        raise ValueError(
            f"TOSA requires output_zp to be zero when the output dtype is {dtype}."
        )
    if x.dtype in (torch.int32, torch.int16) and in_zp != 0:
        raise ValueError(
            f"TOSA requires input_zp to be zero when the input dtype is {dtype}"
        )
    if x.dtype == torch.int8 and not -128 <= in_zp <= 127:
        raise ValueError(f"{in_zp=} outside valid range (-128,127) for int8.")
    if dtype == torch.int8 and not -128 <= out_zp <= 127:
        raise ValueError(f"{out_zp=} outside valid range (-128,127) for int8.")

    return torch.empty_like(x, dtype=dtype)
