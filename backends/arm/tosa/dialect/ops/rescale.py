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
    "RESCALE(Tensor input1, ScalarType dtype, float[] scale, int in_zp, int out_zp, *, bool input_unsigned=False, bool output_unsigned=False) -> Tensor",  # schema
    TosaSpecification.all_versions_for_profile("INT"),  # target TOSA specifications
)
def RESCALE(  # noqa: C901
    x: torch.Tensor,
    dtype: torch.dtype,
    scales: List[float],
    in_zp: int,
    out_zp: int,
    *,
    input_unsigned: bool = False,
    output_unsigned: bool = False,
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
        raise TosaValueError(
            f"tosa::rescale currently only supports int32, int16 and int8, not {dtype}",
            op="RESCALE",
        )
    if input_unsigned and output_unsigned:
        raise TosaValueError(
            "TOSA requires input_unsigned and output_unsigned not both be true.",
            op="RESCALE",
        )

    if input_unsigned:
        if x.dtype not in (torch.int8, torch.int16, torch.uint8):
            raise TosaValueError(
                f"input_unsigned requires int8/int16/uint8 input dtype, got {x.dtype}.",
                op="RESCALE",
            )
        if x.dtype == torch.int32:
            raise TosaValueError(
                "TOSA forbids input_unsigned for int32 inputs.", op="RESCALE"
            )
        if x.dtype == torch.int16:
            if in_zp not in (0, 32768):
                raise TosaValueError(
                    f"{in_zp=} outside valid range (0,32768) for uint16.",
                    op="RESCALE",
                )
        else:
            if not 0 <= in_zp <= 255:
                raise TosaValueError(
                    f"{in_zp=} outside valid range (0,255) for uint8.",
                    op="RESCALE",
                )
    else:
        if x.dtype in (torch.int32, torch.int16) and in_zp != 0:
            raise TosaValueError(
                f"TOSA requires input_zp to be zero when the input dtype is {x.dtype}.",
                op="RESCALE",
            )
        if x.dtype == torch.int8 and not -128 <= in_zp <= 127:
            raise TosaValueError(
                f"{in_zp=} outside valid range (-128,127) for int8.",
                op="RESCALE",
            )

    if output_unsigned:
        if dtype not in (torch.int8, torch.int16):
            raise TosaValueError(
                f"output_unsigned requires int8/int16 output dtype, got {dtype}.",
                op="RESCALE",
            )
        if dtype == torch.int32:
            raise TosaValueError(
                "TOSA forbids output_unsigned for int32 outputs.", op="RESCALE"
            )
        if dtype == torch.int16:
            if out_zp not in (0, 32768):
                raise TosaValueError(
                    f"{out_zp=} outside valid range (0,32768) for uint16.",
                    op="RESCALE",
                )
        else:
            if not 0 <= out_zp <= 255:
                raise TosaValueError(
                    f"{out_zp=} outside valid range (0,255) for uint8.",
                    op="RESCALE",
                )
    else:
        if dtype in (torch.int32, torch.int16) and out_zp != 0:
            raise TosaValueError(
                f"TOSA requires output_zp to be zero when the output dtype is {dtype}.",
                op="RESCALE",
            )
        if dtype == torch.int8 and not -128 <= out_zp <= 127:
            raise TosaValueError(
                f"{out_zp=} outside valid range (-128,127) for int8.",
                op="RESCALE",
            )

    return torch.empty_like(x, dtype=dtype)
