# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _validate_resize_mode(resize_mode: str) -> None:
    if resize_mode not in ("nearest", "bilinear"):
        raise TosaValueError(f"Unsupported resize mode {resize_mode} for TOSA RESIZE")


def _get_output_dtype(
    dtype: torch.dtype, tosa_spec: TosaSpecification, resize_mode: str
) -> torch.dtype:
    if dtype == torch.int8:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integers", op="RESIZE"
            )
        output_dtype = torch.int8 if resize_mode == "nearest" else torch.int32
    elif dtype == torch.int16:
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"Context TOSA spec {tosa_spec} doesn't support int16", op="RESIZE"
            )
        output_dtype = dtype
    elif dtype in (torch.float16, torch.float32, torch.bfloat16):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float", op="RESIZE"
            )
        if dtype == torch.bfloat16 and not tosa_spec.support_extension("bf16"):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support bf16", op="RESIZE"
            )
        output_dtype = dtype
    else:
        raise TosaValueError(f"Unsupported input dtype {dtype}", op="RESIZE")
    return output_dtype


def _validate_resize_parameters(scale, border, resize_mode):
    def in_int16_range(values):
        return all((x >= -(2**15)) and (x <= 2**15 - 1) for x in values)

    if not in_int16_range(scale):
        raise TosaValueError("scale is out of the int16 range", op="RESIZE")
    if not in_int16_range(border):
        raise TosaValueError("border is out of the int16 range", op="RESIZE")
    if resize_mode == "bilinear":
        scale_y_n, scale_y_d, scale_x_n, scale_x_d = scale
        if scale_y_d >= 16 * scale_y_n or scale_x_d >= 16 * scale_x_n:
            raise TosaValueError(
                "Bilinear RESIZE downscale must be strictly greater than 1/16",
                op="RESIZE",
            )


@register_fake_tosa_op(
    "RESIZE(Tensor input, SymInt[4] scale_factors, SymInt[2] offset, SymInt[2] border, *, str resize_mode) -> Tensor",  # schema
    TosaSpecification.all_versions_and_profiles(),  # target TOSA specifications
)
def RESIZE(
    x: torch.Tensor,
    scale: list[torch.SymInt],
    offset: list[torch.SymInt],
    border: list[torch.SymInt],
    *,
    resize_mode: Literal["nearest", "bilinear"],
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    if x.dim() != 4:
        raise TosaValueError(
            f"Input tensor must be 4D, but got {x.dim()}D", op="RESIZE"
        )
    _validate_resize_mode(resize_mode)
    _validate_resize_parameters(scale, border, resize_mode)
    output_dtype = _get_output_dtype(x.dtype, tosa_spec, resize_mode)

    input_shape = x.shape
    scale_y_n, scale_y_d, scale_x_n, scale_x_d = scale
    offset_y, offset_x = offset
    border_y, border_x = border
    H, W = input_shape[1], input_shape[2]
    # RESIZE first upscales the input by an integer value, to "upscale space".
    H_upscaled = (H - 1) * scale_y_n
    # offset and border are provided in this scale, therefore adjust for these while in this space.
    H_shifted = H_upscaled - offset_y + border_y
    # Then, complete the RESIZE by downscaling with another integer value, approximating multplication with a fraction.
    OH = (H_shifted // scale_y_d) + 1
    # Mirror the same computation horizontally for the output width.
    W_upscaled = (W - 1) * scale_x_n
    W_shifted = W_upscaled - offset_x + border_x
    OW = (W_shifted // scale_x_d) + 1
    fake_aten_tensor = torch.empty(
        size=(input_shape[0], OH, OW, input_shape[3]), dtype=output_dtype
    )

    return fake_aten_tensor
