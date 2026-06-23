# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from executorch.backends.arm.ao_ext.mxfp import (
    mxfp_str_to_dtype,
    MXFPDType,
    SUPPORTED_MXFP_DTYPES,
)
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from torch.types import IntLikeType


def _validate_mxfp_support(tosa_spec: TosaSpecification) -> None:
    if not tosa_spec.support_extension("mxfp"):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support MXFP block-scaled Conv2d",
            op="CONV2D_BLOCK_SCALED",
        )


def _get_payload_dtype(
    data: torch.Tensor,
    payload_dtype: str = "",
) -> MXFPDType:
    if payload_dtype:
        return mxfp_str_to_dtype(payload_dtype)
    if data.dtype == torch.uint8:
        return torch.float4_e2m1fn_x2
    return data.dtype


def _get_logical_channels(data: torch.Tensor, payload_dtype: str = "") -> int:
    channels = data.shape[-1]
    if _get_payload_dtype(data, payload_dtype) == torch.float4_e2m1fn_x2:
        return channels * 2
    return channels


def _validate_conv2d_block_scaled_dtypes(
    input_data: torch.Tensor,
    input_scale: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    payload_dtype: str = "",
) -> None:
    input_dtype = _get_payload_dtype(input_data, payload_dtype)
    weight_dtype = _get_payload_dtype(weight_data, payload_dtype)
    if input_dtype not in SUPPORTED_MXFP_DTYPES:
        raise TosaValueError(
            f"Unsupported input_data dtype {input_data.dtype}",
            op="CONV2D_BLOCK_SCALED",
        )
    if weight_dtype != input_dtype:
        raise TosaValueError(
            f"weight_data dtype {weight_data.dtype} must match input_data dtype {input_data.dtype}",
            op="CONV2D_BLOCK_SCALED",
        )
    if input_scale.dtype != torch.float8_e8m0fnu:
        raise TosaValueError(
            f"Unsupported input_scale dtype {input_scale.dtype}",
            op="CONV2D_BLOCK_SCALED",
        )
    if weight_scale.dtype != torch.float8_e8m0fnu:
        raise TosaValueError(
            f"Unsupported weight_scale dtype {weight_scale.dtype}",
            op="CONV2D_BLOCK_SCALED",
        )
    if bias.dtype != torch.float32:
        raise TosaValueError(
            f"bias must be torch.float32, got {bias.dtype}",
            op="CONV2D_BLOCK_SCALED",
        )


def _validate_conv2d_block_scaled_shapes(
    input_data: torch.Tensor,
    input_scale: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    block_size: int,
    payload_dtype: str = "",
) -> tuple[
    IntLikeType, IntLikeType, IntLikeType, IntLikeType, IntLikeType, IntLikeType
]:
    if (
        input_data.ndim != 4
        or input_scale.ndim != 4
        or weight_data.ndim != 4
        or weight_scale.ndim != 4
    ):
        raise TosaValueError(
            "CONV2D_BLOCK_SCALED expects rank-4 data and scale tensors, "
            f"but got ranks {input_data.ndim}, {input_scale.ndim}, "
            f"{weight_data.ndim}, and {weight_scale.ndim}",
            op="CONV2D_BLOCK_SCALED",
        )

    n, ih, iw = input_data.shape[:3]
    ic = _get_logical_channels(input_data, payload_dtype)
    oc, kh, kw = weight_data.shape[:3]
    weight_ic = _get_logical_channels(weight_data, payload_dtype)
    if ic != weight_ic:
        raise TosaValueError(
            f"input channels must match weight channels, but got {ic} and {weight_ic}",
            op="CONV2D_BLOCK_SCALED",
        )
    if ic % block_size != 0:
        raise TosaValueError(
            f"Channel dim {ic} must be divisible by block_size {block_size}",
            op="CONV2D_BLOCK_SCALED",
        )

    expected_input_scale_shape = (n, ih, iw, ic // block_size)
    if tuple(input_scale.shape) != expected_input_scale_shape:
        raise TosaValueError(
            f"input_scale shape {tuple(input_scale.shape)} must match "
            f"{expected_input_scale_shape}",
            op="CONV2D_BLOCK_SCALED",
        )

    expected_weight_scale_shape = (oc, kh, kw, ic // block_size)
    if tuple(weight_scale.shape) != expected_weight_scale_shape:
        raise TosaValueError(
            f"weight_scale shape {tuple(weight_scale.shape)} must match "
            f"{expected_weight_scale_shape}",
            op="CONV2D_BLOCK_SCALED",
        )

    if bias.numel() not in (1, oc):
        raise TosaValueError(
            f"bias shape {tuple(bias.shape)} must broadcast over {oc} output channels",
            op="CONV2D_BLOCK_SCALED",
        )

    return n, ih, iw, oc, kh, kw


def _validate_conv2d_block_scaled_params(
    stride: list[IntLikeType], pad: list[IntLikeType], dilation: list[IntLikeType]
) -> None:
    if len(pad) != 4 or len(stride) != 2 or len(dilation) != 2:
        raise TosaValueError(
            "pad/stride/dilation must have lengths 4/2/2, "
            f"but got {len(pad)}/{len(stride)}/{len(dilation)}",
            op="CONV2D_BLOCK_SCALED",
        )


@register_fake_tosa_op(
    "CONV2D_BLOCK_SCALED("
    "Tensor input_data, Tensor input_scale, Tensor weight_data, Tensor weight_scale, "
    "Tensor bias, int[2] stride, int[4] pad, int[2] dilation, SymInt block_size, "
    "str payload_dtype=''"
    ") -> Tensor",
    TosaSpecification.all_versions_for_profile("FP"),
)
def CONV2D_BLOCK_SCALED(
    input_data: torch.Tensor,
    input_scale: torch.Tensor,
    weight_data: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    stride: list[IntLikeType],
    pad: list[IntLikeType],
    dilation: list[IntLikeType],
    block_size: int,
    payload_dtype: str = "",
) -> torch.Tensor:
    tosa_spec = get_context_spec()

    _validate_mxfp_support(tosa_spec)

    if block_size != 32:
        raise TosaValueError(
            f"Unsupported block_size {block_size}",
            op="CONV2D_BLOCK_SCALED",
        )

    _validate_conv2d_block_scaled_dtypes(
        input_data,
        input_scale,
        weight_data,
        weight_scale,
        bias,
        payload_dtype,
    )
    n, ih, iw, oc, kh, kw = _validate_conv2d_block_scaled_shapes(
        input_data,
        input_scale,
        weight_data,
        weight_scale,
        bias,
        block_size,
        payload_dtype,
    )
    _validate_conv2d_block_scaled_params(stride, pad, dilation)

    oh = (ih + pad[0] + pad[1] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (iw + pad[2] + pad[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1

    return input_data.new_empty((n, oh, ow, oc), dtype=torch.float32)
