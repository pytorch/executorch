# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MXFP Conv2d transform for the Arm backend.

TorchAO extension for MXFP conv2d. It replaces ``nn.Conv2d`` with a wrapper
module that stores precomputed MXFP weights and emits a backend-internal custom
op during export.

"""

from typing import cast

import torch
import torch.nn.functional as F

from executorch.backends.arm.ao_ext.mxfp import (
    _cast_to_block_scaled_cpu_ref,
    mxfp_dtype_to_str,
    mxfp_str_to_dtype,
    MXFPDType,
    MXFPOpConfig,
)
from executorch.backends.arm.ao_ext.mxfp_tosa_lib import MXFP_TOSA_LIB
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx

MXFP_TOSA_LIB.define(
    "conv2d("
    "Tensor input, Tensor weight_qdata, Tensor weight_scale, "
    "Tensor? bias=None, int[] stride, int[] padding, "
    "int[] dilation, SymInt groups=1, SymInt block_size=32, "
    "str weight_payload_dtype=''"
    ") -> Tensor"
)


_SUPPORTED_OUTPUT_DTYPES: set[torch.dtype] = {
    torch.float32,
    torch.bfloat16,
}


def _get_mx_elem_dtype(
    weight_qdata: torch.Tensor,
    weight_payload_dtype: str = "",
) -> MXFPDType:
    if weight_payload_dtype:
        return mxfp_str_to_dtype(weight_payload_dtype)
    if weight_qdata.dtype == torch.uint8:
        return torch.float4_e2m1fn_x2
    return weight_qdata.dtype


def _get_num_input_channels(
    weight_qdata: torch.Tensor,
    weight_payload_dtype: str = "",
) -> int:
    num_input_channels = weight_qdata.shape[-1]
    if _get_mx_elem_dtype(weight_qdata, weight_payload_dtype) == torch.float4_e2m1fn_x2:
        num_input_channels *= 2
    return num_input_channels


def _conv2d_output_shape(
    input_shape: torch.Size,
    weight_shape: torch.Size,
    stride: list[int] | tuple[int, int],
    padding: list[int] | tuple[int, int],
    dilation: list[int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    n, _c, h_in, w_in = input_shape
    out_channels, kernel_h, kernel_w, _in_channels = weight_shape
    if len(stride) != 2:
        raise ValueError(f"Expected stride with 2 values, got {stride}")
    if len(padding) != 2:
        raise ValueError(f"Expected padding with 2 values, got {padding}")
    if len(dilation) != 2:
        raise ValueError(f"Expected dilation with 2 values, got {dilation}")
    pad_h, pad_w = padding

    h_out = (h_in + 2 * pad_h - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    w_out = (w_in + 2 * pad_w - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    return (n, out_channels, h_out, w_out)


@torch.library.register_fake(  # type: ignore[misc]
    "tosa_mxfp::conv2d",
    lib=MXFP_TOSA_LIB,
)
def _mxfp_conv2d_fake(
    input: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: list[int] | tuple[int, int] = (1, 1),
    padding: list[int] | tuple[int, int] = (0, 0),
    dilation: list[int] | tuple[int, int] = (1, 1),
    groups: int = 1,
    block_size: int = 32,
    weight_payload_dtype: str = "",
) -> torch.Tensor:
    del bias

    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported, got {block_size}")
    if groups != 1:
        raise ValueError(f"Only groups=1 is supported initially, got {groups}")
    if input.ndim != 4:
        raise ValueError(f"Expected rank-4 input, got {input.ndim}")
    if weight_qdata.ndim != 4:
        raise ValueError(
            f"Expected rank-4 weight_qdata for Conv2d, got {weight_qdata.ndim}"
        )
    num_input_channels = _get_num_input_channels(weight_qdata, weight_payload_dtype)
    if num_input_channels % block_size != 0:
        raise ValueError(
            f"Weight in_channels={num_input_channels} must be divisible by "
            f"{block_size=}"
        )
    if input.shape[1] != num_input_channels:
        raise ValueError(
            f"Input channels {input.shape[1]} must match weight in_channels "
            f"{num_input_channels}"
        )

    expected_scale_shape = (
        weight_qdata.shape[0],
        weight_qdata.shape[1],
        weight_qdata.shape[2],
        num_input_channels // block_size,
    )
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected weight_scale shape {expected_scale_shape}, got "
            f"{tuple(weight_scale.shape)}"
        )

    output_shape = _conv2d_output_shape(
        input.shape,
        weight_qdata.shape,
        stride,
        padding,
        dilation,
    )
    return input.new_empty(output_shape, dtype=torch.float32)


@torch.library.impl(
    "tosa_mxfp::conv2d",
    "cpu",
    lib=MXFP_TOSA_LIB,
)
def _mxfp_conv2d_cpu(
    input: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: list[int] | tuple[int, int] = (1, 1),
    padding: list[int] | tuple[int, int] = (0, 0),
    dilation: list[int] | tuple[int, int] = (1, 1),
    groups: int = 1,
    block_size: int = 32,
    weight_payload_dtype: str = "",
) -> torch.Tensor:
    """CPU reference implementation of the MXFP conv2d op."""

    if groups != 1:
        raise ValueError(f"Only groups=1 is supported initially, got {groups}")
    if len(stride) != 2:
        raise ValueError(f"Expected stride with 2 values, got {stride}")
    if len(padding) != 2:
        raise ValueError(f"Expected padding with 2 values, got {padding}")
    if len(dilation) != 2:
        raise ValueError(f"Expected dilation with 2 values, got {dilation}")

    elem_dtype = _get_mx_elem_dtype(weight_qdata, weight_payload_dtype)

    input = _cast_to_block_scaled_cpu_ref(
        input.permute(0, 2, 3, 1),
        elem_dtype,
        block_size,
    ).permute(0, 3, 1, 2)
    weight = to_dtype(
        weight_qdata,
        weight_scale,
        elem_dtype,
        block_size,
        torch.float32,
    ).permute(0, 3, 1, 2)

    if bias is not None:
        bias = bias.to(torch.float32)

    return F.conv2d(
        input,
        weight,
        bias,
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        groups,
    )


class MXFPConv2dOp(torch.nn.Module):
    """Conv2d wrapper that stores MXFP weights and emits a custom op."""

    def __init__(
        self,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        weight_dtype: MXFPDType,
        block_size: int,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight_dtype = mxfp_dtype_to_str(weight_dtype)
        self.block_size = block_size
        self.output_dtype = output_dtype

        self.register_buffer("weight_qdata", weight_qdata, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)

        self.bias: torch.nn.Parameter | None
        bias_param = (
            torch.nn.Parameter(bias.detach(), requires_grad=False)
            if bias is not None
            else None
        )
        self.register_parameter(
            "bias",
            bias_param,
        )

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.ops.tosa_mxfp.conv2d.default(
            x,
            self.weight_qdata,
            self.weight_scale,
            self.bias,
            list(self.stride),
            list(self.padding),
            list(self.dilation),
            self.groups,
            self.block_size,
            self.weight_dtype,
        )
        if self.output_dtype != torch.float32:
            output = output.to(self.output_dtype)
        return output

    def extra_repr(self) -> str:
        weight_qdata = cast(torch.Tensor, self.weight_qdata)
        weight_shape = weight_qdata.shape
        in_channels = _get_num_input_channels(weight_qdata, self.weight_dtype)
        repr_parts = [
            f"in_channels={in_channels}",
            f"out_channels={weight_shape[0]}",
            f"kernel_size={(weight_shape[1], weight_shape[2])}",
            f"stride={self.stride}",
            f"padding={self.padding}",
            f"dilation={self.dilation}",
            f"groups={self.groups}",
            f"bias={self.bias is not None}",
            f"weight_dtype={self.weight_dtype}",
            f"block_size={self.block_size}",
        ]
        return ", ".join(repr_parts)


def transform_conv2d_to_mxfp(
    module: torch.nn.Module,
    config: MXFPOpConfig,
) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Conv2d)

    if module.groups != 1:
        raise ValueError(f"Only groups=1 is supported initially, got {module.groups}")
    if isinstance(module.padding, str):
        raise ValueError(f"Unsupported Conv2d padding mode: {module.padding}")
    stride = (module.stride[0], module.stride[1])
    padding = (module.padding[0], module.padding[1])
    dilation = (module.dilation[0], module.dilation[1])

    weight_ohwi = module.weight.detach().permute(0, 2, 3, 1).contiguous()
    if weight_ohwi.shape[-1] % config.block_size != 0:
        raise ValueError(
            f"Conv2d in_channels={weight_ohwi.shape[-1]} must be divisible by "
            f"block_size={config.block_size}"
        )

    weight_scale, weight_qdata = to_mx(
        weight_ohwi,
        elem_dtype=config.weight_dtype,
        block_size=config.block_size,
        scaling_mode=config.weight_scaling_mode,
    )

    bias = module.bias.detach().to(torch.float32) if module.bias is not None else None
    output_dtype = weight_ohwi.dtype
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise ValueError(f"Unsupported output_dtype: {output_dtype}")
    return MXFPConv2dOp(
        weight_qdata,
        weight_scale,
        bias,
        stride,
        padding,
        dilation,
        module.groups,
        config.weight_dtype,
        config.block_size,
        output_dtype,
    )
