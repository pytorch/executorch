# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Optional

import torch

from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library


m = Library("cadence", "IMPL", "CompositeExplicitAutograd")

qdtype_map: dict[ScalarType, torch.dtype] = {
    ScalarType.QINT8: torch.qint8,
    ScalarType.QUINT8: torch.quint8,
    ScalarType.QINT32: torch.qint32,
}


@impl(m, "quantize_per_tensor")
def quantize_per_tensor(
    input_tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Quantizes a floating-point tensor to an integral tensor.

    Args:
        - input_tensor (Tensor): input tensor
        - scale (float): Inverse of quantization scale. Derived from the ratio
            between the min/max of the floating-point tensor and the
            min/max of the quantized range, and then inverted.
        - zero_point (int): The point which represents 0 in the quantized
            range. For example, consider the floating point range [-1., 2.] and
            quantized integer range [-7, 7]. In this case, 0 is 1/3 of way from
            -1. to 2. So, the point that represents 0 in the quantized range should
            be 1/3 of the way from [-7, 7]. This ends up being -2 in the integer space.
        - quant_min (int): The smallest value in the quantized domain. Unused since scale
            is already provided.
        - quant_max (int): The largest value in the quantized domain. Unused since scale
            is already provided.
        - dtype (torch.dtype): The type of the output tensor
    """
    supported_quant_types = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint8,
        torch.uint16,
    ]
    if dtype not in supported_quant_types:
        raise ValueError(
            f"Unsupported dtype to quantize to. Supported dtypes must be one of {supported_quant_types}"
        )

    dequantized = torch.round(input_tensor * scale + zero_point).to(dtype)
    return torch.max(
        torch.min(dequantized, torch.tensor(quant_max)),
        torch.tensor(quant_min),
    )


@impl(m, "dequantize_per_tensor")
def dequantize_per_tensor(
    input_tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantizes an integral tensor to a floating-point tensor.

    Args:
        - input (Tensor): input tensor
        - scale (float): Quantization scale. Derived from the ratio
            between the min/max of the floating-point tensor and the
            min/max of the quantized range.
        - zero_point (int): The point which represents 0 in the quantized
            range. For example, consider the floating point range [-1., 2.] and
            quantized integer range [-7, 7]. In this case, 0 is 1/3 of way from
            -1. to 2. So, the point that represents 0 in the quantized range should
            be 1/3 of the way from [-7, 7]. This ends up being -2 in the integer space.
        - quant_min (int): The smallest value in the quantized domain. Unused since scale
            is already provided.
        - quant_max (int): The largest value in the quantized domain. Unused since scale
            is already provided.
        - dtype (torch.dtype): The type of the output tensor. Must be a floating point type.
    """
    supported_quant_types = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint8,
        torch.uint16,
    ]
    if input_tensor.dtype not in supported_quant_types:
        raise ValueError(f"Input dtype must be one of {supported_quant_types}")
    supported_dequant_types = [
        torch.float,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]
    if dtype not in supported_dequant_types:
        raise ValueError(
            f"Unsupported dtype to dequantize to. Supported dtypes must be one of {supported_dequant_types}"
        )

    # Needed to prevent underflow in cases where the zero_point is larger than
    # the quantized value.
    if not input_tensor.dtype.is_signed:
        input_tensor = input_tensor.to(torch.int32)

    return (input_tensor - zero_point).to(dtype) * scale


@impl(m, "quantized_add")
def quantized_add(
    X: torch.Tensor,
    X_scale: torch.Tensor,
    X_zero_point: torch.Tensor,
    Y: torch.Tensor,
    Y_scale: torch.Tensor,
    Y_zero_point: torch.Tensor,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    """
    Sums up two quantized tensors and returns another quantized tensor. The intuition
    is that we want dequant(out) ~= dequant(X) + dequant(Y)

    If we do that math, we get
    out_scale(out - out_zero_point) = X_scale(X - X_zero_point) + Y_scale(Y - Y_zero_point)

    Rearranging, we get
    out = (X_scale(X - X_zero_point) + Y_scale(Y - Y_zero_point)) / out_scale + out_zero_point

    Args:
        - X (Tensor): The first operand
        - X_scale (Tensor): The ratio between the sizes of X's floating point and quantized
            ranges
        - X_zero_point (Tensor): The quantized mapping of zero for X
        - Y (Tensor): The second operand
        - Y_scale (Tensor): The ratio between the sizes of Y's floating point and quantized
            ranges
        - Y_zero_point (Tensor): The quantized mapping of zero for Y
        - out_scale (float): The ratio between the sizes of the output's floating point and
            quantized ranges
        - out_zero_point (int): The quantized mapping of zero for the output
    """
    supported_dtypes = [torch.int8, torch.uint8]
    if X.dtype != Y.dtype:
        raise ValueError("X and Y dtypes need to match")

    dtype = X.dtype
    if dtype not in supported_dtypes:
        raise ValueError(
            f"X and Y dtypes need to be in {supported_dtypes}. Got {dtype}"
        )

    if dtype == torch.uint8:
        X = X.to(torch.int8)
        Y = Y.to(torch.int8)

    # TODO(agrebenisan): This should be done in fixed point arithmetic, but to match the quantized_add_out.cpp
    # reference implementation, we'll do it in floating point.
    dequant_X = X_scale * (X - X_zero_point)
    dequant_Y = Y_scale * (Y - Y_zero_point)

    out_scale_inv = 1 / out_scale

    # q_min/q_max are unused args
    return quantize_per_tensor(
        dequant_X + dequant_Y,
        out_scale_inv,
        out_zero_point,
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        dtype,
    )


@impl(m, "quantized_linear")
def quantized_linear(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
    out_zero_point: int,
    offset: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Quantized linear (transposed matmul) operation.

    Args:
        - src (Tensor): The activations tensor
        - weight (Tensor): The weight tensor
        - bias (Tensor): The bias tensor
        - in_zero_point (int): The quantized mapping of zero for the input
        - weight_zero_point (Tensor): The quantized mapping of zero for the weight
        - out_multiplier (Tensor): The multiplier used to scale the output
        - out_shift (Tensor): The shift used to scale the output
        - out_zero_point (int): The quantized mapping of zero for the output
        - offset (Tensor): Unused
    """
    out_scale = -out_multiplier * (1 / (1 << 31)) * (2 ** out_shift[0])
    out_scale_inv = 1 / out_scale

    N, K = weight.shape

    leading_dims = src.shape[:-1]
    src = src.view(-1, K)

    dtype = src.dtype
    supported_dtypes = [torch.int8, torch.uint8, torch.int32]
    if dtype not in supported_dtypes:
        raise ValueError(
            f"Unsupported dtype to quantize to. Supported dtypes must be one of {supported_dtypes}"
        )

    out = torch.nn.functional.linear(
        src - in_zero_point, weight - weight_zero_point, bias
    )
    return quantize_per_tensor(
        out,
        out_scale_inv,
        out_zero_point,
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        dtype,
    ).reshape(*leading_dims, N)


@impl(m, "quantized_layer_norm_per_tensor")
def quantized_layer_norm_per_tensor(
    input_tensor: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    normalized_shape: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_scale: float,
    output_zero_point: int,
) -> torch.Tensor:
    """
    Quantized layer norm operation.

    Args:
        - input_tensor (Tensor): The activations tensor
        - X_scale (float): The scale of the input
        - X_zero_point (int): The zero point of the input
        - normalized_shape (int): The shape of the input
        - weight (Tensor): The weight tensor
        - bias (Tensor): The bias tensor
        - eps (float): The epsilon value
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
    """
    supported_dtypes = [torch.int8, torch.uint8]
    if input_tensor.dtype not in supported_dtypes:
        raise ValueError(
            f"Input dtype must be one of {supported_dtypes}. Got {input_tensor.dtype}"
        )

    float_input_tensor = dequantize_per_tensor(
        input_tensor, X_scale, X_zero_point, -128, 127, torch.float32
    )
    out = torch.nn.functional.layer_norm(
        float_input_tensor, (normalized_shape,), weight, bias, eps=eps
    )

    return quantize_per_tensor(
        out,
        1 / output_scale,
        output_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )


def quantized_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    """
    Quantized convolution operation.

    Args:
        - input_tensor (Tensor): The activations tensor
        - weight (Tensor): The weight tensor
        - bias (Tensor): The bias tensor
        - stride (Tuple[int]): The stride of the convolution
        - padding (Tuple[int]): The padding of the convolution
        - dilation (Tuple[int]): The dilation of the convolution
        - groups (int): The number of groups
        - in_zero_point (int): The quantized mapping of zero for the input
        - weight_zero_point (Tensor): The quantized mapping of zero for the weight
        - bias_scale (Tensor): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (Tensor): Unused
        - out_shift (Tensor): Unused
    """
    if weight_zero_point.view(-1).shape != (1,):
        raise ValueError("Weight zero point must be a scalar")

    if bias_scale.view(-1).shape != (1,):
        raise ValueError("Bias scale must be a scalar")

    if len(input_tensor.shape) == 3:
        float_out = torch.nn.functional.conv1d(
            (input_tensor - in_zero_point).float(),
            (weight - weight_zero_point).float(),
            (bias * bias_scale).float(),
            stride[1],
            padding[1],
            dilation[1],
            groups,
        )

    elif len(input_tensor.shape) == 4:
        float_out = torch.nn.functional.conv2d(
            (input_tensor - in_zero_point).float(),
            (weight - weight_zero_point).float(),
            (bias * bias_scale).float(),
            stride,
            padding,
            dilation,
            groups,
        )
    else:
        raise ValueError("Input tensor must be 3D or 4D")

    return quantize_per_tensor(
        float_out,
        1.0 / output_scale,
        output_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )


@impl(m, "quantized_conv_nchw")
def quantized_conv_nchw(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    """
    Quantized convolution operation.

    Args:
        - input_tensor (Tensor): The activations tensor
        - weight (Tensor): The weight tensor
        - bias (Tensor): The bias tensor
        - stride (Tuple[int]): The stride of the convolution
        - padding (Tuple[int]): The padding of the convolution
        - dilation (Tuple[int]): The dilation of the convolution
        - groups (int): The number of groups
        - in_zero_point (int): The quantized mapping of zero for the input
        - weight_zero_point (Tensor): The quantized mapping of zero for the weight
        - bias_scale (Tensor): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (Tensor): Unused
        - out_shift (Tensor): Unused
    """
    if not input_tensor.is_contiguous(memory_format=torch.contiguous_format):
        raise ValueError("Input tensor must be in NCHW format")
    return quantized_conv(
        input_tensor,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out_multiplier,
        out_shift,
    )


@impl(m, "quantized_conv_nhwc")
def quantized_conv_nhwc(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: torch.Tensor,
    bias_scale: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    """
    Quantized convolution operation.

    Args:
        - input_tensor (Tensor): The activations tensor
        - weight (Tensor): The weight tensor
        - bias (Tensor): The bias tensor
        - stride (Tuple[int]): The stride of the convolution
        - padding (Tuple[int]): The padding of the convolution
        - dilation (Tuple[int]): The dilation of the convolution
        - groups (int): The number of groups
        - in_zero_point (int): The quantized mapping of zero for the input
        - weight_zero_point (Tensor): The quantized mapping of zero for the weight
        - bias_scale (Tensor): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (Tensor): Unused
        - out_shift (Tensor): Unused
    """

    if not input_tensor.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("Input tensor must be in NHWC format")

    return quantized_conv(
        input_tensor,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out_multiplier,
        out_shift,
    )


@impl(m, "requantize")
def requantize(
    input: torch.Tensor,
    in_scale: torch.Tensor,
    in_zero_point: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    dtype: ScalarType,
) -> torch.Tensor:
    if dtype in qdtype_map:
        # Old quantization mechanism
        return torch.quantize_per_tensor(
            torch.dequantize(input), out_scale, out_zero_point, qdtype_map[dtype]
        )

    # For in_scale or out_scale other than scalar, it requires quant/dequant
    # per channel, but the channel dimension value is missing
    if in_scale.numel() > 1 or out_scale.numel() > 1:
        raise NotImplementedError("Only scalar scales are supported")

    quant_min = torch.iinfo(input.dtype).min
    quant_max = torch.iinfo(input.dtype).max
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_min = torch.iinfo(dtype).min
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_max = torch.iinfo(dtype).max
    return torch.ops.quantized_decomposed.quantize_per_tensor(
        torch.ops.quantized_decomposed.dequantize_per_tensor(
            input,
            in_scale.flatten()[0],
            in_zero_point.flatten()[0],
            quant_min,
            quant_max,
            input.dtype,
        ),
        out_scale.flatten()[0],
        out_zero_point.flatten()[0],
        out_quant_min,
        out_quant_max,
        dtype,
    )
