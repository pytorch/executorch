# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library

m = Library("cadence", "IMPL", "CompositeExplicitAutograd")
torch.ops.load_library("//executorch/kernels/quantized:custom_ops_generated_lib")

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
        - scale (float): Quantization scale. Derived from the ratio
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

    return torch.ops.quantized_decomposed.quantize_per_tensor(
        input_tensor,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
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
        - dtype (torch.dtype): The type of the input tensor.
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
    if input_tensor.dtype != dtype:
        raise ValueError("Input dtype must match dtype")

    # Use the reference implementation from torch quantized_decomposed library
    # Unlike quantize_per_tensor, dequantize_per_tensor doesn't have a behavior
    # difference, since there's no rounding algorithm (just arithmetic).
    return torch.ops.quantized_decomposed.dequantize_per_tensor(
        input_tensor, scale, zero_point, quant_min, quant_max, dtype
    )


@impl(m, "quantized_add.per_tensor")
def quantized_add_per_tensor(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
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
        - X: The first operand
        - X_scale: The ratio between the sizes of X's floating point and quantized
            ranges
        - X_zero_point: The quantized mapping of zero for X
        - Y: The second operand
        - Y_scale: The ratio between the sizes of Y's floating point and quantized
            ranges
        - Y_zero_point: The quantized mapping of zero for Y
        - out_scale: The ratio between the sizes of the output's floating point and
            quantized ranges
        - out_zero_point: The quantized mapping of zero for the output
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

    # q_min/q_max are unused args
    return quantize_per_tensor(
        dequant_X + dequant_Y,
        out_scale,
        out_zero_point,
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        dtype,
    )


@impl(m, "quantized_add_asym8sxasym8s_asym8s.per_tensor")
def quantized_add_asym8sxasym8s_asym8s_per_tensor(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    if X.dtype != torch.int8:
        raise ValueError("X dtype must be torch.int8")
    if Y.dtype != torch.int8:
        raise ValueError("Y dtype must be torch.int8")

    return quantized_add_per_tensor(
        X, X_scale, X_zero_point, Y, Y_scale, Y_zero_point, out_scale, out_zero_point
    )


@impl(m, "quantized_add_asym8uxasym8u_asym8u.per_tensor")
def quantized_add_asym8uxasym8u_asym8u_per_tensor(
    X: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_scale: float,
    Y_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    if X.dtype != torch.uint8:
        raise ValueError("X dtype must be torch.int8")
    if Y.dtype != torch.uint8:
        raise ValueError("Y dtype must be torch.int8")

    return quantized_add_per_tensor(
        X, X_scale, X_zero_point, Y, Y_scale, Y_zero_point, out_scale, out_zero_point
    )


def quantized_linear_common(
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_zero_point: int,
    weight_zero_point: torch.Tensor | int,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
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
    out_scale = 1.0 / (-out_multiplier * (1 / (1 << 31)) * (2**out_shift))

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
        (src - in_zero_point).float(),
        (weight - weight_zero_point).float(),
        bias.float(),
    )
    return quantize_per_tensor(
        out,
        out_scale,
        out_zero_point,
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        dtype,
    ).reshape(*leading_dims, N)


def quantized_linear_variant(
    per_tensor: bool,
    fully_connected: bool,
    src_dtype: torch.dtype | None = None,
    weight_dtype: torch.dtype | None = None,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            src: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            in_zero_point: int,
            weight_zero_point: torch.Tensor | int,
            out_multiplier: torch.Tensor | int,
            out_shift: torch.Tensor | int,
            out_zero_point: int,
            offset: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if fully_connected and src.shape[0] != 1:
                raise ValueError(
                    "Fully connected quantized linear only supports batch size of 1"
                )
            if src_dtype and src.dtype != src_dtype:
                raise ValueError(
                    f"src dtype must be {src_dtype}. Got {src.dtype} instead"
                )
            if weight_dtype and weight.dtype != weight_dtype:
                raise ValueError(
                    f"weight dtype must be {weight_dtype}. Got {weight.dtype} instead"
                )
            if bias.dtype != torch.int32:
                raise ValueError(
                    f"bias dtype must be torch.int32. Got {bias.dtype} instead"
                )

            if per_tensor:
                assert isinstance(weight_zero_point, int)
                assert isinstance(out_multiplier, int)
                assert isinstance(out_shift, int)
                _out_shift = out_shift
                _out_multiplier = out_multiplier
            else:
                assert isinstance(out_shift, torch.Tensor)
                assert isinstance(out_multiplier, torch.Tensor)
                if out_shift.numel() != 1:
                    raise ValueError("out_shift must be a scalar")

                if out_shift.dtype != torch.int64:
                    raise ValueError("out_shift must be an int64")

                _out_shift = int(out_shift.item())
                _out_multiplier = int(out_multiplier[0].item())

            return quantized_linear_common(
                src,
                weight,
                bias,
                in_zero_point,
                weight_zero_point,
                _out_multiplier,
                _out_shift,
                out_zero_point,
            )

        return variant

    return decorator


@impl(m, "quantized_linear")
@quantized_linear_variant(False, False)
def quantized_linear() -> torch.Tensor: ...


@impl(m, "quantized_linear.per_tensor")
@quantized_linear_variant(True, False)
def quantized_linear_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_linear_asym8sxasym8s_asym8s.per_tensor")
@quantized_linear_variant(True, False, torch.int8, torch.int8)
def quantized_linear_asym8sxasym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_linear_asym8uxasym8u_asym8u.per_tensor")
@quantized_linear_variant(True, False, torch.uint8, torch.uint8)
def quantized_linear_asym8uxasym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_fully_connected")
@quantized_linear_variant(False, True)
def quantized_fully_connected() -> torch.Tensor: ...


@impl(m, "quantized_fully_connected.per_tensor")
@quantized_linear_variant(True, True)
def quantized_fully_connected_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor")
@quantized_linear_variant(True, True, torch.int8, torch.int8)
def quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor")
@quantized_linear_variant(True, True, torch.uint8, torch.uint8)
def quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "fully_connected")
def fully_connected(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if input_tensor.shape[0] != 1:
        raise ValueError("Fully connected linear only supports batch size of 1")
    return F.linear(input_tensor, weight, bias)


@impl(m, "quantized_matmul")
def quantized_matmul(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: torch.Tensor | None,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    """
    Quantized matmul operation.

    Args:
        - X (Tensor): The activations tensor
        - X_zero_point (int): The quantized mapping of zero for the input
        - Y (Tensor): The weight tensor
        - Y_zero_point (int): The quantized mapping of zero for the weight
        - bias (Tensor): The bias tensor
        - out_multiplier (int): The multiplier used to scale the output
        - out_shift (int): The shift used to scale the output
        - out_zero_point (int): The quantized mapping of zero for the output
        - transposed (bool): Whether to transpose the weight tensor
    """
    if bias is not None and not torch.all(bias == 0):
        raise ValueError("bias must be None or all zeros since unused in out variant")

    # Looks weird, but quantized linear assumes weights are pre-transposed,
    # hence we transpose only if `transposed` is False.
    if not transposed:
        Y = Y.T

    return quantized_linear_common(
        X,
        Y,
        bias or torch.zeros(1, dtype=torch.int32),
        X_zero_point,
        Y_zero_point,
        out_multiplier,
        out_shift,
        out_zero_point,
    )


@impl(m, "quantized_matmul_asym8sxasym8s_asym8s")
def quantized_matmul_asym8sxasym8s_asym8s(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: torch.Tensor | None,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    if X.dtype != torch.int8:
        raise ValueError("X dtype must be torch.int8")
    if Y.dtype != torch.int8:
        raise ValueError("Y dtype must be torch.int8")

    return quantized_matmul(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
    )


@impl(m, "quantized_matmul_asym8uxasym8u_asym8u")
def quantized_matmul_asym8uxasym8u_asym8u(
    X: torch.Tensor,
    X_zero_point: int,
    Y: torch.Tensor,
    Y_zero_point: int,
    bias: torch.Tensor | None,
    out_multiplier: int,
    out_shift: int,
    out_zero_point: int,
    transposed: bool = False,
) -> torch.Tensor:
    if X.dtype != torch.uint8:
        raise ValueError("X dtype must be torch.uint8")
    if Y.dtype != torch.uint8:
        raise ValueError("Y dtype must be torch.uint8")

    return quantized_matmul(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
    )


@impl(m, "quantized_layer_norm.per_tensor")
def quantized_layer_norm_per_tensor(
    input_tensor: torch.Tensor,
    X_scale: float,
    X_zero_point: int,
    normalized_shape: list[int],
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
        input_tensor, X_scale, X_zero_point, -128, 127, input_tensor.dtype
    )
    out = torch.nn.functional.layer_norm(
        float_input_tensor, normalized_shape, weight, bias, eps=eps
    )

    return quantize_per_tensor(
        out,
        output_scale,
        output_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )


def quantized_conv_per_tensor(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
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
        - weight_zero_point (int): The quantized mapping of zero for the weight
        - bias_scale (float): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (int): Unused
        - out_shift (int): Unused
    """
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
        output_scale,
        output_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )


@impl(m, "quantized_conv2d_nchw.per_tensor")
def quantized_conv2d_nchw_per_tensor(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
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
        - weight_zero_point (int): The quantized mapping of zero for the weight
        - bias_scale (float): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (int): Unused
        - out_shift (int): Unused
    """
    if not input_tensor.is_contiguous(memory_format=torch.contiguous_format):
        raise ValueError("Input tensor must be in NCHW format")
    return quantized_conv_per_tensor(
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


@impl(m, "quantized_conv2d_nhwc.per_tensor")
def quantized_conv2d_nhwc_per_tensor(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    in_zero_point: int,
    weight_zero_point: int,
    bias_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_multiplier: int,
    out_shift: int,
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
        - weight_zero_point (int): The quantized mapping of zero for the weight
        - bias_scale (float): The quantized bias scale
        - output_scale (float): The scale of the output
        - output_zero_point (int): The zero point of the output
        - out_multiplier (int): Unused
        - out_shift (int): Unused
    """

    # Convert to NCHW format to reuse the existing implementation
    conv_is_1d = False
    if len(input_tensor.shape) == 3:
        conv_is_1d = True
        input_tensor = input_tensor.movedim(-1, 1).contiguous()
        if len(weight.shape) != 3:
            raise ValueError("Weight tensor must be 3D if input is 3D")
        weight = weight.movedim(-1, 1).contiguous()
    else:
        input_tensor = input_tensor.movedim(-1, -3)
        if len(weight.shape) != 4:
            raise ValueError("Weight tensor must be 4D if input is nd > 3")
        weight = torch.permute(weight, (0, -1, 1, 2)).contiguous()

    nchw_out = quantized_conv_per_tensor(
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

    if conv_is_1d:
        return nchw_out.movedim(1, -1).contiguous()
    else:
        return nchw_out.movedim(-3, -1).contiguous()


def quantized_conv_variant(
    layout: str,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    is_1d: bool = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Create a quantized conv variant with type checking."""

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            stride: tuple[int, int],
            padding: tuple[int, int],
            dilation: tuple[int, int],
            groups: int,
            in_zero_point: int,
            weight_zero_point: int,
            bias_scale: float,
            output_scale: float,
            output_zero_point: int,
            out_multiplier: int,
            out_shift: int,
        ) -> torch.Tensor:
            assert (
                input_tensor.dtype == input_dtype
            ), f"Expected input dtype {input_dtype}, got {input_tensor.dtype}"
            assert (
                weight.dtype == weight_dtype
            ), f"Expected weight dtype {weight_dtype}, got {weight.dtype}"

            assert (
                bias.dtype == torch.int32
            ), f"Expected bias dtype int32, got {bias.dtype}"

            if is_1d:
                assert (
                    len(input_tensor.shape) == 3
                ), f"1D convolution requires 3D input tensor, got {len(input_tensor.shape)}D"
                assert (
                    len(weight.shape) == 3
                ), f"1D convolution requires 3D weight tensor, got {len(weight.shape)}D"

            # Call the appropriate base function
            match layout:
                case "nchw":
                    return quantized_conv2d_nchw_per_tensor(
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
                case "nhwc":
                    return quantized_conv2d_nhwc_per_tensor(
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
                case _:
                    raise ValueError(f"Unknown layout {layout}")

        return variant

    return decorator


@impl(m, "quantized_conv2d_nchw_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nchw_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nhwc_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nhwc_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor() -> (
    torch.Tensor
): ...


@impl(m, "quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor() -> (
    torch.Tensor
): ...


@impl(m, "quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor() -> (
    torch.Tensor
): ...


@impl(m, "quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor() -> (
    torch.Tensor
): ...


@impl(m, "quantized_conv1d_ncl_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8, is_1d=True)
def quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv1d_ncl_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8, is_1d=True)
def quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv1d_nlc_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8, is_1d=True)
def quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_conv1d_nlc_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8, is_1d=True)
def quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


def quantized_relu_common(
    X: torch.Tensor,
    X_zero_point: torch.Tensor | int,
    out_zero_point: int,
    out_multiplier: int,
    out_shift: int,
) -> torch.Tensor:
    """
    Quantized ReLU operation followed by requantization.

    Args:
        - X (Tensor): The input tensor
        - X_zero_point (Tensor): The quantized mapping of zero for the input
        - out_zero_point (int): The quantized mapping of zero for the output
        - out_multiplier (Tensor): The multiplier used to scale the output
        - out_shift (Tensor): The shift used to scale the output
    """
    supported_dtypes = [torch.int8, torch.int16, torch.uint8, torch.uint16]
    if X.dtype not in supported_dtypes:
        raise ValueError(f"X dtype must be one of {supported_dtypes}. Got {X.dtype}")

    out_scale = 1.0 / (-out_multiplier * (1 / (1 << 31)) * (2**out_shift))
    dequantized_X = torch.where(
        X > X_zero_point, X - X_zero_point, torch.zeros_like(X)
    ).to(torch.float32)
    return quantize_per_tensor(
        dequantized_X,
        out_scale,
        out_zero_point,
        torch.iinfo(X.dtype).min,
        torch.iinfo(X.dtype).max,
        X.dtype,
    )


def quantized_relu_variant(
    per_tensor: bool,
    dtype: torch.dtype | None = None,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Create a quantized relu variant with type checking."""

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            X: torch.Tensor,
            X_zero_point: torch.Tensor | int,
            out_zero_point: int,
            out_multiplier: torch.Tensor | int,
            out_shift: torch.Tensor | int,
        ) -> torch.Tensor:
            if per_tensor:
                if dtype and X.dtype != dtype:
                    raise ValueError(f"X dtype must be {dtype}. Got {X.dtype}")

                assert isinstance(out_shift, int)
                assert isinstance(out_multiplier, int)
                _out_shift = out_shift
                _out_multiplier = out_multiplier
            else:
                assert isinstance(out_multiplier, torch.Tensor)
                if out_multiplier.numel() > 1:
                    raise ValueError("Only scalar out_multiplier is supported")

                assert isinstance(out_shift, torch.Tensor)
                if out_shift.numel() > 1:
                    raise ValueError("Only scalar out_shift is supported")

                assert isinstance(X_zero_point, torch.Tensor)
                if X_zero_point.shape != X.shape:
                    raise ValueError(
                        f"X_zero_point shape must be {X.shape}. Got {X_zero_point.shape}"
                    )

                _out_multiplier = int(out_multiplier.item())
                _out_shift = int(out_shift.item())

            return quantized_relu_common(
                X,
                X_zero_point,
                out_zero_point,
                _out_multiplier,
                _out_shift,
            )

        return variant

    return decorator


@impl(m, "quantized_relu")
@quantized_relu_variant(False)
def quantized_relu() -> torch.Tensor: ...


@impl(m, "quantized_relu.per_tensor")
@quantized_relu_variant(True)
def quantized_relu_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_relu_asym8s_asym8s.per_tensor")
@quantized_relu_variant(True, torch.int8)
def quantized_relu_asym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl(m, "quantized_relu_asym8u_asym8u.per_tensor")
@quantized_relu_variant(True, torch.uint8)
def quantized_relu_asym8u_asym8u_per_tensor() -> torch.Tensor: ...


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


@impl(m, "rms_norm")
def rms_norm(
    X: torch.Tensor,
    normalized_shape: tuple[int],
    W: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return W * nn.RMSNorm(list(normalized_shape), eps=eps, dtype=X.dtype)(X)


@impl(m, "where_Scalar")
def where_Scalar(
    condition: torch.Tensor,
    if_true: float,
    if_false: float,
) -> torch.Tensor:
    if condition.dtype != torch.bool:
        raise ValueError("condition must be a bool tensor")

    return torch.where(condition, if_true, if_false)
