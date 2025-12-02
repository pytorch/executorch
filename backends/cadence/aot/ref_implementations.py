# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Protocol, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir.scalar_type import ScalarType
from torch.library import impl, Library

m = Library("cadence", "IMPL", "CompositeExplicitAutograd")
torch.ops.load_library("//executorch/kernels/quantized:custom_ops_generated_lib")

# Registry to track all ops with reference implementations
_REGISTERED_REF_IMPLEMENTATIONS: set[str] = set()

T = TypeVar("T", bound=Callable[..., torch.Tensor | tuple[torch.Tensor, ...]])


class MyDecorator(Protocol):
    def __call__(self, __f: T) -> T: ...


# Custom impl wrapper that tracks registrations
def impl_tracked(lib: Library, op_name: str) -> MyDecorator:
    """Wrapper around impl that tracks registered ops."""
    _REGISTERED_REF_IMPLEMENTATIONS.add(op_name)
    return impl(lib, op_name)


def get_registered_ref_implementations() -> set[str]:
    """Get all ops that have reference implementations."""
    return _REGISTERED_REF_IMPLEMENTATIONS.copy()


qdtype_map: dict[ScalarType, torch.dtype] = {
    ScalarType.QINT8: torch.qint8,
    ScalarType.QUINT8: torch.quint8,
    ScalarType.QINT32: torch.qint32,
}


def quantize_per_tensor_common(
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
            f"Unsupported dtype to quantize to {dtype}. Supported dtypes must be one of {supported_quant_types}"
        )

    return torch.ops.quantized_decomposed.quantize_per_tensor(
        input_tensor,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
    )


def quantize_per_tensor_variant(
    dtype: torch.dtype | None = None,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Create a quantize_per_tensor variant with type checking."""

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            input_tensor: torch.Tensor,
            scale: float,
            zero_point: int,
            quant_min: int,
            quant_max: int,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            if dtype and out_dtype != dtype:
                raise ValueError(f"dtype must be {dtype}. Got {out_dtype}")

            return quantize_per_tensor_common(
                input_tensor,
                scale,
                zero_point,
                quant_min,
                quant_max,
                out_dtype,
            )

        return variant

    return decorator


@impl_tracked(m, "quantize_per_tensor")
@quantize_per_tensor_variant()
def quantize_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantize_per_tensor_asym8u")
@quantize_per_tensor_variant(torch.uint8)
def quantize_per_tensor_asym8u() -> torch.Tensor: ...


@impl_tracked(m, "quantize_per_tensor_asym8s")
@quantize_per_tensor_variant(torch.int8)
def quantize_per_tensor_asym8s() -> torch.Tensor: ...


@impl_tracked(m, "quantize_per_tensor_asym16u")
@quantize_per_tensor_variant(torch.uint16)
def quantize_per_tensor_asym16u() -> torch.Tensor: ...


@impl_tracked(m, "quantize_per_tensor_asym16s")
@quantize_per_tensor_variant(torch.int16)
def quantize_per_tensor_asym16s() -> torch.Tensor: ...


@impl_tracked(m, "quantize_per_tensor_asym32s")
@quantize_per_tensor_variant(torch.int32)
def quantize_per_tensor_asym32s() -> torch.Tensor: ...


def dequantize_per_tensor_common(
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

    return torch.ops.quantized_decomposed.dequantize_per_tensor(
        input_tensor, scale, zero_point, quant_min, quant_max, dtype
    )


def dequantize_per_tensor_variant(
    dtype: torch.dtype | None = None,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Create a dequantize_per_tensor variant with type checking."""

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            input_tensor: torch.Tensor,
            scale: float,
            zero_point: int,
            quant_min: int,
            quant_max: int,
            in_dtype: torch.dtype,
        ) -> torch.Tensor:
            if dtype and in_dtype != dtype:
                raise ValueError(f"dtype must be {dtype}. Got {in_dtype}")

            return dequantize_per_tensor_common(
                input_tensor,
                scale,
                zero_point,
                quant_min,
                quant_max,
                in_dtype,
            )

        return variant

    return decorator


@impl_tracked(m, "dequantize_per_tensor")
@dequantize_per_tensor_variant()
def dequantize_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "dequantize_per_tensor_asym8u")
@dequantize_per_tensor_variant(torch.uint8)
def dequantize_per_tensor_asym8u() -> torch.Tensor: ...


@impl_tracked(m, "dequantize_per_tensor_asym32s")
@dequantize_per_tensor_variant(torch.int32)
def dequantize_per_tensor_asym32s() -> torch.Tensor: ...


@impl_tracked(m, "dequantize_per_tensor_asym16u")
@dequantize_per_tensor_variant(torch.uint16)
def dequantize_per_tensor_asym16u() -> torch.Tensor: ...


@impl_tracked(m, "dequantize_per_tensor_asym8s")
@dequantize_per_tensor_variant(torch.int8)
def dequantize_per_tensor_asym8s() -> torch.Tensor: ...


@impl_tracked(m, "dequantize_per_tensor_asym16s")
@dequantize_per_tensor_variant(torch.int16)
def dequantize_per_tensor_asym16s() -> torch.Tensor: ...


@impl_tracked(m, "quantized_add.per_tensor")
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


@impl_tracked(m, "quantized_add")
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
    return quantized_add_per_tensor(
        X,
        float(X_scale.item()),
        int(X_zero_point.item()),
        Y,
        float(Y_scale.item()),
        int(Y_zero_point.item()),
        out_scale,
        out_zero_point,
    )


@impl_tracked(m, "quantized_add_asym8sxasym8s_asym8s.per_tensor")
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


@impl_tracked(m, "quantized_add_asym8uxasym8u_asym8u.per_tensor")
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
    supported_dtypes = [torch.int8, torch.uint8, torch.int16, torch.int32]
    if dtype not in supported_dtypes:
        raise ValueError(
            f"Unsupported dtype to quantize to {dtype}. Supported dtypes must be one of {supported_dtypes}"
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

                if out_shift.dtype != torch.int32:
                    raise ValueError("out_shift must be an int32")

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


@impl_tracked(m, "quantized_linear")
@quantized_linear_variant(False, False)
def quantized_linear() -> torch.Tensor: ...


@impl_tracked(m, "quantized_linear.per_tensor")
@quantized_linear_variant(True, False)
def quantized_linear_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_linear_asym8sxasym8s_asym8s.per_tensor")
@quantized_linear_variant(True, False, torch.int8, torch.int8)
def quantized_linear_asym8sxasym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_linear_asym8uxasym8u_asym8u.per_tensor")
@quantized_linear_variant(True, False, torch.uint8, torch.uint8)
def quantized_linear_asym8uxasym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_fully_connected")
@quantized_linear_variant(False, True)
def quantized_fully_connected() -> torch.Tensor: ...


@impl_tracked(m, "quantized_fully_connected.per_tensor")
@quantized_linear_variant(True, True)
def quantized_fully_connected_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor")
@quantized_linear_variant(True, True, torch.int8, torch.int8)
def quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor")
@quantized_linear_variant(True, True, torch.uint8, torch.uint8)
def quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "fully_connected")
def fully_connected(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if input_tensor.shape[0] != 1:
        raise ValueError("Fully connected linear only supports batch size of 1")
    return F.linear(input_tensor, weight, bias)


@impl_tracked(m, "quantized_matmul")
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
        - transposed (bool): Whether Y is transposed.
    """
    if bias is not None and not torch.all(bias == 0):
        raise ValueError("bias must be None or all zeros since unused in out variant")

    if transposed:
        Y = Y.transpose(-1, -2)

    out_scale = 1.0 / (-out_multiplier * (1 / (1 << 31)) * (2**out_shift))

    out = torch.matmul(
        (X - X_zero_point).float(),
        (Y - Y_zero_point).float(),
    )
    return quantize_per_tensor(
        out,
        out_scale,
        out_zero_point,
        torch.iinfo(X.dtype).min,
        torch.iinfo(X.dtype).max,
        X.dtype,
    )


@impl_tracked(m, "quantized_matmul_asym8sxasym8s_asym8s")
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


@impl_tracked(m, "quantized_matmul_asym8uxasym8u_asym8u")
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


@impl_tracked(m, "quantized_layer_norm.per_tensor")
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
    assert isinstance(float_input_tensor, torch.Tensor)
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


@impl_tracked(m, "quantized_layer_norm")
def quantized_layer_norm(
    input_tensor: torch.Tensor,
    X_scale: torch.Tensor,
    X_zero_point: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_scale: float,
    output_zero_point: int,
) -> torch.Tensor:
    return quantized_layer_norm_per_tensor(
        input_tensor,
        float(X_scale.item()),
        int(X_zero_point.item()),
        normalized_shape,
        weight,
        bias,
        eps,
        output_scale,
        output_zero_point,
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
            stride[-1],
            padding[-1],
            dilation[-1],
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


@impl_tracked(m, "quantized_conv2d_nchw.per_tensor")
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


@impl_tracked(m, "quantized_conv2d_nchw")
def quantized_conv2d_nchw(
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
    return quantized_conv2d_nchw_per_tensor(
        input_tensor,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        int(weight_zero_point.item()),
        float(bias_scale.item()),
        output_scale,
        output_zero_point,
        int(out_multiplier.item()),
        int(out_shift.item()),
    )


@impl_tracked(m, "quantized_w8a32_conv")
def quantized_w8a32_conv(
    src: torch.Tensor,
    weight: torch.Tensor,
    w_scale: float,
    bias: torch.Tensor,
    b_scale: float,
) -> torch.Tensor:

    if len(weight.shape) != 3:
        raise ValueError("Weight tensor must be 3D")

    kernel_size, out_channels, in_channels = weight.shape
    if kernel_size != 3:
        raise ValueError("Kernel size must be 3")
    if (out_channels % 4) != 0:
        raise ValueError("Out channels must be a multiple of 4")
    if (in_channels % 4) != 0:
        raise ValueError("In channels must be a multiple of 4")

    assert weight.dtype == torch.int8
    assert bias.dtype == torch.int8

    # To make compliant with torch (LCN -> NCL format)
    weight = weight.permute(1, 2, 0).contiguous()

    # channels last to channels first
    src = src.permute(0, 2, 1).contiguous()

    dequant_weight = weight.float() * w_scale

    # Dequantize bias using scale
    dequant_bias = bias.float() * b_scale

    # Perform 1D convolution
    # src: [batch, in_channel, in_length]
    # weight: [out_ch, in_ch, kernel_dim]
    # bias: [out_ch]
    output = torch.nn.functional.conv1d(
        src.float(),
        dequant_weight,
        dequant_bias,
    )

    return output


@impl_tracked(m, "quantized_w8a32_linear")
def quantized_w8a32_linear(
    src: torch.Tensor,
    weight: torch.Tensor,
    w_scale: float,
    bias: torch.Tensor,
    b_scale: float,
) -> torch.Tensor:
    # src comes in shape [leading_dims, in_dim]
    # weight comes in shape [in_dim, out_dim]
    # output comes in empty with shape [leading_dims, out_dim]
    assert weight.dtype == torch.int8
    assert bias.dtype == torch.int8
    if len(src.shape) >= 2:
        assert src.shape[-2] == 1, "Only supporting vector-matrix multiplication"

    # need to transpose to make compliant with torch linear (in, out -> out, in)
    weight = weight.transpose(1, 0).contiguous()
    dequant_weight = weight.float() * w_scale
    dequant_bias = bias.float() * b_scale

    output = torch.nn.functional.linear(
        src.float(),
        dequant_weight,
        dequant_bias,
    )

    return output


@impl_tracked(m, "quantized_w8a32_gru")
def quantized_w8a32_gru(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    weights_inputs: torch.Tensor,
    w_i_scale: float,
    weights_hidden: torch.Tensor,
    w_h_scale: float,
    bias_inputs: torch.Tensor,
    b_i_scale: float,
    bias_hidden: torch.Tensor,
    b_h_scale: float,
) -> torch.Tensor:
    assert weights_inputs.dtype == torch.int8
    assert weights_hidden.dtype == torch.int8
    assert bias_inputs.dtype == torch.int8
    assert bias_hidden.dtype == torch.int8
    assert inputs.dtype == torch.float32
    assert hidden.dtype == torch.float32

    if len(hidden.shape) > 2:
        raise ValueError("Hidden state must be 2D or 1D")

    if len(hidden.shape) == 2 and hidden.shape[0] != 1:
        raise ValueError("Leading dimension of hidden state must be 1")

    original_hidden_shape = hidden.shape
    hidden = hidden.view(-1)

    hidden_dim = hidden.shape[0]
    if (hidden_dim % 4) != 0:
        raise ValueError(
            "Hidden dimension must be a multiple of 4 for HiFi SIMD operations"
        )

    dequant_weights_inputs = weights_inputs.float() * w_i_scale
    dequant_weights_hidden = weights_hidden.float() * w_h_scale

    # C++ implementation averages the two bias scales
    avg_bias_scale = (b_i_scale + b_h_scale) / 2
    dequant_bias_inputs = bias_inputs.float() * avg_bias_scale
    dequant_bias_hidden = bias_hidden.float() * avg_bias_scale

    gi = F.linear(inputs, dequant_weights_inputs, dequant_bias_inputs)
    gh = F.linear(hidden, dequant_weights_hidden, dequant_bias_hidden)

    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)

    reset_gate = torch.sigmoid(i_r + h_r)
    update_gate = torch.sigmoid(i_z + h_z)
    new_gate = torch.tanh(i_n + reset_gate * h_n)

    new_hidden = (1 - update_gate) * new_gate + update_gate * hidden

    if new_hidden.shape[0] != 1:
        raise ValueError("Leading dimension of hidden state must be 1")

    assert new_hidden.shape == original_hidden_shape

    new_hidden = new_hidden.view(-1)
    return torch.stack([new_hidden, new_hidden], dim=0)


@impl_tracked(m, "quantized_conv2d_nhwc.per_tensor")
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


@impl_tracked(m, "quantized_conv2d_nhwc")
def quantized_conv2d_nhwc(
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
    return quantized_conv2d_nhwc_per_tensor(
        input_tensor,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        int(weight_zero_point.item()),
        float(bias_scale.item()),
        output_scale,
        output_zero_point,
        int(out_multiplier.item()),
        int(out_shift.item()),
    )


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


@impl_tracked(m, "quantized_conv2d_nchw_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nchw_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nhwc_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nhwc_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8)
def quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor() -> (
    torch.Tensor
): ...


@impl_tracked(m, "quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8)
def quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor() -> (
    torch.Tensor
): ...


@impl_tracked(m, "quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8)
def quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor() -> (
    torch.Tensor
): ...


@impl_tracked(m, "quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8)
def quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor() -> (
    torch.Tensor
): ...


@impl_tracked(m, "quantized_conv1d_ncl_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nchw", torch.int8, torch.int8, is_1d=True)
def quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv1d_ncl_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nchw", torch.uint8, torch.uint8, is_1d=True)
def quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv1d_nlc_asym8sxsym8s_asym8s.per_tensor")
@quantized_conv_variant("nhwc", torch.int8, torch.int8, is_1d=True)
def quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_conv1d_nlc_asym8uxsym8u_asym8u.per_tensor")
@quantized_conv_variant("nhwc", torch.uint8, torch.uint8, is_1d=True)
def quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "conv1d")
def conv1d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    groups: int,
) -> torch.Tensor:
    conv_out = torch.nn.functional.conv1d(
        input_tensor, weight, bias, stride[0], padding[0], dilation[0], groups
    )

    return conv_out


@impl_tracked(m, "conv2d")
def conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    conv_out = torch.nn.functional.conv2d(
        input_tensor, weight, bias, stride, padding, dilation, groups
    )

    return conv_out


@impl_tracked(m, "conv3d")
def conv3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    conv_out = torch.nn.functional.conv3d(
        input_tensor, weight, bias, stride, padding, dilation, groups
    )

    return conv_out


@impl_tracked(m, "transposed_convolution")
def transposed_convolution(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    output_padding: tuple[int, int],
    groups: int,
    channel_last: bool = False,
) -> torch.Tensor:

    conv_is_1d = len(input_tensor.shape) == 3
    if channel_last:
        if conv_is_1d:
            input_tensor = input_tensor.movedim(-1, 1).contiguous()
            if len(weight.shape) != 3:
                raise ValueError("Weight tensor must be 3D if input is 3D")
            weight = weight.movedim(-1, 1).contiguous()
        else:
            input_tensor = input_tensor.movedim(-1, -3)
            if len(weight.shape) != 4:
                raise ValueError("Weight tensor must be 4D if input is nd > 3")
            weight = torch.permute(weight, (0, -1, 1, 2)).contiguous()

    _stride: tuple[int, int] | int = stride
    _padding: tuple[int, int] | int = padding
    _dilation: tuple[int, int] | int = dilation
    _output_padding: tuple[int, int] | int = output_padding
    if conv_is_1d:
        conv = torch.nn.functional.conv_transpose1d
        _stride = stride[0]
        _padding = padding[0]
        _dilation = dilation[0]
        _output_padding = output_padding[0]
    else:
        conv = torch.nn.functional.conv_transpose2d

    conv_out = conv(
        input_tensor,
        weight,
        bias,
        _stride,
        _padding,
        _output_padding,
        groups,
        _dilation,
    )
    if channel_last:
        if conv_is_1d:
            conv_out = conv_out.movedim(1, -1).contiguous()
        else:
            conv_out = conv_out.movedim(-3, -1).contiguous()

    return conv_out


@impl_tracked(m, "avg_pool2d")
def avg_pool2d(
    input_tensor: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    divisor_override: int | None = None,
    in_zero_point: torch.Tensor | None = None,
    channel_last: bool = False,
) -> torch.Tensor:
    if channel_last:
        raise NotImplementedError("Channel last is not yet supported for avg_pool2d")

    in_dtype = input_tensor.dtype
    pad_h, pad_w = padding
    if in_zero_point is not None:
        # Avg pool2d does not allow non-0 padding,
        # so we manually pad the input
        pad_value = in_zero_point.item()
        if not count_include_pad:
            # To simulate this, just pad with 0s
            pad_value = 0

        input_tensor = torch.nn.functional.pad(
            input_tensor,
            (pad_w, pad_w, pad_h, pad_h),
            mode="constant",
            value=pad_value,
        ).float()

        padding = (0, 0)

    out = torch.nn.functional.avg_pool2d(
        input_tensor,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )

    if in_zero_point is not None:
        min_val = torch.iinfo(in_dtype).min
        max_val = torch.iinfo(in_dtype).max
        out = torch.clamp(torch.round(out), min_val, max_val)

    return out.to(in_dtype)


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
    out = quantize_per_tensor(
        dequantized_X,
        out_scale,
        out_zero_point,
        torch.iinfo(X.dtype).min,
        torch.iinfo(X.dtype).max,
        X.dtype,
    )
    assert isinstance(out, torch.Tensor)
    return out


def quantized_relu_variant(
    dtype: torch.dtype | None = None,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Create a quantized relu variant with type checking."""

    def decorator(_: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def variant(
            X: torch.Tensor,
            X_zero_point: int,
            out_zero_point: int,
            out_multiplier: int,
            out_shift: int,
        ) -> torch.Tensor:
            if dtype and X.dtype != dtype:
                raise ValueError(f"X dtype must be {dtype}. Got {X.dtype}")

            return quantized_relu_common(
                X,
                X_zero_point,
                out_zero_point,
                out_multiplier,
                out_shift,
            )

        return variant

    return decorator


@impl_tracked(m, "quantized_relu.per_tensor")
@quantized_relu_variant()
def quantized_relu_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_relu_asym8s_asym8s.per_tensor")
@quantized_relu_variant(torch.int8)
def quantized_relu_asym8s_asym8s_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_relu_asym8u_asym8u.per_tensor")
@quantized_relu_variant(torch.uint8)
def quantized_relu_asym8u_asym8u_per_tensor() -> torch.Tensor: ...


@impl_tracked(m, "quantized_relu")
def quantized_relu(
    X: torch.Tensor,
    X_zero_point: torch.Tensor,
    out_zero_point: int,
    out_multiplier: torch.Tensor,
    out_shift: torch.Tensor,
) -> torch.Tensor:
    return quantized_relu_per_tensor(
        X, X_zero_point.item(), out_zero_point, out_multiplier.item(), out_shift.item()
    )


@impl_tracked(m, "requantize.per_tensor")
def requantize_per_tensor(
    input: torch.Tensor,
    in_scale: float,
    in_zero_point: int,
    out_scale: float,
    out_zero_point: int,
    dtype: ScalarType,
) -> torch.Tensor:
    if dtype in qdtype_map:
        # Old quantization mechanism
        return torch.quantize_per_tensor(
            torch.dequantize(input), out_scale, out_zero_point, qdtype_map[dtype]
        )

    quant_min = torch.iinfo(input.dtype).min
    quant_max = torch.iinfo(input.dtype).max
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_min = torch.iinfo(dtype).min
    # pyre-fixme[6]: This dtype is actually the right one.
    out_quant_max = torch.iinfo(dtype).max
    return torch.ops.quantized_decomposed.quantize_per_tensor(
        torch.ops.quantized_decomposed.dequantize_per_tensor(
            input,
            in_scale,
            in_zero_point,
            quant_min,
            quant_max,
            input.dtype,
        ),
        out_scale,
        out_zero_point,
        out_quant_min,
        out_quant_max,
        dtype,
    )


@impl_tracked(m, "requantize")
def requantize(
    input_tensor: torch.Tensor,
    in_scale: torch.Tensor,
    in_zero_point: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    dtype: ScalarType,
) -> torch.Tensor:
    return requantize_per_tensor(
        input_tensor,
        float(in_scale.item()),
        int(in_zero_point.item()),
        float(out_scale.item()),
        int(out_zero_point.item()),
        dtype,
    )


@impl_tracked(m, "rms_norm")
def rms_norm(
    X: torch.Tensor,
    normalized_shape: tuple[int],
    W: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return W * nn.RMSNorm(list(normalized_shape), eps=eps, dtype=X.dtype)(X)


@impl_tracked(m, "where_Scalar")
def where_Scalar(
    condition: torch.Tensor,
    if_true: float,
    if_false: float,
) -> torch.Tensor:
    if condition.dtype != torch.bool:
        raise ValueError("condition must be a bool tensor")

    return torch.where(condition, if_true, if_false)


@impl_tracked(m, "rope")
def rope(
    input_tensor: torch.Tensor,
    sin_tensor: torch.Tensor,
    cos_tensor: torch.Tensor,
    pos: torch.Tensor | None,
) -> torch.Tensor:
    original_shape = input_tensor.shape

    if len(original_shape) not in [4, 5]:
        raise ValueError(
            f"Input tensor must be 4D or 5D. Got {len(original_shape)}D tensor"
        )
    if original_shape[0] != 1:
        raise ValueError("Input tensor must have batch size 1")
    if len(original_shape) == 5:
        input_tensor = input_tensor.view(
            input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], -1
        )

    _, s, h, hd = input_tensor.shape

    if hd % 2:
        raise ValueError("Hidden dimension must be divisible by 2")

    if sin_tensor.shape != (s, hd // 2) or cos_tensor.shape != (s, hd // 2):
        raise ValueError(
            f"sin_tensor and cos_tensor must have shape {s, hd // 2}. Got {sin_tensor.shape} and {cos_tensor.shape}"
        )

    if pos is not None:
        if pos.shape != (input_tensor.shape[1],):
            raise ValueError(
                f"pos must have shape {input_tensor.shape[1]}. Got {pos.shape}"
            )
        sin_tensor = sin_tensor[pos]
        cos_tensor = cos_tensor[pos]

    sin_tensor = sin_tensor.unsqueeze(1)
    cos_tensor = cos_tensor.unsqueeze(1)

    x0, x1 = input_tensor[..., ::2], input_tensor[..., 1::2]
    rotated = torch.cat(
        [x0 * cos_tensor - x1 * sin_tensor, x0 * sin_tensor + x1 * cos_tensor], dim=-1
    )
    return rotated.view(original_shape)


@impl_tracked(m, "im2row")
def im2row(
    input_tensor: torch.Tensor,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int],
    stride: tuple[int, int],
    in_zero_point: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    """
    Converts an input tensor into a 2D matrix where each row is a flattened sliding window (patch)
    from the input, suitable for use in convolution as a matrix multiplication (im2row).

    Args:
        - input_tensor: Input tensor of shape (N, C, H, W) or (N, H, W, C) if channel_last.
        - kernel_size: Size of the convolution kernel.
        - dilation: Dilation of the convolution kernel.
        - padding: Padding to apply to the input.
        - stride: Stride of the convolution.
        - in_zero_point : Zero point for input quantization (broadcastable to input).
        - channel_last: If True, input is in NHWC format, else NCHW.

    Returns:
        - Tensor of shape (N, num_patches, patch_size)
    """
    if len(input_tensor.shape) == 3:
        height_dim = 1 if channel_last else 2
        input_tensor = input_tensor.unsqueeze(height_dim)

    if in_zero_point is not None:
        if in_zero_point.numel() != 1 and in_zero_point.shape != (
            input_tensor.shape[0],
        ):
            raise ValueError(
                f"Input zero point must be a scalar or broadcastable to input shape {input_tensor.shape}"
            )
        if in_zero_point.dtype != torch.int32:
            raise ValueError("Input zero point must be an int32 tensor")

    if channel_last:
        input_tensor = input_tensor.movedim(-1, -3).contiguous()  # NHWC -> NCHW

    N, C, H, W = input_tensor.shape
    kH, kW = kernel_size
    dH, dW = dilation
    pH, pW = padding
    sH, sW = stride

    # Handle padding with zero point values
    if in_zero_point is not None and (pH > 0 or pW > 0):
        # Expand zero point to (N, 1, 1, 1) for broadcasting
        in_zero_point = in_zero_point.expand(N)

        # Pad input with the per-batch zero point values
        input_tensor = torch.stack(
            [
                torch.nn.functional.pad(
                    input_tensor[i],
                    (pW, pW, pH, pH),
                    mode="constant",
                    value=in_zero_point[i].item(),
                )
                for i in range(len(input_tensor))
            ]
        )

        padding = (0, 0)  # Already padded manually

    # Use unfold to extract sliding local blocks
    # Unfold: (N, C, H, W) -> (N, C, L, kH, kW), where L = number of sliding windows
    # torch.nn.functional.unfold returns (N, C*kH*kW, L)
    patches = torch.nn.functional.unfold(
        input_tensor.float(),  # unfold not implemented for int
        kernel_size=(kH, kW),
        dilation=(dH, dW),
        padding=padding,
        stride=(sH, sW),
    ).to(
        input_tensor.dtype
    )  # (N, C*kH*kW, L)

    # Transpose to (N, L, C*kH*kW)
    patches = patches.transpose(1, 2).contiguous()

    # Reshape to (N*L, C*kH*kW)
    patches = patches.view(N, -1, C * kH * kW)

    # If channel_last, output should be in NHWC patch order (but im2row is always row-major)
    return patches


@impl_tracked(m, "im2row.per_tensor")
def im2row_per_tensor(
    input_tensor: torch.Tensor,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int],
    stride: tuple[int, int],
    in_zero_point: int,
    channel_last: bool = False,
) -> torch.Tensor:
    out = im2row(
        input_tensor,
        kernel_size,
        dilation,
        padding,
        stride,
        torch.tensor(in_zero_point, dtype=torch.int32),
        channel_last,
    )
    assert isinstance(out, torch.Tensor)
    return out


@impl_tracked(m, "transposed_im2row")
def transposed_im2row(
    input_tensor: torch.Tensor,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int],
    stride: tuple[int, int],
    output_padding: tuple[int, int],
    in_zero_point: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    """
    Converts input tensor patches into im2row format for transposed convolutions.
    This function extracts patches from input in a pattern suitable for transposed convolution.

    Args:
        - input_tensor: Input spatial tensor, NCHW or NHWC format (3D or 4D).
        - kernel_size: Size of the convolution kernel.
        - dilation: Dilation of the convolution kernel.
        - padding: Padding to apply to the input.
        - stride: Stride of the convolution.
        - output_padding: Additional output padding for transposed convolution.
        - in_zero_point: Zero point for input quantization (broadcastable to input).
        - channel_last: If True, input is in NHWC format, else NCHW.

    Returns:
        - 3D tensor of shape (N, output_h * output_w, kernel_h * kernel_w * in_c)
    """
    # Handle 1D convolution case by adding height dimension
    if len(input_tensor.shape) == 3:
        height_dim = 1 if channel_last else 2
        input_tensor = input_tensor.unsqueeze(height_dim)

    if in_zero_point is not None:
        if in_zero_point.dtype != torch.int32:
            raise ValueError("Input zero point must be an int32 tensor")

    # Move to NCHW for processing if needed
    if channel_last:
        input_tensor = input_tensor.movedim(-1, -3).contiguous()  # NHWC -> NCHW

    N, C, H_in, W_in = input_tensor.shape

    # Output: (N, C*H_in*W_in, H_out, W_out)
    H_out = (
        (H_in - 1) * stride[0]
        + kernel_size[0]
        + output_padding[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
    )
    W_out = (
        (W_in - 1) * stride[1]
        + kernel_size[1]
        + output_padding[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
    )

    # For each input pixel, create a channel where the upsampled (transposed conv) patch is placed
    # Output: (N, C*H_in*W_in, H_out, W_out)
    inp_flat = input_tensor.reshape(N, C * H_in * W_in)

    # Calculate output spatial size
    H_out = (
        (H_in - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    W_out = (
        (W_in - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )

    # Compute the upsampled (top-left) position for each input pixel
    h_idx = torch.arange(H_in, device=input_tensor.device)
    w_idx = torch.arange(W_in, device=input_tensor.device)
    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing="ij")
    out_h_idx = grid_h * stride[0] - padding[0]
    out_w_idx = grid_w * stride[1] - padding[1]

    # Compute all input pixel positions (flattened)
    ch_idx = torch.arange(C * H_in * W_in, device=input_tensor.device)
    ij_idx = ch_idx % (H_in * W_in)
    i_idx = ij_idx // W_in
    j_idx = ij_idx % W_in

    # For each input pixel, compute the output positions for the kernel window
    kh_idx = torch.arange(kernel_size[0], device=input_tensor.device)
    kw_idx = torch.arange(kernel_size[1], device=input_tensor.device)
    kh_grid, kw_grid = torch.meshgrid(kh_idx, kw_idx, indexing="ij")
    kh_grid = kh_grid.reshape(-1)
    kw_grid = kw_grid.reshape(-1)
    num_kernel = kernel_size[0] * kernel_size[1]

    # Broadcast to all channels and kernel positions
    ch_idx_b = ch_idx.repeat_interleave(num_kernel)
    n_kernel = ch_idx.shape[0] * num_kernel

    i_idx_b = i_idx.repeat_interleave(num_kernel)
    j_idx_b = j_idx.repeat_interleave(num_kernel)
    kh_b = kh_grid.repeat(ch_idx.shape[0])
    kw_b = kw_grid.repeat(ch_idx.shape[0])

    h_out = out_h_idx[i_idx_b, j_idx_b] + kh_b * dilation[0]
    w_out = out_w_idx[i_idx_b, j_idx_b] + kw_b * dilation[1]

    # Mask for valid output positions
    valid = (h_out >= 0) & (h_out < H_out) & (w_out >= 0) & (w_out < W_out)

    # Prepare indices for advanced indexing
    n_idx = (
        torch.arange(N, device=input_tensor.device)
        .view(-1, 1)
        .expand(N, n_kernel)
        .reshape(-1)
    )
    ch_idx_full = ch_idx_b.expand(N, n_kernel).reshape(-1)
    h_out_full = h_out.expand(N, n_kernel).reshape(-1)
    w_out_full = w_out.expand(N, n_kernel).reshape(-1)
    valid_full = valid.expand(N, n_kernel).reshape(-1)

    # Gather input values for each channel
    inp_vals = inp_flat[:, ch_idx_b].reshape(-1)

    # Create output tensor
    patches = torch.zeros((N, C * H_in * W_in, H_out, W_out), dtype=input_tensor.dtype)

    # If in_zero_point is provided, fill patches with it
    if in_zero_point is not None:
        if in_zero_point.numel() == 1:
            patches.fill_(in_zero_point.item())
        else:
            # Broadcast in_zero_point to (N, C, H_in, W_in)
            assert in_zero_point.shape == (N,)
            in_zero_point = in_zero_point.view(N, 1, 1, 1)
            patches = patches + in_zero_point

    # Scatter input values to output positions (only valid positions)
    patches[
        n_idx[valid_full],
        ch_idx_full[valid_full],
        h_out_full[valid_full],
        w_out_full[valid_full],
    ] = inp_vals[valid_full]

    # Optionally, flatten to (N, num_patches, patch_size) if needed
    patches = patches.view(N, C * H_in * W_in, -1).transpose(1, 2).contiguous()
    return patches


@impl_tracked(m, "quantized_embedding_byte")
def quantized_embedding_byte(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: torch.Tensor | None,
    indices: torch.Tensor,
    pruned_weights: bool = False,
) -> torch.Tensor:
    if pruned_weights:
        raise NotImplementedError("Pruned weights not supported")

    # Cannot use torch.ops.quantized_decomposed.embedding_byte.dtype because
    # it doesn't support num_groups == 1
    num_groups = 1
    if len(weight_scales.shape) == 2:
        num_groups = weight_scales.shape[1]

    group_size = weight.shape[1] // num_groups
    weight = torch.ops.torchao.dequantize_affine.default(
        input=weight,
        block_size=(1, group_size),
        scale=weight_scales,
        zero_point=weight_zero_points,
        input_dtype=weight.dtype,
        quant_min=torch.iinfo(weight.dtype).min,
        quant_max=torch.iinfo(weight.dtype).max,
    )

    return weight[indices]


@impl_tracked(m, "idma_copy")
def idma_copy(src: torch.Tensor, task_num: int = 0, channel: int = 0) -> torch.Tensor:
    return src.clone()


@impl_tracked(m, "idma_store")
def idma_store(src: torch.Tensor, task_num: int = 0, channel: int = 0) -> torch.Tensor:
    return src.clone()


@impl_tracked(m, "idma_load")
def idma_load(src: torch.Tensor, task_num: int = 0, channel: int = 0) -> torch.Tensor:
    return src.clone()


@impl_tracked(m, "idma_wait")
def idma_wait(src: torch.Tensor, task_num: int = 0, channel: int = 0) -> torch.Tensor:
    return src.clone()


@impl_tracked(m, "linalg_svd")
def linalg_svd(
    A: torch.Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert compute_uv
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)
    return U.contiguous(), S.contiguous(), Vh.contiguous()


@impl_tracked(m, "_softmax_f32_f32")
def softmax_f32_f32(
    input_tensor: torch.Tensor,
    dim: int,
    half_to_float: bool | None = None,
) -> torch.Tensor:
    assert input_tensor.dtype == torch.float32, "input_tensor must be float32"
    assert not half_to_float, "half_to_float is not supported"
    return torch.nn.functional.softmax(input_tensor, dim=dim, dtype=torch.float32)


def quantized_softmax_per_tensor_common(
    input_tensor: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int,
    in_scale: float,
    in_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    """
    Quantized softmax operation.

    Args:
        - input_tensor (Tensor): The quantized input tensor
        - mask (Tensor): Mask tensor
        - dim (int): The dimension along which softmax is computed
        - in_scale (float): The scale of the input quantization
        - in_zero_point (int): The zero point of the input quantization
        - out_scale (float): The scale of the output quantization
        - out_zero_point (int): The zero point of the output quantization
    """
    # TODO: T228751479 - Add support for mask parameter in softmax
    assert mask is None
    supported_dtypes = [torch.int8, torch.uint8, torch.int16]
    if input_tensor.dtype not in supported_dtypes:
        raise ValueError(
            f"Input dtype must be one of {supported_dtypes}. Got {input_tensor.dtype}"
        )

    float_input_tensor = dequantize_per_tensor(
        input_tensor,
        in_scale,
        in_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )

    softmax_output = torch.nn.functional.softmax(float_input_tensor, dim=dim)

    return quantize_per_tensor(
        softmax_output,
        out_scale,
        out_zero_point,
        torch.iinfo(input_tensor.dtype).min,
        torch.iinfo(input_tensor.dtype).max,
        input_tensor.dtype,
    )


@impl_tracked(m, "quantized_softmax.per_tensor")
def quantized_softmax_per_tensor(
    input_tensor: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int,
    in_scale: float,
    in_zero_point: int,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    return quantized_softmax_per_tensor_common(
        input_tensor,
        mask,
        dim,
        in_scale,
        in_zero_point,
        out_scale,
        out_zero_point,
    )


@impl_tracked(m, "quantized_softmax")
def quantized_softmax(
    input_tensor: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int,
    in_scale: torch.Tensor,
    in_zero_point: torch.Tensor,
    out_scale: float,
    out_zero_point: int,
) -> torch.Tensor:
    return quantized_softmax_per_tensor_common(
        input_tensor,
        mask,
        dim,
        float(in_scale.item()),
        int(in_zero_point.item()),
        out_scale,
        out_zero_point,
    )
