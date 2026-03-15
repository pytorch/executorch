from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.utils import (
    Frameworks,
    unified_dtype_converter,
)


def shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: int,
) -> TRTTensor:
    """
    This is the general shape layer implementation in TensorRT.
    sym_size.int ops map to addShape layer in TensorRT and returns
    the dynamic shape of the tensor optionally taking in a dim argument.
    """
    shape_layer = ctx.net.add_shape(input_val)
    input_shape = shape_layer.get_output(0)
    input_shape = cast_trt_tensor(
        ctx,
        input_shape,
        trt.int32,
        name + "_shape_casted",
    )
    set_layer_name(shape_layer, target, name + "_shape", source_ir)

    n_dims = len(input_val.shape)
    dim = get_positive_dim(dim, n_dims)
    dim_tensor = get_trt_tensor(ctx, dim, name + "_dim")
    gather_layer = ctx.net.add_gather(input_shape, dim_tensor, axis=0)
    set_layer_name(gather_layer, target, name + "_gather", source_ir)
    input_shape = gather_layer.get_output(0)

    return input_shape


def get_shape_with_dynamic_shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    shape: List[int] | Tuple[int, ...] | torch.Tensor,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Prepare the real output tensor shape for dynamic shape mode tensor input.
    How this functions works:
    Assuming the input_val has actual shape [2048, 256, 512], expected reduce operation
    output shape is [-1, 128, 256], this function should return [2048, 128, 256] as the actual
    reduce operation output shape. Steps of calculations are:
        1. get the actual tensor shape of input_val via add_shape layer;
        2. create a all 0 tensor [0, 0, 0];
        3. run elementwise comparison the [0, 0, 0] and [-1, 128, 256] tensor, get a condition tensor [True, False, False];
        4. use the condition tensor [True, False, False] to do selection between [2048, 256, 512] and [-1, 128, 256], replace
           all -1 dynamic shape dimensions with actual batch_size value;
        5. output shape with actual batch_size as [2048, 128, 256]

    Args:
        ctx (ConversionContext): TensorRT ConversionContext object.
        shape: calculated shape of the expected output tensor
        input_val (TRTTensor): A TensorRT ITensor.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
    Returns:
        TensorRT ITensors that represents the actual shape of the input_val
    """
    # Ger real shape info for input_val
    input_shape = ctx.net.add_shape(input_val).get_output(0)
    input_shape = cast_trt_tensor(
        ctx,
        input_shape,
        trt.int32,
        name + "_int32_casted",
    )
    # input_shape.dtype is int64 in TRT 10.0
    input_np_dtype = unified_dtype_converter(input_shape.dtype, Frameworks.NUMPY)
    scale_layer = ctx.net.add_constant(
        input_shape.shape, np.ascontiguousarray(shape, dtype=input_np_dtype)
    )
    set_layer_name(scale_layer, target, f"{name}_scale")
    scale_res = scale_layer.get_output(0)

    length = input_shape.shape[0]

    zero_layer = ctx.net.add_constant(
        input_shape.shape, np.zeros((length), dtype=np.int32)
    )

    set_layer_name(zero_layer, target, f"{name}_zeros")

    condition_val = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_shape",
        trt.ElementWiseOperation.LESS,
        scale_res,
        zero_layer.get_output(0),
    )
    select_layer = ctx.net.add_select(condition_val, input_shape, scale_res)
    set_layer_name(select_layer, target, f"{name}_select")
    return select_layer.get_output(0)
