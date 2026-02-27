from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    cast_trt_tensor,
    get_trt_tensor,
    prepend_ones,
    promote_trt_tensors_to_same_dtype,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import ne


def where(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Union[TRTTensor, np.ndarray, torch.Tensor],
    other: Union[TRTTensor, np.ndarray, torch.Tensor],
    condition: Union[TRTTensor, np.ndarray, torch.Tensor],
) -> TRTTensor:
    if not (broadcastable(input, other)):
        assert "The two torch tensors should be broadcastable"

    x_shape = list(input.shape)
    y_shape = list(other.shape)
    condition_shape = list(condition.shape)
    max_shape_len = max(len(x_shape), len(y_shape), len(condition_shape))

    if not isinstance(condition, TRTTensor):
        condition = get_trt_tensor(ctx, condition, f"{name}_condition")

    if condition.dtype != trt.bool:
        condition = cast_trt_tensor(ctx, condition, trt.float32, f"{name}_cast")
        condition = ne(ctx, target, source_ir, f"{name}_cond_zero", condition, 0)

    diff = max_shape_len - len(condition_shape)
    if diff > 0:
        condition = prepend_ones(ctx, condition, f"{name}_condition_broadcast", diff)

    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_x")
    diff = max_shape_len - len(x_shape)
    if diff > 0:
        input = prepend_ones(ctx, input, f"{name}_input_broadcast", diff)

    if not isinstance(other, TRTTensor):
        other = get_trt_tensor(ctx, other, f"{name}_y")
    diff = max_shape_len - len(y_shape)
    if diff > 0:
        other = prepend_ones(ctx, other, f"{name}_other_broadcast", diff)

    # Ensure that input and other have the same TRT dtype
    input, other = promote_trt_tensors_to_same_dtype(ctx, input, other, name)

    return select(ctx, target, source_ir, name, input, other, condition)


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    condition: TRTTensor,
) -> TRTTensor:
    select_layer = ctx.net.add_select(condition, input, other)
    set_layer_name(select_layer, target, name + "_select", source_ir)
    return select_layer.get_output(0)
