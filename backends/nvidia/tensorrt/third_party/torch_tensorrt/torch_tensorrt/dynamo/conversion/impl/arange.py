from typing import Optional, Union

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def arange(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    start: Union[int, TRTTensor],
    end: Union[int, TRTTensor],
    step: Union[int, TRTTensor],
) -> TRTTensor:
    """
    Creates a sequence of values (arange) either dynamically or statically,
    then outputs a TensorRT tensor.

    If any of (start, end, step) is a TRT tensor, it sets up a dynamic arange
    using a Fill layer. Otherwise, it creates a static NumPy array and converts
    it into a TensorRT constant tensor.
    """
    # If any argument is a TRT tensor, use dynamic arange with a Fill layer
    if any(isinstance(x, TRTTensor) for x in (start, end, step)):
        # Convert start, end, step into TRT tensors with appropriate rank
        start_rank_0 = get_trt_tensor(ctx, start, name + "_start_rank_0", min_rank=0)
        start_rank_1 = get_trt_tensor(ctx, start, name + "_start_rank_1", min_rank=1)
        end = get_trt_tensor(ctx, end, name + "_end", min_rank=1)
        step = get_trt_tensor(ctx, step, name + "_step", min_rank=1)

        # Compute (end - start) / step to determine the output length
        shape = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", end, start_rank_1
        )
        shape = impl.elementwise.trunc_div(
            ctx, target, source_ir, name + "_shape", shape, step
        )
        shape = cast_trt_tensor(ctx, shape, end.dtype, name + "_shape_casted")

        # Build a Fill layer in LINSPACE mode
        fill_layer = ctx.net.add_fill(
            shape.shape, trt.FillOperation.LINSPACE, shape.dtype
        )
        fill_layer.set_input(0, shape)  # output length
        fill_layer.set_input(1, start_rank_0)  # start value
        fill_layer.set_input(2, step)  # step size

        return fill_layer.get_output(0)

    else:
        # All arguments are static, so use NumPy arange and create a TRT constant
        arr = np.arange(start, end, step, dtype=np.int32)
        weights = trt.Weights(arr)
        const_layer = ctx.net.add_constant(arr.shape, weights)
        set_layer_name(const_layer, target, f"{name}_arange_const", source_ir)
        return const_layer.get_output(0)
