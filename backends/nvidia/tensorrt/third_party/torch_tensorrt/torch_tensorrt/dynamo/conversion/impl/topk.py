from typing import Optional, Tuple, Union

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    flatten_dims,
    get_axes_for_reduce_op,
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM


def argmax_argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    topk_option: trt.TopKOperation,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    if input.dtype == trt.int32:
        input = cast_trt_tensor(ctx, input, trt.float32, name, target, source_ir)

    # Three different cases here:
    # 1. dim == None, flatten input tensor first, keep_dim will be ignore and the output rank == input rank
    # 2. input rank == 1: TopK layer does not support 1 dimensional topk operation. Broadcast input to rank == 2
    # 3. normal cases, no additional handlings
    out = input
    is_dynamic_present = has_dynamic_shape(input.shape)

    if dim is None:
        if is_dynamic_present and len(input.shape) != 1:
            multiplier = get_trt_tensor(ctx, 1, name + "_shape")
            for i in range(0, len(input.shape)):
                if input.shape[i] != DYNAMIC_DIM:
                    multiplier = convert_binary_elementwise(
                        ctx,
                        target,
                        source_ir,
                        name + f"_shape_{i}",
                        trt.ElementWiseOperation.PROD,
                        multiplier,
                        input.shape[i],
                    )
                else:
                    multiplier = convert_binary_elementwise(
                        ctx,
                        target,
                        source_ir,
                        name + f"_shape_{i}",
                        trt.ElementWiseOperation.PROD,
                        multiplier,
                        get_shape(
                            ctx,
                            target,
                            source_ir,
                            name + f"_shape_dim_stop_{i}",
                            input,
                            i,
                        ),
                    )
            # form shape tensor
            new_shape_layer = ctx.net.add_concatenation(
                [multiplier, get_trt_tensor(ctx, 1, name + "_one_shape")]
            )
            set_layer_name(
                new_shape_layer, target, name + "_new_shape_concat", source_ir
            )
            concat_tensor = new_shape_layer.get_output(0)

            reshape_dynamic_layer = ctx.net.add_shuffle(input)
            reshape_dynamic_layer.set_input(1, concat_tensor)
            set_layer_name(
                reshape_dynamic_layer, target, name + "_reshape_layer", source_ir
            )
            out = reshape_dynamic_layer.get_output(0)

        else:
            new_shape = (*flatten_dims(input, 0, -1), 1)
            out = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_flatten", input, new_shape
            )
    elif len(input.shape) == 1:
        new_shape = (*input.shape, 1)
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_broadcast", input, new_shape
        )

    # Reduce over the flattened input if the dimension is None, otherwise the specified dimension
    reduce_mask = get_axes_for_reduce_op(
        get_positive_dim(dim if dim is not None else 0, len(out.shape))
    )

    topk_layer = ctx.net.add_topk(out, topk_option, 1, reduce_mask)
    set_layer_name(topk_layer, target, name, source_ir)

    out = topk_layer.get_output(1)

    if dim is None:
        new_shape = ((1,) * len(input.shape)) if keep_dim else ()
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_unflatten", out, new_shape
        )
    elif len(input.shape) == 1:
        out = impl.squeeze.squeeze(
            ctx,
            target,
            source_ir,
            f"{name}_squeeze",
            out,
            1 if keep_dim else (0, 1),
        )
    elif not keep_dim:
        out = impl.squeeze.squeeze(ctx, target, source_ir, f"{name}_squeeze", out, dim)

    return out


def argmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MAX, dim, keep_dim
    )


def argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MIN, dim, keep_dim
    )


def sort(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    descending: bool,
    return_indices: bool = True,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    dim = get_positive_dim(dim, len(input.shape))
    k = input.shape[dim]
    return topk(
        ctx,
        target,
        source_ir,
        name,
        input,
        k,
        dim,
        descending,
        sorted=None,
        return_indices=return_indices,
    )


def topk(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: Optional[bool],
    return_indices: bool = True,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    if largest:
        topk_layer = ctx.net.add_topk(
            input,
            trt.TopKOperation.MAX,
            k,
            get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape))),
        )
    else:
        topk_layer = ctx.net.add_topk(
            input,
            trt.TopKOperation.MIN,
            k,
            get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape))),
        )

    # TensorRT ITopKLayer does not have a sorted flag, it is always returning the sorted topk elements
    # so here no matter sorted is True or False the returned the topk Tensor object is always sorted
    set_layer_name(topk_layer, target, f"{name}_topk", source_ir)

    if return_indices:
        return topk_layer.get_output(0), topk_layer.get_output(1)
    else:
        return topk_layer.get_output(0)
