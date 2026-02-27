import functools
import time
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    append,
    cast_trt_tensor,
    get_trt_tensor,
    set_item,
    set_layer_name,
    to_numpy,
)


def embedding(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
) -> TRTTensor:
    indices_tensor = input
    embedding_tensor = weight
    if isinstance(indices_tensor, torch.Tensor) and indices_tensor.dtype == torch.int64:
        raise RuntimeError(
            "The `embedding` op has indices_tensor dtype=int64. This is incorrect since it has to be int32 to run on TRT."
        )
    indices_tensor = get_trt_tensor(ctx, indices_tensor, f"{name}_indices_tensor")
    embedding_tensor = get_trt_tensor(ctx, embedding_tensor, f"{name}_embedding_tensor")
    # unsupported parameters
    # ignore padding_idx, scale_grad_by_freq, and sparse
    # since they are meaningful for training only

    # useful for training only
    # if scale_grad_by_freq:
    #     raise RuntimeError(
    #         "Currently we don't support scale gradient by word frequency."
    #     )

    # if sparse:
    #     raise RuntimeError("Currently we don't support sparse gradient.")

    # Implement embedding lookup with gather layer
    gather_layer = ctx.net.add_gather(embedding_tensor, indices_tensor, axis=0)
    set_layer_name(gather_layer, target, f"{name}_gather", source_ir)
    return gather_layer.get_output(0)


def embedding_bag_with_traversable_offsets(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    embed: TRTTensor,
    offsets_list: Union[torch.Tensor, np.ndarray, Sequence[int]],
    mode: int,
    include_last_offset: bool,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    reduce_name = ""
    if mode == 0:  # sum
        reduce_op = functools.partial(
            impl.reduce.sum, ctx=ctx, target=target, source_ir=source_ir
        )
        reduce_name = "sum"
    elif mode == 1:  # mean
        reduce_op = functools.partial(
            impl.reduce.mean, ctx=ctx, target=target, source_ir=source_ir
        )
        reduce_name = "mean"
    elif mode == 2:  # max
        reduce_op = functools.partial(
            impl.reduce.max,
            ctx=ctx,
            target=target,
            source_ir=source_ir,
            return_indices=False,
        )
        reduce_name = "max"

    offsets: np.ndarray = to_numpy(offsets_list)
    len_embed = embed.shape[0]

    if include_last_offset:
        # modify the last index of offsets to the end index
        # however, pytorch doc says if `include_last_offset` is True, the size of offsets
        # is equal to the number of bags + 1. The last element is the size of the input,
        # or the ending index position of the last bag (sequence).

        # Notes: here offsets should always be 1d array
        if len(offsets.shape) != 1:
            raise TypeError(
                f"The offsets should be in 1d array, here offset shape is {offsets.shape}."
            )
        offsets[-1] = len_embed
    else:
        # add the end index to offsets
        offsets = np.append(offsets, len_embed)

    zero_tensor = get_trt_tensor(
        ctx, np.zeros((1, embed.shape[1]), dtype=np.float32), f"{name}_zero_tensor"
    )

    # separately reduce embeddings for different bags
    reduced_embed_bags = []
    len_offsets = offsets.shape[0]
    for i in range(len_offsets - 1):
        if offsets[i] < offsets[i + 1]:
            sliced_embed = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_embed_{i}",
                embed,
                0,
                int(offsets[i]),
                int(offsets[i + 1]),
                1,
            )
            reduced_one_bag = reduce_op(
                name=f"{name}_{reduce_name}_{i}",
                input_val=sliced_embed,
                dim=0,
                keepdim=True,
            )
            reduced_embed_bags.append(reduced_one_bag)
        else:  # offsets[i] == offsets[i + 1]
            reduced_embed_bags.append(zero_tensor)

    out = impl.cat.cat(ctx, target, source_ir, f"{name}_cat", reduced_embed_bags, 0)
    return out, None, None, None


def embedding_bag_with_ITensor_offsets(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    embed: TRTTensor,
    offsets: TRTTensor,
    mode: int,
    include_last_offset: bool,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    # prepare some tensors for future use
    constant_0 = get_trt_tensor(ctx, 0, f"{name}_constant_tensor_0")
    constant_1 = get_trt_tensor(ctx, 1, f"{name}_constant_tensor_1")

    embed_shape = ctx.net.add_shape(embed).get_output(0)
    len_embed = ctx.net.add_gather(embed_shape, constant_0, 0).get_output(0)

    if include_last_offset:
        # modify the last index of offsets to the end index
        # however, pytorch doc says if `include_last_offset` is True, the size of offsets
        # is equal to the number of bags + 1. The last element is the size of the input,
        # or the ending index position of the last bag (sequence).
        offsets = set_item(
            ctx, target, source_ir, f"{name}_set_item", offsets, -1, len_embed
        )
    else:
        # add the end index to `offsets`
        offsets = append(ctx, target, source_ir, f"{name}_append", offsets, len_embed)

    # create a placeholder tensor, whose shape is the same as an embedding
    # if mode is 0 (sum) or 1 (mean), the placeholder tensor is filled with zeros
    # if mode is 2 (max), the placeholder tensor is filled with negative infinity
    placeholder_tensor = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_zero_tensors", embed, 0
    )
    zero_tensor = ctx.net.add_gather(placeholder_tensor, constant_0, 0).get_output(0)
    zero_tensor = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_reshape_zero_tensor", zero_tensor, (-1,)
    )

    if mode == 2:
        placeholder_tensor = impl.elementwise.add(
            ctx,
            target,
            source_ir,
            f"{name}_negative_inf_tensor",
            placeholder_tensor,
            -np.inf,
        )

    # Use two for loops to calculate the embedding of each bag
    ###### Outer loop: traverse offsets ######
    loop1 = ctx.net.add_loop()

    offsets_shape = ctx.net.add_shape(offsets).get_output(0)
    offsets_size = ctx.net.add_gather(offsets_shape, constant_0, 0).get_output(0)
    trip_limit1 = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_trip_limit1", offsets_size, 1
    )
    # change to 0d tensor
    trip_limit1 = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_reshape_trip_limit1", trip_limit1, ()
    )

    loop1.add_trip_limit(trip_limit1, trt.TripLimit.COUNT)

    rec1_i_tensor = loop1.add_recurrence(constant_1)
    set_layer_name(rec1_i_tensor, target, f"{name}_rec1_i_tensor", source_ir)
    i_tensor = rec1_i_tensor.get_output(0)

    start = ctx.net.add_gather(offsets, constant_0, 0).get_output(0)
    rec1_start = loop1.add_recurrence(start)
    set_layer_name(rec1_start, target, f"{name}_rec1_start", source_ir)
    start = rec1_start.get_output(0)

    end = ctx.net.add_gather(offsets, constant_1, 0).get_output(0)
    rec1_end = loop1.add_recurrence(end)
    set_layer_name(rec1_end, target, f"{name}_rec1_end", source_ir)
    end = rec1_end.get_output(0)

    ###### Inner loop: traverse indices ######
    loop2 = ctx.net.add_loop()
    trip_limit2 = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_reshape_trip_limit2", len_embed, ()
    )
    loop2.add_trip_limit(trip_limit2, trt.TripLimit.COUNT)
    rec2_j_tensor = loop2.add_recurrence(constant_0)
    set_layer_name(rec2_j_tensor, target, f"{name}_rec2_j_tensor", source_ir)
    j_tensor = rec2_j_tensor.get_output(0)

    # create a TRT Select layer
    cond1 = impl.elementwise.ge(
        ctx, target, source_ir, f"{name}_ge_{time.time()}", j_tensor, start
    )
    cond2 = impl.elementwise.lt(
        ctx, target, source_ir, f"{name}_lt_{time.time()}", j_tensor, end
    )
    condition1 = impl.elementwise.logical_and(
        ctx, target, source_ir, f"{name}_and_{time.time()}", cond1, cond2
    )
    next_j = impl.elementwise.add(
        ctx, target, source_ir, f"{name}_j_tensor_add_1_{time.time()}", j_tensor, 1
    )
    rec2_j_tensor.set_input(1, next_j)
    loop_out2 = loop2.add_loop_output(condition1, trt.LoopOutput.CONCATENATE)
    loop_out2.set_input(1, trip_limit2)
    ####### Inner loop end #######

    select_layer1 = ctx.net.add_select(
        loop_out2.get_output(0), embed, placeholder_tensor
    )
    one_bag = select_layer1.get_output(0)

    # reduce the one_bag along the dim=0, the result of which is an embedding of each bag
    if mode == 0:  # sum
        reduced_one_bag = impl.reduce.sum(
            ctx,
            target,
            source_ir,
            name=f"{name}_sum_bag{time.time()}",
            input_val=one_bag,
            dim=0,
            keepdim=False,
        )

    # Since one_bag includes many zeros, directly calculating mean will cause results incorrect
    elif mode == 1:  # mean
        reduced_one_bag = impl.reduce.sum(
            ctx,
            target,
            source_ir,
            name=f"{name}_sum_bag{time.time()}",
            input_val=one_bag,
            dim=0,
            keepdim=False,
        )
        diff = impl.elementwise.sub(
            ctx, target, source_ir, f"{name}_diff_bag{time.time()}", end, start
        )
        reduced_one_bag = impl.elementwise.div(
            ctx,
            target,
            source_ir,
            f"{name}_div_bag{time.time()}",
            reduced_one_bag,
            diff,
        )

    elif mode == 2:  # max
        reduced_one_bag = impl.reduce.max(
            ctx,
            target,
            source_ir,
            name=f"{name}_max_bag{time.time()}",
            input_val=one_bag,
            dim=0,
            keepdim=False,
            return_indices=False,
        )

    # create a TRT conditional layer
    conditional_layer1 = ctx.net.add_if_conditional()
    condition2 = impl.elementwise.eq(
        ctx, target, source_ir, f"{name}_condition2_eq_{time.time()}", start, end
    )
    condition2 = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_condition2_eq_{time.time()}",
        condition2,
        [],
    )
    # set the combined condition to the conditional layer
    conditional_layer1.set_condition(condition2)
    # if true, run this subgraph
    true_sg = conditional_layer1.add_input(zero_tensor)
    # if false, run this subgraph
    false_sg = conditional_layer1.add_input(reduced_one_bag)

    reduced_one_bag_layer = conditional_layer1.add_output(
        true_sg.get_output(0), false_sg.get_output(0)
    )

    # reset the variables for the next iteration of the outer loop
    next_i = impl.elementwise.add(
        ctx, target, source_ir, f"{name}_i_tensor_add_1_{time.time()}", i_tensor, 1
    )
    rec1_i_tensor.set_input(1, next_i)
    rec1_start.set_input(1, end)
    rec1_end.set_input(1, ctx.net.add_gather(offsets, next_i, 0).get_output(0))

    loop_out1 = loop1.add_loop_output(
        reduced_one_bag_layer.get_output(0), trt.LoopOutput.CONCATENATE
    )
    loop_out1.set_input(1, trip_limit1)
    reduced_embed_bags = loop_out1.get_output(0)
    ####### Outer loop end #######
    return reduced_embed_bags, None, None, None


def embedding_bag(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    weight: TRTTensor,
    indices: TRTTensor,
    offsets: TRTTensor,
    mode: int,
    per_sample_weights: Optional[TRTTensor],  # for sum mode only
    include_last_offset: bool,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    """
    This function is for calculating embedding bags.

    In PyTorch, `offsets` is only used when input is 1D. If input is 2D of shape (B, N),
    it will be treated as B bags (sequences) each of fixed length N, and this will return
    B values aggregated in a way depending on the mode. `offsets` is ignored and required
    to be None in this case.

    However, according to the schema, `offsets` is required for input with any dimensions.
    Accordingly, this function flattens N-D input to 1D and then to calculate embedding bags.
    """

    # TODO: support 2D inputs
    # indices = impl.shuffle.reshape(ctx, target, source_ir, f"{name}_reshape_indices", indices, (-1,))

    # calculate embedding
    embed = embedding(
        ctx,
        target,
        source_ir,
        f"{name}_embedding",
        indices,
        weight,
    )
    embed = cast_trt_tensor(
        ctx, embed, torch.float, f"{name}_cast_embed_to_fp32", target, source_ir
    )

    # give weights to embedding
    if per_sample_weights is not None:
        assert (
            per_sample_weights.shape == indices.shape
        ), f"`per_sample_weights` (shape: {per_sample_weights.shape}) must have exactly the same shape as indices/input (shape: {indices.shape})!"
        per_sample_weights = get_trt_tensor(
            ctx, per_sample_weights, f"{name}_per_sample_weights", np.float32
        )
        per_sample_weights = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_per_sample_weights",
            per_sample_weights,
            (-1, 1),
        )
        embed = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            f"{name}_mul_per_sample_weights",
            embed,
            per_sample_weights,
        )

    if isinstance(offsets, TRTTensor):
        return embedding_bag_with_ITensor_offsets(
            ctx, target, source_ir, name, embed, offsets, mode, include_last_offset
        )
    else:
        # this branch has less time complexity
        return embedding_bag_with_traversable_offsets(
            ctx, target, source_ir, name, embed, offsets, mode, include_last_offset
        )
