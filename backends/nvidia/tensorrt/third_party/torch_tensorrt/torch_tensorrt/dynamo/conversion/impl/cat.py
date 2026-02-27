from typing import List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    set_layer_name,
)


def unify_and_concat_trt_tensors(
    ctx: ConversionContext,
    target: Target,
    name: str,
    inputs: Sequence[Union[int, np.ndarray, torch.Tensor, TRTTensor]],
    concat_axis: int,
    cast_dtype: Union[_enums.dtype, trt.DataType, np.dtype] = None,
    force_trt_output: bool = False,
) -> Union[TRTTensor, List[int]]:
    """
    Normalize all inputs to TRT tensors if needed, optionally cast, and concat if any dynamic.

    Args:
        ctx: TensorRT conversion context.
        target: Operation Target.
        name: Operation Name.
        inputs: Sequence of ints / numpy arrays / torch tensors / TRT tensors.
        concat_axis: Axis along which to concatenate tensors if dynamic.
        cast_dtype: Optional target dtype for casting TRT tensors.
        force_trt_output: If True, return TRT tensor even if all inputs are static ints. (True for concat operations)
    """
    has_dynamic = any(not isinstance(x, int) for x in inputs)
    trt_tensors = []

    for i, x in enumerate(inputs):
        # convert to TRTTensor
        if isinstance(x, TRTTensor):
            t = x
        elif isinstance(x, int) and not has_dynamic and not force_trt_output:
            t = x  # pure static path
        else:
            const_arr = np.array([x], dtype=np.int32)
            shape = (1,)
            if not isinstance(x, int):
                const_arr = np.array(x, dtype=np.int32)
                shape = (x.numel(),)

            layer = ctx.net.add_constant(shape, const_arr)
            set_layer_name(layer, target, f"{name}_dim{i}_const")
            t = layer.get_output(0)
        trt_tensors.append(t)

    if not has_dynamic and not force_trt_output:
        return trt_tensors  # all ints

    final_dtype = None
    if cast_dtype:
        # Explicit cast requested
        if isinstance(cast_dtype, _enums.dtype):
            final_dtype = cast_dtype.to(trt.DataType)
        elif isinstance(cast_dtype, (np.dtype, torch.dtype)):
            final_dtype = _enums.dtype._from(cast_dtype).to(trt.DataType)
        else:
            final_dtype = cast_dtype  # already trt.DataType
    else:
        # Automatic promotion
        promoted_type = None
        for t in trt_tensors:
            if isinstance(t, TRTTensor):
                if promoted_type is None:
                    promoted_type = t.dtype
                else:
                    promoted_type = _enums.dtype._from(
                        torch.promote_types(
                            _enums.dtype._from(promoted_type).to(torch.dtype),
                            _enums.dtype._from(t.dtype).to(torch.dtype),
                        )
                    ).to(trt.DataType)
        final_dtype = promoted_type

    # promote remaining ints to TRT consts before concat
    for i, t in enumerate(trt_tensors):
        if isinstance(t, int):
            const = ctx.net.add_constant((1,), np.array([t], dtype=np.int32))
            set_layer_name(const, target, f"{name}_static_{i}_const")
            trt_tensors[i] = const.get_output(0)

    # final cast
    if final_dtype is not None:
        casted = []
        for i, t in enumerate(trt_tensors):
            if isinstance(t, TRTTensor):
                t = cast_trt_tensor(ctx, t, final_dtype, f"{name}_cast_{i}")
            casted.append(t)
        trt_tensors = casted

    concat = ctx.net.add_concatenation(trt_tensors)
    concat.axis = concat_axis
    set_layer_name(concat, target, f"{name}_concat")
    return concat.get_output(0)


def cat(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    dim: int,
    cast_dtype: Union[_enums.dtype, trt.DataType, np.dtype] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # int is only when cat called in other ops like pad
    if not isinstance(input[0], int):
        dim = get_positive_dim(dim, len(input[0].shape))
    else:
        dim = 0
    return unify_and_concat_trt_tensors(
        ctx,
        target,
        name,
        input,
        concat_axis=dim,
        cast_dtype=cast_dtype,
        force_trt_output=True,
    )
