from typing import List, Optional, Sequence, Union

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape


def split(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"split received input {input} that is not part " "of the TensorRT region!"
        )

    dynamic_shape = has_dynamic_shape(input.shape)
    if dynamic_shape > 0:
        # Check whether slice target dim is dynamic shape dim
        assert input.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    split_sizes = []
    if isinstance(split_size_or_sections, int):
        split_sizes.append(split_size_or_sections)
    else:
        for split_size_or_section in split_size_or_sections:
            split_sizes.append(split_size_or_section)

    start = [0] * len(input.shape)
    stride = [1] * len(start)
    offset = 0
    if len(split_sizes) == 1:
        num_splits = (input.shape[dim] + split_sizes[0] - 1) // split_sizes[0]
        split_sizes = [split_sizes[0]] * num_splits
    else:
        num_splits = len(split_sizes)
        sum_split_sizes = sum(split_sizes)
        if sum_split_sizes != input.shape[dim]:
            raise RuntimeError(
                "split sizes don't add up to the tensor's size in the given dimension"
            )

    if num_splits < 1:
        raise RuntimeError(
            f"Invalid split: {input.shape[dim]} with split_size={split_sizes}"
        )

    max_offset = input.shape[dim]
    # add slice layers
    output = []
    for i in range(num_splits):
        shape = list(input.shape)
        shape[dim] = min(split_sizes[i], max_offset - offset)
        start[dim] = offset
        if dynamic_shape:
            shape = get_shape_with_dynamic_shape(
                ctx, target, source_ir, f"{name}_shape_{i}", shape, input
            )
        layer = ctx.net.add_slice(
            input, start=start, shape=[] if dynamic_shape else shape, stride=stride
        )
        if dynamic_shape:
            layer.set_input(2, shape)
        offset += split_sizes[i]
        set_layer_name(layer, target, f"{name}_{i}")
        output.append(layer.get_output(0))
    return output
