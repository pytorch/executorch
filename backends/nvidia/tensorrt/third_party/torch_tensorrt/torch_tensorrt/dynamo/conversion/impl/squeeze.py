from typing import Optional, Sequence, Union

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    set_layer_name,
)


def squeeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]] = None,
) -> TRTTensor:
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert dim is not None, "We don't support dim=None right now for squeeze."
    dims = []

    if isinstance(dim, int):
        dims.append(dim)
    else:
        for d in dim:
            dims.append(d)

    new_dims = []
    for dim in dims:
        dim = get_positive_dim(
            dim,
            len(input.shape),
        )

        assert input.shape[dim] != -1, "We don't support squeeze dynamic dim."
        new_dims.append(dim)

    dim_to_remove = []
    new_permutation = []
    for i, s in enumerate(input.shape):
        if (i in new_dims) and s == 1:
            dim_to_remove.append(i)
        else:
            new_permutation.append(i)
    # If number of reshape dimensions is less than input, 0s are resolved by aligning
    # the most significant dimensions of input
    output_shape = tuple([0] * len(new_permutation))
    new_permutation += dim_to_remove

    layer = ctx.net.add_shuffle(input)
    layer.first_transpose = new_permutation
    layer.reshape_dims = output_shape
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
