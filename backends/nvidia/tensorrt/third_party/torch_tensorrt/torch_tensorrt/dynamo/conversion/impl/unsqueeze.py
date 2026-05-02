import logging
from typing import List, Optional, Sequence, cast

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
    set_layer_name,
)

logger = logging.getLogger(__name__)


def unsqueeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
) -> TRTTensor:
    # tensorrt version < 10.7.0, use the old unsqueeze implementation
    if is_tensorrt_version_supported("10.7.0"):
        # use the new unsqueeze implementation
        axes = get_trt_tensor(ctx, dim, f"{name}_axes")
        layer = ctx.net.add_unsqueeze(input, axes)
        set_layer_name(layer, target, name, source_ir)
        return layer.get_output(0)
    else:
        logger.warning(
            "IUnsqueezeLayer is supported starting from TensorRT 10.7.0, using the old unsqueeze implementation in the current TensorRT version"
        )
        return unsqueeze_old(ctx, target, source_ir, name, input, dim)


# old implementation for jetson due to IUnsqueezeLayer was not supported prior to 10.7.0
def unsqueeze_old(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
) -> TRTTensor:
    input_val = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"unsqueeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, dim)

    input_shape_size = len(input_val.shape)
    dim = get_positive_dim(dim, input_shape_size + 1)

    intermediate_dim = 0
    dynamic_shape_cnt = 0
    # if unsqueeze the last dimensions, we can directly append to the shape
    if dim == input_shape_size:
        intermediate_dim = dim
    else:
        # since maximum of one dimension is permitted to be specified as -1
        # find the intermediate_dim which has only 1 dynamic_shape_cnt
        # and then we can add a transpose after reshape if it is not the final shape we want
        for i, s in reversed(list(enumerate(input_val.shape))):
            if i >= dim:
                if s == -1:
                    dynamic_shape_cnt += 1
                if dynamic_shape_cnt > 1:
                    intermediate_dim = i + 1
                    break
                if i == dim:
                    intermediate_dim = i
                    break
    # calculate the new_shape for the shuffle layer's reshape_dims
    new_shape = list(
        tuple(input_val.shape)[:intermediate_dim]
        + (1,)
        + tuple(input_val.shape)[intermediate_dim:]
    )
    for i, s in enumerate(new_shape):
        if i < intermediate_dim and s == -1:
            new_shape[i] = 0
    layer = ctx.net.add_shuffle(input_val)
    layer.reshape_dims = tuple(new_shape)
    # if the intermediate_dim is not the final dim we want to unsqueeze, add a second_transpose after reshape
    if intermediate_dim != dim:
        # calculate the second_transpose for the shuffle layer
        permutation = [*range(0, len(new_shape))]
        # for example: if the reshape_dims is (3, 3, 5, 1, 5) and the final shape we want is (3, 1, 3, 5, 5)
        # here intermediate_dim=3, dim=1, we need to move intermediate_dim before [dim: intermediate_dim)
        new_permutation = (
            tuple(permutation[:dim])
            + (intermediate_dim,)
            + tuple(permutation[dim:intermediate_dim])
            + tuple(permutation[intermediate_dim + 1 :])
        )
        layer.second_transpose = new_permutation
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def broadcast_in_dim(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t: TRTTensor,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> TRTTensor:
    augmented_shape_list: List[Optional[int]] = list(shape)

    # For each dimension being broadcasted, set the augmented shape to None
    for broadcast_dim in broadcast_dimensions:
        augmented_shape_list[broadcast_dim] = None

    # TODO: Expand support to arbitrary broadcasts
    assert all(
        dim in (1, None) for dim in augmented_shape_list
    ), "broadcast_in_dim currently only supports unsqueeze broadcasting"

    # Unsqueeze the shape repeatedly to broadcast
    output = input_t
    for idx, x in enumerate(augmented_shape_list):
        # If the value is not None, that dimension is to be broadcasted
        if x is not None:
            output = unsqueeze(
                ctx,
                target,
                source_ir,
                name + f"_unsqueeze_for_broadcast_{idx}",
                output,
                idx,
            )

    assert tuple(output.shape) == tuple(shape), "broadcast_in_dim shapes don't match"

    return output
