import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import Target
from torch_tensorrt import ENABLED_FEATURES
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    create_constant,
    get_axes_for_reduce_op,
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_trt_weights,
)
from torch_tensorrt.dynamo.conversion.impl.cat import cat
from torch_tensorrt.dynamo.conversion.impl.elementwise.ops import ge
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

_LOGGER: logging.Logger = logging.getLogger(__name__)


def batch_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    momentum: float,
    eps: float,
    return_mean_rstd: bool,
    weight: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]] = None,
    bias: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]] = None,
    running_mean: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]] = None,
    running_var: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]] = None,
    training: bool = False,
    cudnn_enabled: bool = False,
) -> Union[trt.ITensor, Tuple[trt.ITensor, torch.Tensor, torch.Tensor]]:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    # Save the original output shape for later use
    output_shape = input.shape
    feature_num = output_shape[1]
    # We perform constant folding for batch norm when the weight, bias, running_mean, and running_var are all tensors.
    # Batch norm operation can be fused into a single layer, which is more efficient than the original implementation.
    # In this way, the batch norm layer will be fused with the Convolution layer and get a performance boost.
    # TODO: lanl: to remove this once we have solved the batchnorm constant folding issue in RTX
    # https://github.com/pytorch/TensorRT/issues/3699
    if ENABLED_FEATURES.tensorrt_rtx or any(
        [
            isinstance(weight, trt.ITensor),
            isinstance(bias, trt.ITensor),
            isinstance(running_mean, trt.ITensor),
            isinstance(running_var, trt.ITensor),
        ]
    ):
        # We name the weight here according to the state_dict name
        with unset_fake_temporarily():
            weight = (
                get_trt_tensor(
                    ctx, torch.ones((feature_num,)), f"{name}_weight", dtype=input.dtype
                )
                if weight is None
                else get_trt_tensor(ctx, weight, f"{name}_weight")
            )
            bias = (
                get_trt_tensor(
                    ctx, torch.zeros((feature_num,)), f"{name}_bias", dtype=input.dtype
                )
                if bias is None
                else get_trt_tensor(ctx, bias, f"{name}_bias")
            )
            running_mean = (
                get_trt_tensor(
                    ctx,
                    torch.zeros((feature_num,)),
                    f"{name}_running_mean",
                    dtype=input.dtype,
                )
                if running_mean is None
                else get_trt_tensor(ctx, running_mean, f"{name}_running_mean")
            )
            running_var = (
                get_trt_tensor(
                    ctx,
                    torch.ones((feature_num,)),
                    f"{name}_running_var",
                    dtype=input.dtype,
                )
                if running_var is None
                else get_trt_tensor(ctx, running_var, f"{name}_running_var")
            )

        # eps_tensor for numerical stability
        eps_tensor = get_trt_tensor(ctx, eps, f"{name}_eps", dtype=input.dtype)

        # adjusted_var = running_var + eps
        adjusted_var = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_adjusted_var", running_var, eps_tensor
        )

        # sqrt_adjusted_var = sqrt(adjusted_var)
        sqrt_adjusted_var = impl.unary.sqrt(
            ctx, target, source_ir, f"{name}_sqrt", adjusted_var
        )

        # scale = weight / sqrt_adjusted_var
        scale = impl.elementwise.div(
            ctx, target, source_ir, f"{name}_scale", weight, sqrt_adjusted_var
        )

        # scaled_running_mean = running_mean * scale
        scaled_running_mean = impl.elementwise.mul(
            ctx, target, source_ir, f"{name}_scaled_running_mean", running_mean, scale
        )

        # bias_adjusted = bias - scaled_running_mean
        bias_adjusted = impl.elementwise.sub(
            ctx, target, source_ir, f"{name}_bias_adjusted", bias, scaled_running_mean
        )

        # Reshape scale and bias_adjusted to match input shape for broadcasting
        expanded_shape = [1] * len(output_shape)
        expanded_shape[1] = feature_num  # Set channel dimension
        scale_reshape = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_scale",
            scale,
            tuple(expanded_shape),
        )
        bias_adjusted_reshape = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_bias",
            bias_adjusted,
            tuple(expanded_shape),
        )

        # Apply the scale and bias to the input
        scaled_input = impl.elementwise.mul(
            ctx, target, source_ir, f"{name}_scaled_input", input, scale_reshape
        )
        output = impl.elementwise.add(
            ctx,
            target,
            source_ir,
            f"{name}_output",
            scaled_input,
            bias_adjusted_reshape,
        )

    else:
        with unset_fake_temporarily():
            if weight is None:
                weight = torch.ones((feature_num,))

            if bias is None:
                bias = torch.zeros((feature_num,))

            if running_mean is None:
                running_mean = torch.zeros((feature_num,))

            if running_var is None:
                running_var = torch.ones((feature_num,))

            power = torch.ones_like(weight)

            adjusted_scale, adjusted_bias = batch_norm_constant_folding(
                weight, bias, running_mean, running_var, eps
            )

        adjusted_scale = to_trt_weights(
            ctx,
            adjusted_scale,
            name,
            layer_type_name="SCALE",
            weight_type_name="SCALE",
            target=target,
            source_ir=source_ir,
        )
        adjusted_bias = to_trt_weights(
            ctx,
            adjusted_bias,
            name,
            layer_type_name="SCALE",
            weight_type_name="SHIFT",
            target=target,
            source_ir=source_ir,
        )

        power = to_trt_weights(
            ctx,
            power,
            name,
            layer_type_name="SCALE",
            weight_type_name="POWER",
            target=target,
            source_ir=source_ir,
        )

        if len(input.shape) < 4:
            new_shape = (
                (input.shape[0], input.shape[1], 1, 1)
                if len(input.shape) == 2
                else (input.shape[0], input.shape[1], input.shape[2], 1)
            )
            input = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape_2d", input, new_shape
            )

        layer = ctx.net.add_scale_nd(
            input, trt.ScaleMode.CHANNEL, adjusted_bias, adjusted_scale, power, 1
        )
        set_layer_name(layer, target, name, source_ir)
        output = layer.get_output(0)

    # For BatchNorm1d, reshape output back to original shape if necessary
    if len(output_shape) < 4:
        output = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_1d",
            output,
            output_shape,
        )

    if return_mean_rstd:
        # return fake mean and rstd for now
        return output, None, None

    return output


def batch_norm_constant_folding(
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adjusted_scale = weight / torch.sqrt(running_var + eps)
    adjusted_bias = bias - running_mean * adjusted_scale
    return adjusted_scale, adjusted_bias


def native_layer_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    normalized_shape: List[int],
    weight: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]],
    bias: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]],
    eps: float,
) -> Tuple[trt.ITensor, torch.Tensor, torch.Tensor]:
    dims = list(range(len(input.shape) - len(normalized_shape), len(input.shape)))
    axes = get_axes_for_reduce_op(dims)

    weight = get_trt_tensor(
        ctx, weight if weight is not None else 1.0, f"{name}_weight"
    )
    bias = get_trt_tensor(ctx, bias if bias is not None else 0.0, f"{name}_bias")

    # Cast weight and bias to have same dtype as input
    weight = cast_trt_tensor(
        ctx, weight, input.dtype, f"{name}_weight_cast", target, source_ir
    )
    bias = cast_trt_tensor(
        ctx, bias, input.dtype, f"{name}_bias_cast", target, source_ir
    )

    if tuple(input.shape) != tuple(weight.shape):
        weight = impl.slice.expand(
            ctx, target, source_ir, f"{name}_expand_weight", weight, input.shape
        )

    if tuple(input.shape) != tuple(bias.shape):
        bias = impl.slice.expand(
            ctx, target, source_ir, f"{name}_expand_bias", bias, input.shape
        )

    layer = ctx.net.add_normalization(input, weight, bias, axes)
    layer.epsilon = eps
    set_layer_name(layer, target, name, source_ir)

    # return fake mean and rstd for now
    return layer.get_output(0), None, None


def native_group_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    weight: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]],
    bias: Optional[Union[trt.ITensor, torch.Tensor, np.ndarray]],
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> Tuple[trt.ITensor, torch.Tensor, torch.Tensor]:
    rank = len(input.shape)

    assert rank >= 3, f"Expected at least 3 dimensions for input tensor but got {rank}"

    assert (
        C == input.shape[1]
    ), f"num_channels ({C}) must be equal to number of channels in input ({input.shape[1]})"

    shape = [1, group] + [1] * (rank - 2)

    with unset_fake_temporarily():
        weight_torch = torch.ones(shape)
        bias_torch = torch.zeros(shape)

    weight_one = get_trt_tensor(ctx, weight_torch, f"{name}_weight_one", input.dtype)
    bias_zero = get_trt_tensor(ctx, bias_torch, f"{name}_bias_zero", input.dtype)

    axes = get_axes_for_reduce_op(list(range(1 if group == 1 else 2, rank)))

    # INormalizationLayer scales the normalized output per-group, but PyTorch scales the normalized output per-channel,
    # hence causing diverse result. Let TensorRT does no-op for scaling here, and do scaling ourselves later
    layer = ctx.net.add_normalization(input, weight_one, bias_zero, axes)
    layer.epsilon = eps
    layer.num_groups = group
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)

    shape[1] = C

    if weight is not None:
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")
        weight = cast_trt_tensor(
            ctx, weight, input.dtype, f"{name}_cast_weight", target, source_ir
        )
        weight = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_weight", weight, shape
        )
        output = impl.elementwise.mul(
            ctx, target, source_ir, f"{name}_mul_weight", output, weight
        )

    if bias is not None:
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")
        bias = cast_trt_tensor(
            ctx, bias, input.dtype, f"{name}_cast_bias", target, source_ir
        )
        bias = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_bias", bias, shape
        )
        output = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_add_bias", output, bias
        )

    # return fake mean and rstd for now
    return output, None, None


def softmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    dim: int,
    half_to_float: bool,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    dim = get_positive_dim(dim, len(input.shape))

    if half_to_float:
        input = cast_trt_tensor(ctx, input, torch.float, name, target, source_ir)

    layer = ctx.net.add_softmax(input)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pdist(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    p: float = 2,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    shape = input.shape
    # Extend input from shape [N, D] to [N, 1, D]
    extend_input = impl.unsqueeze.unsqueeze(
        ctx,
        target,
        source_ir,
        f"{name}_unsqueeze",
        input,
        1,
    )

    # Expand the input from [N, 1, D] to [N, N, D]
    x = impl.slice.expand(
        ctx,
        target,
        source_ir,
        f"{name}_expand",
        extend_input,
        (shape[0], shape[0]) + shape[1:],
    )
    # Subtract the expanded input from original input. Result shape = [N, N, D]
    # This matrix has the distance of each sample to every other sample and hence the shape is [N, N, D]
    x = impl.elementwise.sub(ctx, target, source_ir, f"{name}_sub", x, input)

    if p == 0:
        # norm = torch.sum(x!=0, dim=2)
        nonzero_val = impl.elementwise.ne(ctx, target, source_ir, f"{name}_ne", x, 0)
        norm = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", nonzero_val, dim=2, keepdim=False
        )
        norm = cast_trt_tensor(
            ctx, norm, torch.float32, f"{name}_cast", target, source_ir
        )
    elif p == 1:
        # norm = torch.sum(torch.abs(x), dim=2)
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        norm = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", abs_val, dim=2, keepdim=False
        )
    elif 0 < p < 1 or 1 < p < float("inf"):
        # norm = torch.pow(torch.sum(torch.pow(torch.abs(x), p), dim=2), 1/p)
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        pow_val = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow1", abs_val, p
        )
        sum_val = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", pow_val, dim=2, keepdim=False
        )
        norm = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow2", sum_val, 1 / p
        )
    elif p == float("inf"):
        # norm = torch.max(torch.abs(x))
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        norm = impl.reduce.max(
            ctx,
            target,
            source_ir,
            f"{name}_max",
            abs_val,
            dim=2,
            keepdim=False,
            return_indices=False,
        )
    else:
        raise RuntimeError(
            f"p should between [0, inf], currently p={p} is not supported!"
        )
    if shape[0] == DYNAMIC_DIM:
        dim = get_shape(ctx, target, source_ir, f"{name}_get_shape", input, 0)
        shuffle_layer = ctx.net.add_shuffle(dim)
        shuffle_layer.reshape_dims = trt.Dims()
        set_layer_name(shuffle_layer, target, f"{name}_shuffle", source_ir)
        dim_tensor = shuffle_layer.get_output(0)
        indices_tensor = tri_upper_indices(
            ctx, target, source_ir, f"{name}_triu_indices", dim_tensor
        )
        gather_layer = ctx.net.add_gather_v2(
            norm, indices_tensor, mode=trt.GatherMode.ND
        )
        set_layer_name(gather_layer, target, f"{name}_gather_layer", source_ir)
        gather_layer.axis = 2
        return gather_layer.get_output(0)
    else:
        indices = np.triu_indices(shape[0], k=1)
        return impl.select.index(ctx, target, source_ir, f"{name}_index", norm, indices)


def tri_upper_indices(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    size_tensor: trt.ITensor,
) -> trt.ITensor:
    """
    Return the indices for the upper-triangle part of a square size of matrix in a N-by-2 Tensor,
    where the diagonal offset = 1. One loop is used to calculate the indices like below.
    x = 0, y = 0, y_start = 1
    out_size = size * (size - 1) // 2
    for _ in range(out_size):
        y_out.append(y_start + y)
        x_out.append(x)
        y += 1
        if (y_start + y) >= size:
            x += 1
            y_start += 1
            y = 0
    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network.
        target (Target): Target of calling node.
        source_ir (Optional[SourceIR]): SourceIR of calling converter.
        name (str): Name of the calling layer.
        size_tensor (trt.ITensor): number of rows in the 2-D square matrix. scalar tensor.

    Example:
        if size_tensor is 4, it will return [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    """
    constant_0 = create_constant(ctx, 0, f"{name}_zero", np.int32, 0)
    constant_1 = create_constant(ctx, 1, f"{name}_one", np.int32, 0)
    constant_2 = create_constant(ctx, 2, f"{name}_two", np.int32, 0)

    size_minus_one = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_size_minus_one", size_tensor, constant_1
    )

    size_mult_prev = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_size_mult_prev", size_tensor, size_minus_one
    )

    num_loop = impl.elementwise.floor_divide(
        ctx, target, source_ir, f"{name}_num_loop", size_mult_prev, constant_2
    )

    loop = ctx.net.add_loop()
    loop.add_trip_limit(num_loop, trt.TripLimit.COUNT)

    x_recurrence = loop.add_recurrence(constant_0)
    set_layer_name(x_recurrence, target, f"{name}_x_recurrence", source_ir)
    x_tensor = x_recurrence.get_output(0)

    y_recurrence = loop.add_recurrence(constant_0)
    set_layer_name(y_recurrence, target, f"{name}_y_recurrence", source_ir)
    y_tensor = y_recurrence.get_output(0)

    y_start_recurrence = loop.add_recurrence(constant_1)
    set_layer_name(y_start_recurrence, target, f"{name}_y_start_recurrence", source_ir)
    y_start_tensor = y_start_recurrence.get_output(0)

    x_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_x_inc",
        x_tensor,
        constant_1,
    )

    y_current = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_current",
        y_start_tensor,
        y_tensor,
    )

    y_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_inc",
        y_tensor,
        constant_1,
    )

    next_y = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_next_y",
        y_start_tensor,
        y_inc,
    )

    y_start_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_start_inc",
        y_start_tensor,
        constant_1,
    )
    cond = ge(ctx, target, source_ir, f"{name}_cond", next_y, size_tensor)
    x_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_x_output",
        x_inc,
        x_tensor,
        cond,
    )
    x_recurrence.set_input(1, x_output)

    y_start_current = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_y_start_current",
        y_start_inc,
        y_start_tensor,
        cond,
    )
    y_start_recurrence.set_input(1, y_start_current)

    y_val = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_y_val",
        constant_0,
        y_inc,
        cond,
    )
    y_recurrence.set_input(1, y_val)

    loop_output_x = loop.add_loop_output(x_tensor, trt.LoopOutput.CONCATENATE)
    loop_output_y = loop.add_loop_output(y_current, trt.LoopOutput.CONCATENATE)
    loop_output_x.set_input(1, num_loop)
    loop_output_y.set_input(1, num_loop)

    # Cat two N tensors into 2 x N. [0, 0, 0], [1, 2, 3] -> [[0, 0, 0], [1, 2, 3]]
    x_index = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x_index", loop_output_x.get_output(0), (1, -1)
    )
    y_index = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_y_index", loop_output_y.get_output(0), (1, -1)
    )

    x_y_tensor = cat(
        ctx,
        target,
        source_ir,
        f"{name}_x_y_tensor",
        [x_index, y_index],
        0,
    )

    # Reshape 2 x N output to N x 2. [[0, 0, 0], [1, 2, 3]] -> [[0, 1], [0, 2], [0, 3]]
    indices_tensor = ctx.net.add_shuffle(x_y_tensor)
    set_layer_name(indices_tensor, target, f"{name}_indices_tensor", source_ir)
    indices_tensor.first_transpose = trt.Permutation([1, 0])
    indices_tensor.reshape_dims = (-1, 2)

    return indices_tensor.get_output(0)


def cdist_forward(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    x1: trt.ITensor,
    x2: trt.ITensor,
    p: float,
    compute_mode: Optional[int],
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    """
    Computes pairwise distances between sets of vectors in tensors x1 and x2 using the p-norm. The function treats the last dimension
    of x1 and x2 as feature dimensions, which must be identical for both inputs. The second-to-last dimensions can differ, reflecting
    the number of vectors in each tensor. The dimensions preceding the last are considered as batch dimensions, and pairwise distances
    are computed for each matching set in these dimensions.

    The output tensor's shape is derived by matching the batch dimensions of x1 and x2, where the mismatched batch dimensions are
    merged, and the resulting shape reflects the computed distances for each pair of vectors. It's crucial that the batch dimensions
    (except for the size of sets of vectors to compare) of x1 and x2 either match or one of them is 1 (broadcasting).

    Args:
        x1 (Tensor): input tensor of shape B x P x M.
        x2 (Tensor): input tensor of shape B x R x M.
        p (float): p value for the p-norm distance to calculate between each vector pair
        compute_mode (int): Controls the computation method based on the size of the input sets:
            - None ('use_mm_for_euclid_dist_if_necessary'): Default mode. Uses matrix multiplication to calculate
              Euclidean distance (p=2) if either the number of vectors in x1 or x2 exceeds 25 (P > 25 or R > 25).
            - 1 ('use_mm_for_euclid_dist'): Always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            - 2 ('donot_use_mm_for_euclid_dist'): Never use matrix multiplication approach to calculate
            euclidean distance (p = 2)

    Example:
    - If x1.shape = [2, 3, 10, 5] and x2.shape = [2, 3, 20, 5], both having the same batch dimensions [2, 3], the output shape will be [2, 3, 10, 20].
      This represents computing distances in two batches of three groups, each comparing 10 vectors from x1 with 20 vectors from x2.
    - For x1.shape = [10, 5] (10 vectors, each of 5 features) and x2.shape = [20, 5] (20 vectors, each of 5 features),
      since there are no batch dimensions to match, the output shape is simply [10, 20], comparing all vectors from x1 against all vectors from x2.

    Note: The `compute_mode` parameter is designed to optimize the performance of the Euclidean distance calculation,
    especially useful when working with large datasets. This parameter allows you to control how the distances are computed,
    with different modes available to leverage matrix multiplication for speed improvements.
    """
    if compute_mode is None:
        compute_mode = 0

    x1_expand_shape = list(x1.shape[:-1]) + [1, x1.shape[-1]]
    x2_expand_shape = list(x2.shape[:-2]) + [1] + list(x2.shape[-2:])

    # Reshape x1 and x2 for broadcasting
    x1_expanded = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x1_expand", x1, x1_expand_shape
    )
    x2_expanded = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x2_expand", x2, x2_expand_shape
    )

    diff = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_diff", x1_expanded, x2_expanded
    )

    if p == 0:
        diff_non_zero = impl.elementwise.ne(
            ctx, target, source_ir, f"{name}_diff_non_zero", diff, 0
        )
        diff_non_zero = cast_trt_tensor(
            ctx, diff_non_zero, torch.float32, f"{name}_cast", target, source_ir
        )
        dist = impl.reduce.sum(
            ctx,
            target,
            source_ir,
            f"{name}_sum",
            diff_non_zero,
            dim=-1,
            keepdim=False,
        )
    elif p == 1:
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        dist = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", abs_val, dim=-1, keepdim=False
        )
    elif p == 2:
        if (
            compute_mode == 0 and (x1.shape[-2] > 25 or x2.shape[-2] > 25)
        ) or compute_mode == 1:
            # Compute squared elements
            x1_squared = impl.elementwise.pow(
                ctx, target, source_ir, f"{name}_x1_squared", x1, 2
            )
            x2_squared = impl.elementwise.pow(
                ctx, target, source_ir, f"{name}_x2_squared", x2, 2
            )

            # Sum squares along the last dimension
            x1_sum_squared = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                f"{name}_x1_sum",
                x1_squared,
                dim=-1,
                keepdim=True,
            )
            x2_sum_squared = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                f"{name}_x2_sum",
                x2_squared,
                dim=-1,
                keepdim=True,
            )

            # Reshape sums for broadcasting
            rank = len(x2.shape)
            permute_shape = list(range(rank - 2)) + [rank - 1, rank - 2]
            x1_sum_expanded = x1_sum_squared
            x2_sum_expanded = impl.permutation.permute(
                ctx, target, source_ir, f"{name}_permute", x2_sum_squared, permute_shape
            )

            # Compute dot product of x1 and transposed x2
            x2_tr = impl.permutation.permute(
                ctx, target, source_ir, f"{name}_permute_mm", x2, permute_shape
            )
            dot_product = impl.matmul.matrix_multiply(
                ctx,
                target,
                source_ir,
                f"{name}_dot_product",
                x1,
                x2_tr,
                input_matrix_op=trt.MatrixOperation.NONE,
                other_matrix_op=trt.MatrixOperation.NONE,
            )

            # Combine results to get squared distances
            dist_squared = impl.elementwise.add(
                ctx,
                target,
                source_ir,
                f"{name}_dist_squared_initial",
                x1_sum_expanded,
                x2_sum_expanded,
            )
            dist_squared = impl.elementwise.sub(
                ctx,
                target,
                source_ir,
                f"{name}_dist_squared",
                dist_squared,
                impl.elementwise.mul(
                    ctx, target, source_ir, f"{name}_dot_product_scaled", dot_product, 2
                ),
            )

            # Compute the Euclidean distances
            dist = impl.unary.sqrt(ctx, target, source_ir, f"{name}_dist", dist_squared)
        else:
            diff_squared = impl.elementwise.pow(
                ctx, target, source_ir, f"{name}_diff_squared", diff, 2
            )
            dist_squared = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                f"{name}_dist_sq_sum",
                diff_squared,
                dim=-1,
                keepdim=False,
            )
            dist = impl.unary.sqrt(ctx, target, source_ir, f"{name}_sqrt", dist_squared)
    elif 0 < p < 1 or 1 < p < 2 or 2 < p < float("inf"):
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        pow_val = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow_val_1", abs_val, p
        )
        sum_val = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", pow_val, dim=-1, keepdim=False
        )
        dist = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow_val_2", sum_val, 1 / p
        )
    elif p == float("inf"):
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        dist = impl.reduce.max(
            ctx,
            target,
            source_ir,
            f"{name}_max",
            abs_val,
            dim=-1,
            keepdim=False,
            return_indices=False,
        )
    return dist
