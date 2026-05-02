import logging
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._decomp import register_decomposition
from torch._export.utils import (
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
)
from torch._ops import OpOverload
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim
from torch_tensorrt.dynamo.utils import to_torch_device

from ._decomposition_groups import (
    ENABLED_TORCH_DECOMPOSITIONS,
    TORCH_TRT_DECOMPOSITIONS,
    _core_aten_decompositions,
    aten,
    torch_disabled_decompositions,
    torch_enabled_decompositions,
)

logger = logging.getLogger(__name__)


def register_torch_trt_decomposition(
    aten_op: OpOverload, registry: Optional[Any] = None
) -> Callable[[Any], Any]:
    """Checks if the decomposition already exists in one of the sets
    Registers the decomposition via the Torch utility

    Alerts the user if the decomposition already exists, before registering
    Throws an AssertionError if the user attempts to register a decomposition
    which is present in the set of explicitly disabled decompositions
    """
    if aten_op in torch_enabled_decompositions:
        logger.warning(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in torch_enabled_decompositions. "
            "The custom implementation will take precedence."
        )
    elif aten_op in torch_disabled_decompositions:
        logger.info(
            f"Detected custom decomposition for {aten_op}, which is present "
            "in torch_disabled_decompositions."
        )

    # Conflicts with _core_aten_decompositions will only occur if
    # enable_experimental_decompositions is True in get_decompositions
    if aten_op in _core_aten_decompositions:
        logger.debug(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in core_aten_decompositions. "
            "The custom implementation will take precedence."
        )

    def register(fn: Callable[[Any], Any]) -> Any:
        return register_decomposition(aten_op=aten_op, registry=registry)(fn)

    return register


def replace_inplace_op(aten_op: OpOverload, outplace_op: OpOverload) -> Any:
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_torch_trt_decomposition(aten_op, registry=TORCH_TRT_DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):  # type: ignore
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


replace_inplace_op(aten.add_, aten.add)
replace_inplace_op(aten.addbmm_, aten.addbmm)
replace_inplace_op(aten.addmm_, aten.addmm)
replace_inplace_op(aten.addmv_, aten.addmv)
replace_inplace_op(aten.baddbmm_, aten.baddbmm)
replace_inplace_op(aten.cumprod_, aten.cumprod)
replace_inplace_op(aten.index_put_, aten.index_put)
replace_inplace_op(aten.index_reduce_, aten.index_reduce)
replace_inplace_op(aten.relu_, aten.relu)
replace_inplace_op(aten.round_, aten.round)
replace_inplace_op(aten.scatter_, aten.scatter)
replace_inplace_op(aten.scatter_add_, aten.scatter_add)
replace_inplace_op(aten.scatter_reduce_, aten.scatter_reduce)


@register_torch_trt_decomposition(aten.rsqrt, registry=TORCH_TRT_DECOMPOSITIONS)
def rsqrt_replacement(*args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_torch_trt_decomposition(aten._unsafe_view, registry=TORCH_TRT_DECOMPOSITIONS)
def unsafe_view_replacement(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reshape(x, *args, **kwargs)


@register_torch_trt_decomposition(
    torch.ops.aten.lift_fresh_copy, registry=TORCH_TRT_DECOMPOSITIONS
)
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(aten.alias, registry=TORCH_TRT_DECOMPOSITIONS)
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(
    torch.ops.aten.reciprocal.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


@register_torch_trt_decomposition(
    torch.ops.prims.var.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def var_decomposition(
    input_tensor: torch.Tensor,
    dims: Optional[List[int]],
    correction: int,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dims is None:
        dims = []

    # If the dimensions are empty, variance is taken over all dimensions
    if isinstance(dims, (tuple, list)) and len(dims) == 0:
        N = input_tensor.numel()
    # Otherwise, the number of samples is the product of the dimensions reduced over
    else:
        N = 1
        for dim_i in dims:
            N *= input_tensor.shape[dim_i]

    # Compute the mean, difference, and correction term as per the formula:
    # https://pytorch.org/docs/stable/generated/torch.var.html

    # Additionally, prims does not support keepdim, and so we only keep dimensions
    # on the first reduction, then remove it for the second
    sample_mean = torch.mean(input_tensor, dims, keepdim=True)
    diff = input_tensor - sample_mean
    squared_diff = diff * diff
    variance_unnormalized = torch.sum(squared_diff, dims, keepdim=False)

    if correction is None:
        correction_term = float(N - 1)
    elif isinstance(correction, int):
        correction_term = float(N - correction)
    elif isinstance(correction, float):
        correction_term = float(N) - correction
    else:
        raise RuntimeError("correction must be int or float")

    if correction_term <= 0:
        raise RuntimeError(f"correction term was non-positive, got: {correction_term}")

    variance = variance_unnormalized / correction_term

    return variance


@register_torch_trt_decomposition(
    torch.ops.aten.empty_permuted.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def empty_permuted_decomposition(*args, **kwargs) -> torch.Tensor:  # type: ignore
    empty_size = args[0]
    empty_permute = args[1]
    perm = [0] * len(empty_size)
    for permute_index, permute_element in enumerate(empty_permute):
        perm[permute_element] = permute_index
    kwargs["device"] = to_torch_device(default_device())
    return torch.empty([empty_size[l] for l in empty_permute], **kwargs).permute(perm)


@register_torch_trt_decomposition(
    torch.ops.aten.slice_scatter.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def slice_scatter_decomposition(
    input_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> torch.Tensor:
    dim_size = input_tensor.shape[dim]
    device_input_tensor = input_tensor.device

    start = 0 if start is None else start  # Ensure start is int
    start = get_positive_dim(start, input_tensor.shape[dim])
    if end is None:  # Ensure end is int
        end = dim_size
    end = (
        get_positive_dim(end, input_tensor.shape[dim]) if isinstance(end, int) else end
    )
    if step is None:
        step = 1

    # step == 0 is not a valid torch case
    if start == 0 and end == dim_size and step == 1:
        return src_tensor

    # Ensure start, end, and step are all integers
    assert isinstance(start, (int, torch.SymInt)), "start must be an int or SymInt"
    assert isinstance(end, (int, torch.SymInt)), "end must be an int or SymInt"
    assert isinstance(step, (int, torch.SymInt)), "step must be an int or SymInt"

    indices = torch.arange(
        start, end, step, device=device_input_tensor, dtype=torch.int64
    )
    index_tensor = indices.view(
        [-1 if i == dim else 1 for i in range(input_tensor.dim())]
    )
    index_tensor = index_tensor.expand_as(src_tensor)
    return torch.scatter(input_tensor, dim, index_tensor, src_tensor)


@register_torch_trt_decomposition(
    torch.ops.aten.select_scatter.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def select_scatter_decomposition(
    input_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
    index: int,
) -> torch.Tensor:
    src_tensor = torch.unsqueeze(src_tensor, dim)
    return torch.slice_scatter(input_tensor, src_tensor, dim, index, index + 1, 1)


@register_torch_trt_decomposition(
    torch.ops.aten.empty_strided.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def empty_strided_decomposition(*args, **kwargs) -> torch.Tensor:  # type: ignore
    empty_size = args[0]
    empty_stride = args[1]
    return torch.as_strided(
        torch.empty(empty_size, device=to_torch_device(default_device())),
        empty_size,
        empty_stride,
    )


@register_torch_trt_decomposition(
    torch.ops.aten.silu.default, registry=TORCH_TRT_DECOMPOSITIONS
)
@register_torch_trt_decomposition(
    torch.ops.aten.silu_.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def silu(self: torch.Tensor) -> torch.Tensor:
    return self * torch.sigmoid(self)


@register_torch_trt_decomposition(
    torch.ops.aten.scatter_add.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def scatter_add_decomposition(
    input_tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src_tensor: torch.Tensor,
) -> torch.Tensor:
    scatter_add_tensor = input_tensor
    src_shape = list(src_tensor.shape)
    src_dim = src_shape[dim]
    for i in range(0, src_dim):
        to_scatter_tensor = torch.zeros(input_tensor.shape, dtype=input_tensor.dtype)
        # index and src slice
        src_slice = torch.select(src_tensor, dim, i)
        index_slice = torch.select(index, dim, i)

        # unsqueeze src and index in dim
        src_slice = torch.unsqueeze(src_slice, dim)
        index_slice = torch.unsqueeze(index_slice, dim)

        # moving tensor to default device
        device = input_tensor.device
        scatter_add_tensor = scatter_add_tensor.to(device)
        to_scatter_tensor = to_scatter_tensor.to(device)
        index_slice = index_slice.to(device)
        src_slice = src_slice.to(device)

        scatter_add_tensor = torch.add(
            scatter_add_tensor,
            torch.scatter(to_scatter_tensor, dim, index_slice, src_slice),
        )

    return scatter_add_tensor


# enum class for reduce operation of scatter_reduce
class ReduceOperation(Enum):
    SUM = ("Sum reduce operation", lambda x, y: torch.add(x, y))
    PROD = ("Product reduce operation", lambda x, y: torch.mul(x, y))
    MEAN = ("Mean reduce operation", lambda x, y: torch.add(x, y))
    AMAX = ("Amax reduce operation", lambda x, y: torch.max(x, y))
    AMIN = ("Amin reduce operation", lambda x, y: torch.min(x, y))

    def __new__(cls, description: Any, func: Any) -> Any:
        obj = object.__new__(cls)
        obj._value_ = auto()
        obj.description = description
        obj.func = func
        return obj

    def reduce_operation_with_scatter(
        self,
        operation_lhs: Any,
        initial_tensor: torch.Tensor,
        dim: int,
        index_tensor: torch.Tensor,
        src_tensor: torch.Tensor,
    ) -> Any:
        scatter_tensor = None
        if self == ReduceOperation.SUM or self == ReduceOperation.MEAN:
            scatter_tensor = torch.zeros_like(initial_tensor)
        elif self == ReduceOperation.PROD:
            scatter_tensor = torch.ones_like(initial_tensor)
        elif self == ReduceOperation.AMIN or self == ReduceOperation.AMAX:
            scatter_tensor = initial_tensor
        else:
            # This case would not be encountered from torch itself
            raise ValueError(f"Invalid Operation for Reduce op: {self}")

        operation_rhs = torch.scatter(scatter_tensor, dim, index_tensor, src_tensor)
        device = to_torch_device(scatter_tensor.device)
        operation_lhs = operation_lhs.to(device)
        operation_rhs = operation_rhs.to(device)
        return self.func(operation_lhs, operation_rhs)


@register_torch_trt_decomposition(
    torch.ops.aten.scatter_reduce.two, registry=TORCH_TRT_DECOMPOSITIONS
)
def scatter_reduce_decomposition(
    input_tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src_tensor: torch.Tensor,
    reduce: str,
    include_self: bool = True,
) -> torch.Tensor:
    scatter_loop_tensor = input_tensor
    device_input_tensor = input_tensor.device
    # required for mean reduce operation
    scatter_count_tensor = torch.zeros_like(input_tensor)
    src_shape = list(src_tensor.shape)
    src_dim = src_shape[dim]
    if not include_self:
        raise AssertionError("include_self False for scatter reduce not yet supported")
    for i in range(0, src_dim):
        src_slice = torch.select(src_tensor, dim, i)
        index_slice = torch.select(index, dim, i)
        # unsqueeze src and index in dim
        src_slice = torch.unsqueeze(src_slice, dim)
        index_slice = torch.unsqueeze(index_slice, dim)

        # moving tensor to default device
        scatter_loop_tensor = scatter_loop_tensor.to(device_input_tensor)
        index_slice = index_slice.to(device_input_tensor)
        src_slice = src_slice.to(device_input_tensor)
        if reduce == "sum":
            reduceOp = ReduceOperation.SUM
        elif reduce == "prod":
            reduceOp = ReduceOperation.PROD
        elif reduce == "mean":
            reduceOp = ReduceOperation.MEAN
            scatter_count_tensor = reduceOp.reduce_operation_with_scatter(
                scatter_count_tensor,
                input_tensor,
                dim,
                index_slice,
                torch.ones_like(src_slice),
            )
        elif reduce == "amax":
            reduceOp = ReduceOperation.AMAX
        elif reduce == "amin":
            reduceOp = ReduceOperation.AMIN
        scatter_loop_tensor = reduceOp.reduce_operation_with_scatter(
            scatter_loop_tensor, input_tensor, dim, index_slice, src_slice
        )
    if reduce == "mean":
        scatter_loop_tensor = torch.div(
            scatter_loop_tensor,
            torch.add(scatter_count_tensor, torch.ones_like(scatter_count_tensor)),
            rounding_mode="trunc",
        )
    return scatter_loop_tensor


@register_torch_trt_decomposition(aten._log_softmax, registry=TORCH_TRT_DECOMPOSITIONS)
def log_softmax_decomposition(
    x: torch.Tensor,
    dim: int,
    half_to_float: bool,
) -> torch.Tensor:
    return torch.log(
        torch.softmax(x, dim, dtype=torch.float if half_to_float else None)
    )


@register_torch_trt_decomposition(aten.instance_norm, registry=TORCH_TRT_DECOMPOSITIONS)
def instance_norm_decomposition(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    use_input_stats: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> torch.Tensor:
    if use_input_stats:
        return torch.nn.functional.group_norm(input, input.shape[1], weight, bias, eps)
    else:
        return torch.nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias, False, momentum, eps
        )


@register_torch_trt_decomposition(
    torch.ops.aten.full_like, registry=TORCH_TRT_DECOMPOSITIONS
)
def full_like_decomposition(*args: Any, **kwargs: Any) -> torch.Tensor:
    input = args[0]
    shape = args[0].shape
    fill_value = args[1]
    kwargs["dtype"] = kwargs.get("dtype", None) or input.dtype
    kwargs["device"] = kwargs.get("device", None) or input.device
    return torch.full(shape, fill_value, dtype=kwargs["dtype"], device=kwargs["device"])


@register_torch_trt_decomposition(aten.view.default, registry=TORCH_TRT_DECOMPOSITIONS)
def view_decomposition(x: torch.Tensor, size: List[torch.SymInt]) -> torch.Tensor:
    return aten._reshape_copy.default(x, size)


@register_torch_trt_decomposition(
    aten.scaled_dot_product_attention, registry=TORCH_TRT_DECOMPOSITIONS
)
def scaled_dot_product_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    device = query.device

    if is_causal or attn_mask is not None:
        attn_bias = torch.zeros((L, S), dtype=query.dtype, device=device)

    if is_causal:
        assert attn_mask is None, "attn_mask must be None when is_causal=True"
        temp_mask = torch.ones((L, S), dtype=torch.bool, device=device).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = torch.matmul(query, key.transpose(-2, -1))

    if scale is None:
        scale = torch.sqrt(torch.scalar_tensor(query.size(-1), dtype=torch.int)).to(
            query.dtype
        )
        attn_weight = attn_weight / scale
    else:
        attn_weight = attn_weight * scale

    if is_causal or attn_mask is not None:
        # We only add attn_bias when we have to, otherwise this will have a negative impact on the performance even it's 0.
        attn_weight = attn_weight + attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, value)


@register_torch_trt_decomposition(
    aten._scaled_dot_product_flash_attention, registry=TORCH_TRT_DECOMPOSITIONS
)
def scaled_dot_product_flash_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.SymInt,
    torch.SymInt,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    attn = scaled_dot_product_attention_decomposition(
        query, key, value, None, dropout_p, is_causal, scale=scale
    )
    return attn, None, None, None, 0, 0, None, None, None


@register_torch_trt_decomposition(
    aten._scaled_dot_product_efficient_attention, registry=TORCH_TRT_DECOMPOSITIONS
)
def scaled_dot_product_efficient_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn = scaled_dot_product_attention_decomposition(
        query, key, value, attn_bias, dropout_p, is_causal, scale=scale
    )
    return attn, None, None, None


@register_torch_trt_decomposition(
    aten._scaled_dot_product_cudnn_attention, registry=TORCH_TRT_DECOMPOSITIONS
)
def scaled_dot_product_cudnn_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.SymInt,
    torch.SymInt,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    attn = scaled_dot_product_attention_decomposition(
        query, key, value, attn_bias, dropout_p, is_causal, scale=scale
    )
    return attn, None, None, None, 0, 0, None, None, None


@register_torch_trt_decomposition(
    aten.cudnn_grid_sampler, registry=TORCH_TRT_DECOMPOSITIONS
)
def cudnn_grid_sampler_decomposition(
    x: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    return torch.grid_sampler_2d(x, grid, 0, 0, True)


@register_torch_trt_decomposition(
    aten.masked_scatter, registry=TORCH_TRT_DECOMPOSITIONS
)
def masked_scatter_decomposition(
    input: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """
    Decomposition of `aten.masked_scatter` for TensorRT.

    Emulates the behavior of `input[mask] = source` using only TensorRT-compatible ops.

    Steps:
      1) Broadcast `input` and `mask` to a common shape.
      2) Flatten all tensors for uniform indexing.
      3) Compute gather indices for `source` by applying cumsum to the boolean mask.
         - Use `masked_fill` to avoid invalid indices in positions where `mask` is False.
      4) Gather values from `source` at valid positions.
      5) Use `torch.where` to insert gathered values into `input` where `mask` is True.
      6) Reshape the result back to the original broadcasted shape.
    """

    # 1) Broadcast input and mask to the same shape
    input_b, mask_b = aten.broadcast_tensors([input, mask])

    # 2) Flatten tensors for element-wise operations
    input_flat = input_b.flatten()
    mask_flat = mask_b.flatten()
    source_flat = source.flatten()

    # 3) Compute gather indices from cumsum of the mask
    # Subtract 1 so that the first True position maps to index 0 in source
    source_idx = mask_flat.cumsum(0) - 1
    # Set gather index to 0 where mask is False (these will be ignored later)
    safe_idx = source_idx.masked_fill(~mask_flat, 0)

    # 4) Gather values from source using computed indices
    gathered = source_flat.gather(0, safe_idx)

    # 5) Replace masked positions in input with gathered values
    replaced = torch.where(mask_flat, gathered, input_flat)

    # 6) Reshape the result to match the original broadcasted shape
    return replaced.view(input_b.shape)


def get_decompositions(
    enable_experimental_decompositions: bool = False,
) -> Dict[OpOverload, Callable[[Any], Any]]:
    if enable_experimental_decompositions:
        CORE_ATEN_DECOMPOSITIONS_FILTERED: Dict[OpOverload, Callable[[Any], Any]] = {
            decomp: _core_aten_decompositions[decomp]
            for decomp in _core_aten_decompositions
            if decomp not in torch_disabled_decompositions
        }
        return {**CORE_ATEN_DECOMPOSITIONS_FILTERED, **TORCH_TRT_DECOMPOSITIONS}
    else:
        # changes made here due to torch2.6 changes https://github.com/pytorch/pytorch/pull/135080
        decomp_table = {}
        for op in _collect_all_valid_cia_ops_for_aten_namespace():
            decomp_table[op] = _get_decomp_for_cia(op)

        DECOMP_TABLE_FILTERED: Dict[OpOverload, Callable[[Any], Any]] = {
            decomp: decomp_table[decomp]
            for decomp in decomp_table
            if decomp not in torch_disabled_decompositions
        }

        return {
            **ENABLED_TORCH_DECOMPOSITIONS,
            **DECOMP_TABLE_FILTERED,
            **TORCH_TRT_DECOMPOSITIONS,
        }
