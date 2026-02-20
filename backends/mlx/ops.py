#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Op Handlers - registered handlers for converting ATen/custom ops to MLX.

This module contains all the op handler functions registered with the MLXOpRegistry.
Each handler converts a specific PyTorch operation to the corresponding MLX graph node.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.backends.mlx.program_builder import (
    emit_stop_position,
    IntOrVid,
    MLXProgramBuilder,
    parse_dequant_node,
    REGISTRY,
    Slot,
    to_mlx_qparams,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AbsNode,
    AddIntNode,
    AddmmNode,
    AddNode,
    ARangeNode,
    ArccoshNode,
    ArccosNode,
    ArcsinhNode,
    ArcsinNode,
    ArctanhNode,
    ArctanNode,
    ArgmaxNode,
    ArgminNode,
    AsStridedNode,
    AsTypeNode,
    Atan2Node,
    BroadcastToNode,
    CeilNode,
    ConcatenateNode,
    ContiguousNode,
    Conv1DNode,
    Conv2DNode,
    Conv3DNode,
    CoshNode,
    CosNode,
    DequantizeNode,
    DivideNode,
    EqualNode,
    ErfNode,
    ExpandDimsNode,
    Expm1Node,
    ExpNode,
    FloatOrVid,
    FloorDivideIntNode,
    FloorDivideNode,
    FloorNode,
    FullLikeNode,
    FullNode,
    GatherNode,
    GeluNode,
    GreaterEqualNode,
    GreaterNode,
    IdCopyNode,
    ItemIntNode,
    LayerNormNode,
    LessEqualNode,
    LessNode,
    LinearNode,
    Log10Node,
    Log1pNode,
    Log2Node,
    LogAddExpNode,
    LogicalAndNode,
    LogicalNotNode,
    LogicalOrNode,
    LogNode,
    LogSumExpNode,
    MaximumNode,
    MaxNode,
    MeanNode,
    MinimumNode,
    MinNode,
    MultiplyIntNode,
    MultiplyNode,
    NegNode,
    NotEqualNode,
    PadNode,
    PowerNode,
    ProdNode,
    ReciprocalNode,
    ReshapeNode,
    RMSNormNode,
    RopeNode,
    RoundNode,
    RsqrtNode,
    SigmoidNode,
    SiluNode,
    SinhNode,
    SinNode,
    SliceNode,
    SliceUpdateNode,
    SoftmaxNode,
    SplitNode,
    SqrtNode,
    SquareNode,
    SqueezeNode,
    StdNode,
    SubtractIntNode,
    SubtractNode,
    SumNode,
    SymSizeNode,
    TakeAlongAxisNode,
    TakeNode,
    TanhNode,
    TanNode,
    TidOrVid,
    TileNode,
    TransposeNode,
    TrilNode,
    TriuNode,
    VarNode,
    WhereNode,
)
from torch.fx.node import Node


# =============================================================================
# Parameter validation utilities
# =============================================================================


def require_static_int(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static integer (not a Slot/SymInt).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, int):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static int32 value, but got {value} (type={type(value).__name__})."
        )


def require_static_float(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static float (not a Slot/SymFloat).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, (int, float)):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static float value, but got {value} (type={type(value).__name__})."
        )


def require_static_ints(
    values: Union[List[Any], Any], param_name: str, op_name: str
) -> None:
    """
    Validate that all values in a list are static integers (not Slots/SymInts).

    Raises NotImplementedError if any value is dynamic.

    Args:
        values: List of values to check, or a single value
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if not isinstance(values, list):
        values = [values]

    for v in values:
        require_static_int(v, param_name, op_name)


def require_args(
    args: List[Any],
    min_count: int,
    max_count: int,
    op_name: str,
) -> None:
    """
    Validate that args count is within expected range.

    Raises ValueError if the count is outside the expected range.

    Args:
        args: The handler args list
        min_count: Minimum number of args expected
        max_count: Maximum number of args expected
        op_name: Name of the operation (for error message)
    """
    if not (min_count <= len(args) <= max_count):
        if min_count == max_count:
            raise ValueError(f"{op_name}: expected {min_count} args, got {len(args)}")
        raise ValueError(
            f"{op_name}: expected {min_count}-{max_count} args, got {len(args)}"
        )


def require_kwargs(
    kwargs: Dict[str, Any],
    allowed: Set[str],
    op_name: str,
) -> None:
    """
    Validate that only allowed kwargs are present.

    Raises ValueError if unexpected kwargs are found.

    Args:
        kwargs: The handler kwargs dict
        allowed: Set of allowed kwarg names
        op_name: Name of the operation (for error message)
    """
    unexpected = set(kwargs.keys()) - allowed
    if unexpected:
        raise ValueError(f"{op_name}: unexpected kwargs: {unexpected}")


def require_contiguous_format(
    *,
    layout=None,
    memory_format=None,
    dim_order=None,
    op_name: str,
) -> None:
    """
    Validate that layout/memory_format/dim_order specify contiguous format.

    MLX only supports contiguous (strided) tensors. Raises ValueError if
    sparse layouts or non-contiguous memory formats are requested.

    Args:
        layout: The torch layout (e.g., torch.strided, torch.sparse_coo)
        memory_format: The torch memory format (e.g., torch.contiguous_format,
            torch.channels_last)
        dim_order: The dimension order (list of ints, identity = contiguous)
        op_name: Name of the operation (for error message)
    """
    if layout is not None and layout != torch.strided:
        raise ValueError(f"{op_name}: only strided layout supported, got {layout}")

    if memory_format is not None and memory_format not in (
        torch.contiguous_format,
        torch.preserve_format,
    ):
        raise ValueError(
            f"{op_name}: only contiguous memory format supported, got {memory_format}"
        )

    if dim_order is not None:
        if list(dim_order) != list(range(len(dim_order))):
            raise ValueError(
                f"{op_name}: only contiguous dim_order supported, got {dim_order}"
            )


def is_static_value(value: Any) -> bool:
    """
    Check if a value is static (not a Slot/SymInt).

    Returns:
        True if the value is a static scalar (int, float, bool), False otherwise
    """
    return not isinstance(value, Slot)


def normalize_reduction_dim(
    args: List[Any], start_idx: int = 1
) -> Tuple[Optional[List[int]], bool]:
    """
    Normalize dim argument for reduction operations.

    Extracts and normalizes the dim argument from handler args, returning a list of axes
    and the keepdim flag. Handles both list-based dims (e.g., sum.dim_IntList) and
    single int dims (e.g., prod.dim_int).

    Args:
        args: The handler args list
        start_idx: Index where the dim argument starts (default 1, after self)

    Returns:
        Tuple of (axes, keepdim) where:
        - axes: List of dimension indices, or empty list for reduce-all
        - keepdim: Boolean keepdim flag (default False)
    """
    if len(args) > start_idx and isinstance(args[start_idx], (list, tuple)):
        dim = list(args[start_idx])
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    elif len(args) > start_idx and isinstance(args[start_idx], int):
        dim = [args[start_idx]]
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    else:
        dim = []
        keepdim = False

    return dim, keepdim


# =============================================================================
# Basic ops
# =============================================================================


@REGISTRY.register(target=["NOOP", torch.ops.aten._assert_scalar.default])
def _noop_handler(P: MLXProgramBuilder, n: Node) -> None:
    """No-op handler for nodes that don't emit any MLX instructions."""
    return None


# Handler for auto_functionalized_v2 higher-order op
# This handles mutating ops that have been functionalized
@REGISTRY.register(target=[torch.ops.higher_order.auto_functionalized_v2])
def _auto_functionalized_v2_handler(P: MLXProgramBuilder, n: Node):
    """
    Handler for auto_functionalized_v2 higher-order op.

    auto_functionalized_v2 wraps mutating ops after functionalization.
    It returns a tuple of (token, mutated_values...).

    This handler emits the actual lowering instructions and returns a tuple
    of slots that getitem can index into.

    Currently supported wrapped ops:
    - llama.update_cache.default
    """
    if len(n.args) < 1:
        raise ValueError(
            f"auto_functionalized_v2 requires at least 1 arg, got {len(n.args)}"
        )

    wrapped_op = n.args[0]
    wrapped_op_str = str(wrapped_op)

    # Check which op is wrapped
    if "llama" in wrapped_op_str and "update_cache" in wrapped_op_str:
        result = _handle_update_cache(P, n)
        # Register the result tuple with this node so getitem can find it
        P.slot_manager.set_slot(n, result)
        return result

    # Unknown wrapped op - not supported
    raise NotImplementedError(
        f"auto_functionalized_v2 wrapping '{wrapped_op}' is not supported. "
        f"Only llama.update_cache is currently supported."
    )


def _handle_update_cache(P: MLXProgramBuilder, n: Node) -> Tuple[Slot, Slot]:
    """
    Handle auto_functionalized_v2(llama.update_cache, ...).

    Emits SliceUpdateNode and returns tuple of (token_slot, cache_slot).

    This is for the direct case where:
    - cache is [B, S, H, D]
    - value (update) is [B, S_step, H, D]
    - We emit SliceUpdateNode on axis=1 (the S dimension)
    """
    kwargs = n.kwargs
    value_node = kwargs.get("value")
    start_pos = kwargs.get("start_pos", 0)
    all_bases = kwargs.get("_all_bases", [])

    if not value_node or not all_bases:
        raise ValueError("update_cache handler: missing value or _all_bases in kwargs")

    cache_node = all_bases[0]

    return _emit_update_cache(P, value_node, cache_node, start_pos)


def _emit_update_cache(
    P: MLXProgramBuilder,
    value_node: Node,
    cache_node: Node,
    start_pos,
) -> Tuple[Slot, Slot]:
    """
    Shared logic for emitting SliceUpdateNode for KV cache updates.

    Args:
        P: MLXProgramBuilder
        value_node: Node for the value tensor [B, S_step, H, D]
        cache_node: Node for the cache tensor [B, S, H, D]
        start_pos: Start position (int or Node)

    Returns:
        Tuple of (token_slot, cache_slot)
    """
    # Get slots
    cache_slot = P.slot_map([cache_node])[0]
    value_slot = P.slot_map([value_node])[0]

    # Handle start_pos - could be int or Node
    if isinstance(start_pos, Node):
        start_slot = P.slot_map([start_pos])[0]
    else:
        start_slot = start_pos

    # Calculate stop = start + seq_len
    # value is [B, S_step, H, D], so seq_len is dim 1
    value_meta = value_node.meta.get("val")
    stop_slot = emit_stop_position(
        P,
        start=start_slot,
        length_tensor=value_slot,
        length_dim=1,  # S_step is dim 1 in [B, S_step, H, D]
        length_meta=value_meta,
    )

    # Emit SliceUpdateNode on axis=1
    # cache is [B, S, H, D], value is [B, S_step, H, D]
    # This updates cache[:, start:stop, :, :] = value in-place
    P.emit(
        SliceUpdateNode(
            dst=P.slot_to_tid(cache_slot),
            update=P.slot_to_tid(value_slot),
            axis=IntOrVid.from_literal(1),  # S dimension in [B, S, H, D]
            start=P.to_int_or_vid(start_slot),
            stop=P.to_int_or_vid(stop_slot),
        )
    )

    # Return tuple of (token, updated_cache)
    # - token_slot: create a placeholder (token is not actually used)
    # - cache_slot: the cache that was updated in-place by SliceUpdateNode
    _, token_slot = P.make_tmp_slot()

    # The token is a dummy value that's not used. We emit an IdCopyNode
    # from value to token just to have something valid there.
    P.emit(
        IdCopyNode(
            x=P.slot_to_tid(value_slot),  # Copy from value as a placeholder
            out=P.slot_to_tid(token_slot),
        )
    )

    return (token_slot, cache_slot)


# Import custom ops to register llama.update_cache
try:
    from executorch.extension.llm.custom_ops import (  # noqa: F401
        custom_ops as _llama_ops,
    )
except ImportError:
    pass  # Custom ops not available


@REGISTRY.register(target=[torch.ops.llama.update_cache.default])
def _llama_update_cache_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle direct llama.update_cache.default calls.

    Args:
        n.args[0]: value tensor [B, S_step, H, D]
        n.args[1]: cache tensor [B, S, H, D]
        n.args[2]: start_pos (scalar)

    Returns dummy token slot.
    """
    value_node = n.args[0]
    cache_node = n.args[1]
    start_pos = n.args[2]

    token_slot, _ = _emit_update_cache(P, value_node, cache_node, start_pos)
    return token_slot


@REGISTRY.register(target=[torch.ops.aten.linear.default])
def _linear_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 3, "aten.linear")
    require_kwargs(P.kwargs(n), set(), "aten.linear")
    x, w = args[0], args[1]
    b = args[2] if len(args) > 2 else None
    out = P.make_or_get_slot(n)

    P.emit(
        LinearNode(
            x=P.slot_to_tid(x),
            weight=P.slot_to_tid(w),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(b) if b else None,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.addmm.default])
def _addmm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle addmm: self + (mat1 @ mat2).

    addmm(self, mat1, mat2, *, beta=1, alpha=1) computes:
        beta * self + alpha * (mat1 @ mat2)

    This is typically the result of decomposing linear(x, w, b) in Edge IR:
        permute(w) -> addmm(b, x, permuted_w)

    For the common case where beta=1 and alpha=1, this is equivalent to:
        mat1 @ mat2 + self

    We use AddmmNode which calls matmul directly (no transposition needed).
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 3, 3, "aten.addmm")
    require_kwargs(kwargs, {"beta", "alpha"}, "aten.addmm")
    bias, mat1, mat2 = args[0], args[1], args[2]

    # Get kwargs for beta and alpha (default to 1)
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    out = P.make_or_get_slot(n)

    # Emit AddmmNode with alpha and beta parameters
    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(bias),
            alpha=float(alpha),
            beta=float(beta),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.mm.default])
def _mm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle mm: mat1 @ mat2 (matrix multiplication without bias).

    mm(input, mat2) computes: input @ mat2

    We emit AddmmNode with bias=None, which will use the matmul-only path
    in exec_addmm, avoiding the fused addmm operation.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.mm")
    require_kwargs(P.kwargs(n), set(), "aten.mm")
    mat1, mat2 = args[0], args[1]

    out = P.make_or_get_slot(n)

    # Emit AddmmNode with no bias: uses matmul directly
    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=None,  # No bias - will use matmul path
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.bmm.default])
def _bmm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle bmm: batch matrix multiplication.

    bmm(input, mat2) computes batched matrix multiplication where both inputs are 3D.
    For example, if input is [B, N, M] and mat2 is [B, M, P], the result is [B, N, P].

    MLX's matmul naturally handles batched operations, so we emit AddmmNode with
    bias=None which uses the matmul path in exec_addmm.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.bmm")
    require_kwargs(P.kwargs(n), set(), "aten.bmm")
    mat1, mat2 = args[0], args[1]

    out = P.make_or_get_slot(n)

    # Emit AddmmNode with no bias: uses matmul which handles 3D+ tensors
    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=None,  # No bias - matmul handles batched operations
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.view.default, torch.ops.aten.view_copy.default]
)
def _view_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.view")
    require_kwargs(P.kwargs(n), set(), "aten.view")
    x, shape = args
    out = P.make_or_get_slot(n)

    shape_iovs = [P.to_int_or_vid(s) for s in shape]
    P.emit(
        ReshapeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            shape=shape_iovs,
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.alias_copy.default,
    ]
)
def _clone_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.clone")
    require_kwargs(kwargs, {"memory_format"}, "aten.clone")
    require_contiguous_format(
        memory_format=kwargs.get("memory_format"),
        op_name="aten.clone",
    )
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        ContiguousNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.copy.default])
def _copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.copy - copy data from src to self.

    Schema: aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
    In functionalized Edge IR, this returns a copy of src (args[1]).
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.copy")
    require_kwargs(P.kwargs(n), {"non_blocking"}, "aten.copy")
    src = args[1]
    out = P.make_or_get_slot(n)
    P.emit(
        ContiguousNode(
            x=P.slot_to_tid(src),
            out=P.slot_to_tid(out),
        )
    )
    return out


# Handle Edge IR's dim_order_ops._clone_dim_order (memory layout clone)
# Note: We need to import the EdgeOpOverload to register this properly
try:
    from executorch.exir.dialects._ops import ops as exir_ops

    _dim_order_clone_target = exir_ops.edge.dim_order_ops._clone_dim_order.default

    @REGISTRY.register(target=[_dim_order_clone_target])
    def _dim_order_clone_handler(P: MLXProgramBuilder, n: Node) -> Slot:
        # dim_order_ops._clone_dim_order(Tensor self, *, bool non_blocking=False, int[]? dim_order=None) -> Tensor
        # This is essentially a contiguous/clone operation for memory layout
        args = P.args(n)
        kwargs = P.kwargs(n)
        require_args(args, 1, 1, "dim_order_ops._clone_dim_order")
        require_kwargs(
            kwargs, {"non_blocking", "dim_order"}, "dim_order_ops._clone_dim_order"
        )
        require_contiguous_format(
            dim_order=kwargs.get("dim_order"),
            op_name="dim_order_ops._clone_dim_order",
        )
        x = args[0]
        out = P.make_or_get_slot(n)
        P.emit(
            ContiguousNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
            )
        )
        return out

    # Handle Edge IR's dim_order_ops._to_dim_order_copy (dtype conversion)
    # This is what x.to(dtype) becomes after to_edge() transformation
    _dim_order_copy_target = exir_ops.edge.dim_order_ops._to_dim_order_copy.default

    @REGISTRY.register(target=[_dim_order_copy_target])
    def _dim_order_copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
        # dim_order_ops._to_dim_order_copy(Tensor self, *, ScalarType? dtype=None, ...)
        # If dtype is specified, this is a dtype conversion (use AsTypeNode)
        # If dtype is None/same, this is just a memory layout copy (use ContiguousNode)
        args = P.args(n)
        kwargs = P.kwargs(n)
        require_args(args, 1, 1, "dim_order_ops._to_dim_order_copy")
        require_kwargs(
            kwargs,
            {"dtype", "device", "layout", "non_blocking", "dim_order"},
            "dim_order_ops._to_dim_order_copy",
        )
        require_contiguous_format(
            layout=kwargs.get("layout"),
            dim_order=kwargs.get("dim_order"),
            op_name="dim_order_ops._to_dim_order_copy",
        )
        x = args[0]
        out = P.make_or_get_slot(n)

        dtype = kwargs.get("dtype")
        if dtype is not None:
            # Dtype conversion
            P.emit(
                AsTypeNode(
                    x=P.slot_to_tid(x),
                    out=P.slot_to_tid(out),
                    scalar_type=torch_dtype_to_scalar_type(dtype),
                )
            )
        else:
            # No dtype change, just memory layout (contiguous)
            P.emit(
                ContiguousNode(
                    x=P.slot_to_tid(x),
                    out=P.slot_to_tid(out),
                )
            )
        return out

except ImportError:
    # Edge IR ops not available (e.g., when building from ATen dialect)
    pass


@REGISTRY.register(target=[torch.ops.aten._to_copy.default])
def _to_copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten._to_copy - lower-level dtype/device conversion."""
    # aten._to_copy(Tensor self, *, ScalarType? dtype=None, ...)
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten._to_copy")
    require_kwargs(
        kwargs, {"dtype", "device", "layout", "memory_format"}, "aten._to_copy"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        memory_format=kwargs.get("memory_format"),
        op_name="aten._to_copy",
    )
    x = args[0]
    out = P.make_or_get_slot(n)

    dtype = kwargs.get("dtype")
    if dtype is not None:
        # Dtype conversion
        P.emit(
            AsTypeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
                scalar_type=torch_dtype_to_scalar_type(dtype),
            )
        )
    else:
        # No dtype change, just copy (use contiguous)
        P.emit(
            ContiguousNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten.embedding.default])
def _embedding_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    # aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
    #                 bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
    # padding_idx is only relevant for training (gradient computation)
    # scale_grad_by_freq and sparse are also training-only
    args = P.args(n)
    require_args(args, 2, 3, "aten.embedding")
    require_kwargs(P.kwargs(n), set(), "aten.embedding")
    w, x = args[0], args[1]
    # padding_idx (args[2] if present) is ignored - only affects gradients
    out = P.make_or_get_slot(n)
    P.emit(
        GatherNode(
            table_=P.slot_to_tid(w),
            ids=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.add.Tensor])
def _add_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.add.Tensor")
    require_kwargs(P.kwargs(n), set(), "aten.add.Tensor")
    a, b = args

    # Check if both inputs are scalars (not tensors)
    # We can't support scalar + scalar because:
    # 1. Scalars get lifted to tensors during export
    # 2. But the return type would be a tensor, not a scalar
    # 3. ExecuTorch would expect a scalar return value
    if is_static_value(a) and is_static_value(b):
        raise ValueError(
            "aten.add.Tensor with both scalar inputs is not supported. "
            "Use operator.add for scalar arithmetic."
        )

    out = P.make_or_get_slot(n)
    P.emit(
        AddNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.add])
def _add_scalar_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle Python operator.add for scalar arithmetic (symbolic shapes)."""
    args = P.args(n)
    require_args(args, 2, 2, "operator.add")
    require_kwargs(P.kwargs(n), set(), "operator.add")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        AddIntNode(
            a=P.to_int_or_vid(a),
            b=P.to_int_or_vid(b),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.sub])
def _sub_scalar_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle Python operator.sub for scalar arithmetic (symbolic shapes)."""
    args = P.args(n)
    require_args(args, 2, 2, "operator.sub")
    require_kwargs(P.kwargs(n), set(), "operator.sub")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        SubtractIntNode(
            a=P.to_int_or_vid(a),
            b=P.to_int_or_vid(b),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.mul])
def _mul_scalar_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle Python operator.mul for scalar arithmetic (symbolic shapes)."""
    args = P.args(n)
    require_args(args, 2, 2, "operator.mul")
    require_kwargs(P.kwargs(n), set(), "operator.mul")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        MultiplyIntNode(
            a=P.to_int_or_vid(a),
            b=P.to_int_or_vid(b),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.floordiv])
def _floordiv_scalar_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle Python operator.floordiv (//) for scalar arithmetic (symbolic shapes)."""
    args = P.args(n)
    require_args(args, 2, 2, "operator.floordiv")
    require_kwargs(P.kwargs(n), set(), "operator.floordiv")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        FloorDivideIntNode(
            a=P.to_int_or_vid(a),
            b=P.to_int_or_vid(b),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def _mul_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.mul.Tensor and aten.mul.Scalar."""
    out = P.make_or_get_slot(n)
    args = P.args(n)
    require_args(args, 2, 2, "aten.mul")
    require_kwargs(P.kwargs(n), set(), "aten.mul")
    a = args[0]
    b = args[1]

    # Handle scalar b by creating a constant tensor
    if not isinstance(b, Slot):
        b = P.make_or_get_constant(
            f"_scalar_{b}", torch.tensor([b], dtype=n.meta["val"].dtype)
        )

    # Handle scalar a (for commutative mul)
    if not isinstance(a, Slot):
        a = P.make_or_get_constant(
            f"_scalar_{a}", torch.tensor([a], dtype=n.meta["val"].dtype)
        )

    P.emit(
        MultiplyNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar])
def _div_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.div.Tensor and aten.div.Scalar."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.div")
    require_kwargs(P.kwargs(n), set(), "aten.div")
    out = P.make_or_get_slot(n)
    a = args[0]
    b = args[1]

    # Handle scalar b by creating a constant tensor
    if not isinstance(b, Slot):
        b = P.make_or_get_constant(
            f"_scalar_{b}", torch.tensor([b], dtype=n.meta["val"].dtype)
        )

    # Handle scalar a
    if not isinstance(a, Slot):
        a = P.make_or_get_constant(
            f"_scalar_{a}", torch.tensor([a], dtype=n.meta["val"].dtype)
        )

    P.emit(
        DivideNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.div.Tensor_mode])
def _div_tensor_mode_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.div.Tensor_mode with rounding mode."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 2, 2, "aten.div.Tensor_mode")
    require_kwargs(kwargs, {"rounding_mode"}, "aten.div.Tensor_mode")
    out = P.make_or_get_slot(n)
    a = args[0]
    b = args[1]
    rounding_mode = kwargs.get("rounding_mode", None)

    # Handle scalar b by creating a constant tensor
    if not isinstance(b, Slot):
        b = P.make_or_get_constant(
            f"_scalar_{b}", torch.tensor([b], dtype=n.meta["val"].dtype)
        )

    # Handle scalar a
    if not isinstance(a, Slot):
        a = P.make_or_get_constant(
            f"_scalar_{a}", torch.tensor([a], dtype=n.meta["val"].dtype)
        )

    if rounding_mode == "trunc":
        raise NotImplementedError(
            "aten.div.Tensor_mode with rounding_mode='trunc' is not supported. "
            "MLX does not have a truncate operation."
        )
    elif rounding_mode == "floor":
        P.emit(
            FloorDivideNode(
                a=P.slot_to_tid(a),
                b=P.slot_to_tid(b),
                out=P.slot_to_tid(out),
            )
        )
    else:
        # rounding_mode is None - true division
        P.emit(
            DivideNode(
                a=P.slot_to_tid(a),
                b=P.slot_to_tid(b),
                out=P.slot_to_tid(out),
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten.silu.default])
def _silu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 1, 1, "aten.silu")
    require_kwargs(P.kwargs(n), set(), "aten.silu")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        SiluNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sigmoid.default])
def _sigmoid_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 1, 1, "aten.sigmoid")
    require_kwargs(P.kwargs(n), set(), "aten.sigmoid")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        SigmoidNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.tanh.default])
def _tanh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle tanh activation function.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Returns values in range [-1, 1].
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.tanh")
    require_kwargs(P.kwargs(n), set(), "aten.tanh")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        TanhNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten._softmax.default])
def _softmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle softmax: computes softmax along the specified dimension.

    aten._softmax(self, dim, half_to_float) computes:
        softmax(self, axis=dim)

    The half_to_float parameter is for type conversion and is ignored for MLX.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten._softmax")
    require_kwargs(P.kwargs(n), set(), "aten._softmax")
    x, dim, _ = args[0], args[1], args[2]  # half_to_float is unused for MLX

    out = P.make_or_get_slot(n)

    # Emit SoftmaxNode with the specified axis
    P.emit(
        SoftmaxNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.gelu.default])
def _gelu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.gelu")
    require_kwargs(kwargs, {"approximate"}, "aten.gelu")
    (x,) = args
    # GELU approximate mode: 'none' (default) or 'tanh'
    approximate = kwargs.get("approximate", "none")
    out = P.make_or_get_slot(n)
    P.emit(
        GeluNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            approximate=approximate,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default]
)
def _permute_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.permute")
    require_kwargs(P.kwargs(n), set(), "aten.permute")
    x, dims = args
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            perm=list(dims),
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.transpose.int, torch.ops.aten.transpose_copy.int]
)
def _transpose_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 3, 3, "aten.transpose")
    require_kwargs(P.kwargs(n), set(), "aten.transpose")
    x, dim0, dim1 = args
    perm = list(range(len(n.meta["val"].shape)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            perm=perm,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor]
)
def _slice_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 4, 4, "aten.slice")
    require_kwargs(P.kwargs(n), set(), "aten.slice")
    x, dim, start, stop = args
    if start is None:
        start = 0
    out = P.make_or_get_slot(n)
    P.emit(
        SliceNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=P.to_int_or_vid(dim),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.narrow.default])
def _narrow_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle narrow(input, dim, start, length) -> slice(input, dim, start, start+length).

    This is needed for KV cache updates with dynamic positions where narrow
    is preferred over slice syntax for better torch.export compatibility.
    """
    args = P.args(n)
    require_args(args, 4, 4, "aten.narrow")
    require_kwargs(P.kwargs(n), set(), "aten.narrow")
    x, dim, start, length = args
    out = P.make_or_get_slot(n)

    # Convert narrow (start, length) to slice (start, end)
    # The end is start + length
    start_iov = P.to_int_or_vid(start)
    length_iov = P.to_int_or_vid(length)

    # For stop = start + length, we need to emit an ADD_SCALAR if either is a Vid
    if isinstance(start_iov, IntOrVid) and start_iov.vid is not None:
        # start is a Vid, need to add at runtime
        if isinstance(length_iov, IntOrVid) and length_iov.vid is not None:
            # Both are Vids - emit add to compute stop
            _, stop_slot = P.make_tmp_value_slot()
            stop_vid = P.slot_to_vid(stop_slot)
            P.emit(
                AddIntNode(
                    a=start_iov.vid,
                    b=length_iov.vid,
                    out=stop_vid,
                )
            )
            stop_iov = IntOrVid(int64=None, vid=stop_vid)
        else:
            # start is Vid, length is int - emit add scalar
            _, stop_slot = P.make_tmp_value_slot()
            stop_vid = P.slot_to_vid(stop_slot)
            P.emit(
                AddIntNode(
                    a=start_iov.vid,
                    b=(
                        length_iov.int64
                        if isinstance(length_iov, IntOrVid)
                        else length_iov
                    ),
                    out=stop_vid,
                )
            )
            stop_iov = IntOrVid(int64=None, vid=stop_vid)
    elif isinstance(length_iov, IntOrVid) and length_iov.vid is not None:
        # length is Vid, start is int - emit add scalar
        start_val = start_iov.int64 if isinstance(start_iov, IntOrVid) else start_iov
        _, stop_slot = P.make_tmp_value_slot()
        stop_vid = P.slot_to_vid(stop_slot)
        P.emit(
            AddIntNode(
                a=length_iov.vid,
                b=start_val,
                out=stop_vid,
            )
        )
        stop_iov = IntOrVid(int64=None, vid=stop_vid)
    else:
        # Both are concrete ints
        start_val = start_iov.int64 if isinstance(start_iov, IntOrVid) else start_iov
        length_val = (
            length_iov.int64 if isinstance(length_iov, IntOrVid) else length_iov
        )
        stop_iov = IntOrVid(int64=start_val + length_val, vid=None)

    P.emit(
        SliceNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=P.to_int_or_vid(dim),
            start=start_iov,
            stop=stop_iov,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.unsqueeze.default, torch.ops.aten.unsqueeze_copy.default]
)
def _unsqueeze_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.unsqueeze")
    require_kwargs(P.kwargs(n), set(), "aten.unsqueeze")
    x, dim = args
    out = P.make_or_get_slot(n)
    P.emit(
        ExpandDimsNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.squeeze.dims, torch.ops.aten.squeeze_copy.dims]
)
def _squeeze_dims_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle squeeze operation for specific dimensions.

    Removes dimensions of size 1 from the tensor at specified positions.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.squeeze.dims")
    require_kwargs(P.kwargs(n), set(), "aten.squeeze.dims")
    x, dims = args
    out = P.make_or_get_slot(n)

    dims_list = list(dims) if dims is not None else None

    P.emit(
        SqueezeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=dims_list,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.squeeze.default, torch.ops.aten.squeeze_copy.default]
)
def _squeeze_default_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle squeeze operation without specified dimensions.

    Removes all dimensions of size 1 from the tensor.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.squeeze.default")
    require_kwargs(P.kwargs(n), set(), "aten.squeeze.default")
    (x,) = args
    out = P.make_or_get_slot(n)

    P.emit(
        SqueezeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=None,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.cat.default])
def _cat_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle concatenation of a list of tensors.

    Concatenates tensors along a specified dimension.
    All tensors must have the same shape except in the concatenating dimension.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.cat")
    require_kwargs(P.kwargs(n), set(), "aten.cat")
    # aten.cat.default signature: cat(Tensor[] tensors, int dim=0) -> Tensor
    # args can be (tensors_list,) or (tensors_list, dim)
    tensors_list = args[0]
    dim = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)

    # Convert list of tensor slots to list of Tids
    tensor_tids = [P.slot_to_tid(t) for t in tensors_list]

    # dim is typically an int
    axis = dim if dim is not None else 0

    P.emit(
        ConcatenateNode(
            tensors=tensor_tids,
            out=P.slot_to_tid(out),
            axis=axis,
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_with_sizes_copy.default,
    ]
)
def _split_with_sizes_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle split_with_sizes operation.

    Splits a tensor into chunks with specified sizes along a dimension.
    Returns a tuple of output slots that getitem can extract from.

    PyTorch: split_with_sizes(x, [2, 3, 4], dim=1)
    MLX: split(x, indices=[2, 5], axis=1)  # indices are cumulative positions
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.split_with_sizes")
    require_kwargs(P.kwargs(n), set(), "aten.split_with_sizes")
    x = args[0]
    sizes = args[1]
    dim = args[2] if len(args) > 2 else 0  # dim has default value of 0

    # Convert sizes to IntOrVid (supports both static ints and dynamic Vids)
    sizes_int_or_vid = [P.to_int_or_vid(s) for s in sizes]

    axis = dim if dim is not None else 0

    # Create output slots for multi-output operation
    # make_or_get_slots automatically creates slots based on node.meta["val"]
    output_slots = P.make_or_get_slots(n)

    # Emit SplitNode with all output slots
    P.emit(
        SplitNode(
            x=P.slot_to_tid(x),
            outs=[P.slot_to_tid(s) for s in output_slots],
            sizes=sizes_int_or_vid,
            axis=axis,
        )
    )

    # Return tuple of slots - getitem will extract individual elements
    return output_slots


@REGISTRY.register(
    target=[torch.ops.aten.split.Tensor, torch.ops.aten.split_copy.Tensor]
)
def _split_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle split operation with uniform chunk size.

    Splits a tensor into chunks of a given size along a dimension.
    The last chunk may be smaller if the dimension does not divide evenly.

    PyTorch: split(x, split_size, dim=0)

    We pass [split_size] to the interpreter, which computes the actual
    chunk sizes based on the tensor dimension.
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.split")
    require_kwargs(P.kwargs(n), set(), "aten.split")
    x = args[0]
    split_size = args[1]
    dim = args[2] if len(args) > 2 else 0

    axis = dim if dim is not None else 0
    if axis < 0:
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise RuntimeError("split: missing tensor metadata for negative axis")
        axis += len(x_meta.shape)

    # Create output slots for multi-output operation
    output_slots = P.make_or_get_slots(n)

    # Emit SplitNode - interpreter computes actual chunk sizes from split_size
    P.emit(
        SplitNode(
            x=P.slot_to_tid(x),
            outs=[P.slot_to_tid(s) for s in output_slots],
            sizes=[P.to_int_or_vid(split_size)],
            axis=axis,
        )
    )

    return output_slots


@REGISTRY.register(target=[torch.ops.aten.repeat.default])
def _repeat_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.repeat")
    require_kwargs(P.kwargs(n), set(), "aten.repeat")
    x, reps = args

    # Convert reps to IntOrVid (supports both static ints and dynamic Vids)
    reps_int_or_vid = [P.to_int_or_vid(r) for r in reps]

    out = P.make_or_get_slot(n)
    P.emit(
        TileNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            reps=reps_int_or_vid,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.index.Tensor])
def _index_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.index.Tensor")
    require_kwargs(P.kwargs(n), set(), "aten.index.Tensor")
    x, idx_list = args
    if not isinstance(idx_list, list) or len(idx_list) != 1:
        raise ValueError(
            f"aten.index.Tensor only supported with single index tensor, "
            f"got {len(idx_list) if isinstance(idx_list, list) else type(idx_list)}"
        )

    # Check that indices have the same number of dimensions as the input
    x_meta = n.args[0].meta.get("val")
    idx_meta = n.args[1][0].meta.get("val")
    if x_meta is not None and idx_meta is not None:
        x_ndim = len(x_meta.shape)
        idx_ndim = len(idx_meta.shape)
        if x_ndim != idx_ndim:
            raise ValueError(
                f"aten.index.Tensor requires indices to have same ndim as input for MLX. "
                f"Got input ndim={x_ndim}, indices ndim={idx_ndim}. "
                f"Use aten.embedding.default for lookup tables instead."
            )

    out = P.make_or_get_slot(n)
    P.emit(
        TakeAlongAxisNode(
            x=P.slot_to_tid(x),
            indices=P.slot_to_tid(idx_list[0]),
            out=P.slot_to_tid(out),
            axis=0,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.select.int, torch.ops.aten.select_copy.int])
def _select_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle aten.select_copy.int - select a single index along a dimension.

    select_copy(input, dim, index) returns input[..., index, ...] where the
    indexing happens at dimension `dim`. The selected dimension is removed.

    Maps to MLX's take(array, int index, axis) which also removes the dimension.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten.select_copy.int")
    require_kwargs(P.kwargs(n), set(), "aten.select_copy.int")
    x, dim, index = args
    out = P.make_or_get_slot(n)
    P.emit(
        TakeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            index=P.to_int_or_vid(index),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sym_size.int])
def _sym_size_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.sym_size.int")
    require_kwargs(P.kwargs(n), set(), "aten.sym_size.int")
    a, dim = args
    out = P.make_or_get_slot(n)
    P.emit(
        SymSizeNode(
            a=P.slot_to_tid(a),
            dim=dim,
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.item.default])
def _item_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    if not isinstance(n.meta["val"], torch.SymInt):
        raise ValueError("item only supported if it returns a SymInt")
    args = P.args(n)
    require_args(args, 1, 1, "aten.item")
    require_kwargs(P.kwargs(n), set(), "aten.item")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        ItemIntNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.getitem])
def _getitem_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle getitem(tuple, idx) - extracts element from a tuple of slots.

    The source tuple comes from ops that return multiple values (like
    auto_functionalized_v2). Those handlers return tuples of slots,
    and we just ID_COPY the selected element to a new output slot.
    """
    args = P.args(n)
    require_args(args, 2, 2, "operator.getitem")
    require_kwargs(P.kwargs(n), set(), "operator.getitem")
    a, idx = args
    out = P.make_or_get_slot(n)
    P.emit(
        IdCopyNode(
            x=P.slot_to_tid(a[idx]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.layer_norm.default])
def _layer_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 5, "aten.layer_norm")
    require_kwargs(P.kwargs(n), set(), "aten.layer_norm")
    x, shape = args[0:2]
    if len(shape) > 1:
        raise ValueError(
            "LayerNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    out = P.make_or_get_slot(n)
    P.emit(
        LayerNormNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            weight=P.slot_to_tid(w) if w else None,
            bias=P.slot_to_tid(bias) if bias else None,
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.native_layer_norm.default])
def _native_layer_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle native_layer_norm which returns (output, mean, rstd).

    Only the normalized output (index 0) is computed via fast::layer_norm;
    mean and rstd (indices 1 and 2) are needed only for backward.
    """
    # Verify mean/rstd outputs are unused — we only compute the normalized output.
    for user in n.users:
        if user.target == operator.getitem and user.args[1] in (1, 2):
            if len(user.users) > 0:
                raise ValueError(
                    f"native_layer_norm output {user.args[1]} (mean/rstd) is used, "
                    "but only the normalized output (index 0) is supported"
                )

    args = P.args(n)
    require_args(args, 2, 5, "aten.native_layer_norm")
    require_kwargs(P.kwargs(n), set(), "aten.native_layer_norm")
    x, shape = args[0:2]
    if len(shape) > 1:
        raise ValueError(
            "LayerNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    # native_layer_norm returns (output, mean, rstd) — allocate all 3 slots
    output_slots = P.make_or_get_slots(n)

    P.emit(
        LayerNormNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(output_slots[0]),
            weight=P.slot_to_tid(w) if w else None,
            bias=P.slot_to_tid(bias) if bias else None,
            eps=eps,
        )
    )
    return output_slots


@REGISTRY.register(target=[torch.ops.aten.arange.default])
def _arange_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with just stop, or (start, stop) or (start, stop, step).

    Supports both static (literal int) and dynamic (Slot from item()) values.
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 3, "aten.arange")
    require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.arange")
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.arange",
    )
    if len(args) == 1:
        start = 0
        stop = args[0]
    else:
        start, stop = args[0:2]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = kwargs.get("dtype", torch.int64)
    scalar_type_val = torch_dtype_to_scalar_type(dtype)

    out = P.make_or_get_slot(n)
    P.emit(
        ARangeNode(
            out=P.slot_to_tid(out),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
            step=P.to_int_or_vid(step),
            scalar_type=scalar_type_val,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.arange.start_step])
def _arange_start_step_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with start, end, and step arguments.

    Supports both static (literal int) and dynamic (Slot from item()) start/stop/step.
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 2, 3, "aten.arange.start_step")
    require_kwargs(
        kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.arange.start_step"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.arange.start_step",
    )
    start = args[0]
    stop = args[1]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = kwargs.get("dtype", torch.int64)
    scalar_type_val = torch_dtype_to_scalar_type(dtype)

    out = P.make_or_get_slot(n)
    P.emit(
        ARangeNode(
            out=P.slot_to_tid(out),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
            step=P.to_int_or_vid(step),
            scalar_type=scalar_type_val,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.rms_norm.default])
def _aten_rms_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 4, "aten.rms_norm")
    require_kwargs(P.kwargs(n), set(), "aten.rms_norm")
    x, normalized_shape = args[0], args[1]
    if len(normalized_shape) > 1:
        raise ValueError(
            "RMSNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    eps = args[3] if len(args) > 3 else 1e-5
    out = P.make_or_get_slot(n)
    P.emit(
        RMSNormNode(
            x=P.slot_to_tid(x),
            weight=P.slot_to_tid(w) if w else None,
            out=P.slot_to_tid(out),
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.mlx.rope.default])
def _rope_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 3, 7, "mlx.rope")
    require_kwargs(P.kwargs(n), set(), "mlx.rope")
    x, dims, pos = args[0], args[1], args[2]
    traditional = args[3] if len(args) > 3 else False
    base = args[4] if len(args) > 4 else 500000.0
    scale = args[5] if len(args) > 5 else 1.0
    freqs = args[6] if len(args) > 6 else None
    out = P.make_or_get_slot(n)

    # pos must be a Slot (SymInt) from input_pos.item() during tracing
    # The schema supports both Vid (scalar) and Tid (tensor) for offset
    if not isinstance(pos, Slot):
        raise ValueError(
            f"RopeNode.offset must be a SymInt (traced via tensor.item()), got {type(pos)}. "
            "Make sure input_pos is a tensor and you call input_pos.item() to get a SymInt."
        )

    P.emit(
        RopeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=dims,
            offset=TidOrVid.from_vid(P.slot_to_vid(pos)),
            freqs=P.slot_to_tid(freqs) if freqs else None,
            traditional=traditional,
            base=base,
            scale=scale,
        )
    )

    return out


def _emit_channel_last_weight(P: MLXProgramBuilder, w_node: Node, perm: list) -> Slot:
    """Get convolution weight in channel-last format.

    If the weight is a placeholder (static parameter), permute at compile time
    and store as a constant.  If it comes from another node (e.g. dequantize
    output), emit a runtime TransposeNode instead.
    """
    if w_node.op == "placeholder":
        w_target, w_tensor = P.get_placeholder_target_and_tensor(w_node)
        return P.make_or_get_constant(
            f"{w_target}_channel_last", w_tensor.permute(perm).contiguous()
        )
    else:
        w_slot = P.slot_map([w_node])[0]
        _, w = P.make_tmp_slot()
        P.emit(
            TransposeNode(
                x=P.slot_to_tid(w_slot),
                out=P.slot_to_tid(w),
                perm=perm,
            )
        )
        return w


def _emit_conv_bias(
    P: MLXProgramBuilder, bias: Optional[Slot], tmp: Slot, ndim: int
) -> None:
    """Reshape conv bias to channel-last broadcast shape and add to tmp in-place.

    After the convolution the activation is in channel-last layout, so the bias
    (shape ``[C_out]``) must be reshaped to ``[1, …, 1, -1]`` with *ndim*
    dimensions before being added.  Does nothing when *bias* is ``None``.
    """
    if bias is None:
        return
    _, tmp2 = P.make_tmp_slot()
    shape = [IntOrVid.from_literal(1)] * (ndim - 1) + [IntOrVid.from_literal(-1)]
    P.emit(
        ReshapeNode(
            x=P.slot_to_tid(bias),
            out=P.slot_to_tid(tmp2),
            shape=shape,
        )
    )
    P.emit(
        AddNode(
            a=P.slot_to_tid(tmp),
            b=P.slot_to_tid(tmp2),
            out=P.slot_to_tid(tmp),
        )
    )


@REGISTRY.register(target=[torch.ops.aten.conv1d.default])
def _conv1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    require_args(n.args, 2, 7, "aten.convolution (Conv1D)")
    require_kwargs(P.kwargs(n), set(), "aten.convolution (Conv1D)")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    stride = n.args[3] if len(n.args) > 3 else 1
    if isinstance(stride, list):
        assert (
            len(stride) == 1
        ), f"Conv1D stride must be a single value, got {len(stride)} values"
        stride = stride[0]
    padding = n.args[4] if len(n.args) > 4 else 0
    if isinstance(padding, list):
        assert (
            len(padding) == 1
        ), f"Conv1D padding must be a single value, got {len(padding)} values"
        padding = padding[0]
    dilation = n.args[5] if len(n.args) > 5 else 1
    if isinstance(dilation, list):
        assert (
            len(dilation) == 1
        ), f"Conv1D dilation must be a single value, got {len(dilation)} values"
        dilation = dilation[0]
    groups = n.args[6] if len(n.args) > 6 else 1

    # Validate all parameters are static integers
    require_static_int(stride, "stride", "aten.conv1d (Conv1DNode)")
    require_static_int(padding, "padding", "aten.conv1d (Conv1DNode)")
    require_static_int(dilation, "dilation", "aten.conv1d (Conv1DNode)")
    require_static_int(groups, "groups", "aten.conv1d (Conv1DNode)")

    # Weight: [O, I/G, K] -> [O, K, I]
    w = _emit_channel_last_weight(P, w_node, [0, 2, 1])

    x, bias = P.slot_map([x_node, bias_node])

    # Transpose input: (N, C_in, W) -> (N, W, C_in)
    tmp_name, tmp = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(tmp),
            perm=[0, 2, 1],
        )
    )

    # Conv1D
    P.emit(
        Conv1DNode(
            x=P.slot_to_tid(tmp),
            w=P.slot_to_tid(w),
            out=P.slot_to_tid(tmp),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    )

    _emit_conv_bias(P, bias, tmp, ndim=3)

    # Transpose output: (N, W, C_out) -> (N, C_out, W)
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(tmp),
            out=P.slot_to_tid(out),
            perm=[0, 2, 1],
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.conv2d.default])
def _conv2d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv2d or aten.convolution with 2D inputs.

    PyTorch format: (N, C_in, H, W) with weights (C_out, C_in/G, KH, KW)
    MLX format: (N, H, W, C_in) with weights (C_out, KH, KW, C_in/G)
    """
    require_args(n.args, 2, 7, "aten.convolution (Conv2D)")
    require_kwargs(P.kwargs(n), set(), "aten.convolution (Conv2D)")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    stride = n.args[3] if len(n.args) > 3 else [1, 1]
    padding = n.args[4] if len(n.args) > 4 else [0, 0]
    dilation = n.args[5] if len(n.args) > 5 else [1, 1]
    groups = n.args[6] if len(n.args) > 6 else 1

    # Extract stride values
    if isinstance(stride, list):
        stride_h, stride_w = (
            (stride[0], stride[1]) if len(stride) == 2 else (stride[0], stride[0])
        )
    else:
        stride_h = stride_w = stride

    # Extract padding values
    if isinstance(padding, list):
        padding_h, padding_w = (
            (padding[0], padding[1]) if len(padding) == 2 else (padding[0], padding[0])
        )
    else:
        padding_h = padding_w = padding

    # Extract dilation values
    if isinstance(dilation, list):
        dilation_h, dilation_w = (
            (dilation[0], dilation[1])
            if len(dilation) == 2
            else (dilation[0], dilation[0])
        )
    else:
        dilation_h = dilation_w = dilation

    # Validate all parameters are static integers
    require_static_int(stride_h, "stride_h", "aten.convolution (Conv2D)")
    require_static_int(stride_w, "stride_w", "aten.convolution (Conv2D)")
    require_static_int(padding_h, "padding_h", "aten.convolution (Conv2D)")
    require_static_int(padding_w, "padding_w", "aten.convolution (Conv2D)")
    require_static_int(dilation_h, "dilation_h", "aten.convolution (Conv2D)")
    require_static_int(dilation_w, "dilation_w", "aten.convolution (Conv2D)")
    require_static_int(groups, "groups", "aten.convolution (Conv2D)")

    # Weight: (C_out, C_in/G, KH, KW) -> (C_out, KH, KW, C_in/G)
    w = _emit_channel_last_weight(P, w_node, [0, 2, 3, 1])

    x, bias = P.slot_map([x_node, bias_node])

    # Transpose input: (N, C_in, H, W) -> (N, H, W, C_in)
    tmp_name, tmp = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(tmp),
            perm=[0, 2, 3, 1],
        )
    )

    # Conv2D
    P.emit(
        Conv2DNode(
            x=P.slot_to_tid(tmp),
            w=P.slot_to_tid(w),
            out=P.slot_to_tid(tmp),
            stride_h=stride_h,
            stride_w=stride_w,
            padding_h=padding_h,
            padding_w=padding_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            groups=groups,
        )
    )

    _emit_conv_bias(P, bias, tmp, ndim=4)

    # Transpose output: (N, H, W, C_out) -> (N, C_out, H, W)
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(tmp),
            out=P.slot_to_tid(out),
            perm=[0, 3, 1, 2],
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.conv3d.default])
def _conv3d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv3d with 3D inputs.

    PyTorch format: (N, C_in, D, H, W) with weights (C_out, C_in/G, KD, KH, KW)
    MLX format: (N, D, H, W, C_in) with weights (C_out, KD, KH, KW, C_in/G)
    """
    require_args(n.args, 2, 7, "aten.conv3d")
    require_kwargs(P.kwargs(n), set(), "aten.conv3d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    stride = n.args[3] if len(n.args) > 3 else [1, 1, 1]
    padding = n.args[4] if len(n.args) > 4 else [0, 0, 0]
    dilation = n.args[5] if len(n.args) > 5 else [1, 1, 1]
    groups = n.args[6] if len(n.args) > 6 else 1

    # MLX only supports groups=1 for 3D convolutions
    if groups != 1:
        raise ValueError(
            "aten.conv3d with groups != 1 is not supported by MLX. "
            f"Got groups={groups}."
        )

    # Extract stride values
    if isinstance(stride, list):
        if len(stride) == 3:
            stride_d, stride_h, stride_w = stride
        else:
            stride_d = stride_h = stride_w = stride[0]
    else:
        stride_d = stride_h = stride_w = stride

    # Extract padding values
    if isinstance(padding, list):
        if len(padding) == 3:
            padding_d, padding_h, padding_w = padding
        else:
            padding_d = padding_h = padding_w = padding[0]
    else:
        padding_d = padding_h = padding_w = padding

    # Extract dilation values
    if isinstance(dilation, list):
        if len(dilation) == 3:
            dilation_d, dilation_h, dilation_w = dilation
        else:
            dilation_d = dilation_h = dilation_w = dilation[0]
    else:
        dilation_d = dilation_h = dilation_w = dilation

    # Validate all parameters are static integers
    require_static_int(stride_d, "stride_d", "aten.conv3d")
    require_static_int(stride_h, "stride_h", "aten.conv3d")
    require_static_int(stride_w, "stride_w", "aten.conv3d")
    require_static_int(padding_d, "padding_d", "aten.conv3d")
    require_static_int(padding_h, "padding_h", "aten.conv3d")
    require_static_int(padding_w, "padding_w", "aten.conv3d")
    require_static_int(dilation_d, "dilation_d", "aten.conv3d")
    require_static_int(dilation_h, "dilation_h", "aten.conv3d")
    require_static_int(dilation_w, "dilation_w", "aten.conv3d")
    require_static_int(groups, "groups", "aten.conv3d")

    # Weight: (C_out, C_in/G, KD, KH, KW) -> (C_out, KD, KH, KW, C_in/G)
    w = _emit_channel_last_weight(P, w_node, [0, 2, 3, 4, 1])

    x, bias = P.slot_map([x_node, bias_node])

    # Transpose input: (N, C_in, D, H, W) -> (N, D, H, W, C_in)
    tmp_name, tmp = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(tmp),
            perm=[0, 2, 3, 4, 1],
        )
    )

    # Conv3D
    P.emit(
        Conv3DNode(
            x=P.slot_to_tid(tmp),
            w=P.slot_to_tid(w),
            out=P.slot_to_tid(tmp),
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            padding_d=padding_d,
            padding_h=padding_h,
            padding_w=padding_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            groups=groups,
        )
    )

    _emit_conv_bias(P, bias, tmp, ndim=5)

    # Transpose output: (N, D, H, W, C_out) -> (N, C_out, D, H, W)
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(tmp),
            out=P.slot_to_tid(out),
            perm=[0, 4, 1, 2, 3],
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sub.Tensor])
def _sub_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.sub.Tensor")
    require_kwargs(P.kwargs(n), set(), "aten.sub.Tensor")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.rsqrt.default])
def _rsqrt_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 1, 1, "aten.rsqrt")
    require_kwargs(P.kwargs(n), set(), "aten.rsqrt")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        RsqrtNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.maximum.default])
def _maximum_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.maximum.default - element-wise maximum of two tensors."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.maximum")
    require_kwargs(P.kwargs(n), set(), "aten.maximum")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        MaximumNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.minimum.default])
def _minimum_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.minimum.default - element-wise minimum of two tensors."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.minimum")
    require_kwargs(P.kwargs(n), set(), "aten.minimum")
    a, b = args
    out = P.make_or_get_slot(n)
    P.emit(
        MinimumNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.relu.default])
def _relu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.relu.default - rectified linear unit.

    ReLU(x) = max(x, 0), implemented using MaximumNode with a scalar zero.
    Uses broadcasting in maximum operation for efficiency.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.relu")
    require_kwargs(P.kwargs(n), set(), "aten.relu")
    (x,) = args  # x is already a Slot

    # Get input dtype
    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for relu")
    dtype = x_meta.dtype

    # Create a temporary slot for scalar zero using slot_manager
    _, zero_slot = P.make_tmp_slot()

    # Emit FullNode to create a scalar zero (shape=[])
    # Maximum will broadcast this scalar to match input shape
    P.emit(
        FullNode(
            shape=[],  # Scalar (will be broadcast in maximum)
            v=FloatOrVid.from_literal(0.0),
            scalar_type=torch_dtype_to_scalar_type(dtype),
            out=P.slot_to_tid(zero_slot),
        )
    )

    # Emit MaximumNode(x, scalar_zero)
    out = P.make_or_get_slot(n)
    P.emit(
        MaximumNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(zero_slot),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten._log_softmax.default])
def _log_softmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten._log_softmax.default - log of softmax.

    LogSoftmax(x, dim) = x - logsumexp(x, dim, keepdims=True)

    This is numerically stable because it avoids computing softmax
    (which can underflow to 0) followed by log (which gives -inf for 0).
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten._log_softmax")
    require_kwargs(P.kwargs(n), set(), "aten._log_softmax")
    x, dim, _half_to_float = args  # x is already a Slot

    # Create temporary slot for logsumexp output
    _, logsumexp_slot = P.make_tmp_slot()

    # Emit LogSumExpNode with keepdims=True
    P.emit(
        LogSumExpNode(
            x=P.slot_to_tid(x),
            axes=[dim],
            keepdims=True,
            out=P.slot_to_tid(logsumexp_slot),
        )
    )

    # Emit SubtractNode: x - logsumexp(x)
    out = P.make_or_get_slot(n)
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(logsumexp_slot),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.constant_pad_nd.default])
def _constant_pad_nd_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.constant_pad_nd - pad with a constant value.

    PyTorch pad format: [left_0, right_0, left_1, right_1, ...]
    MLX pad_width format: [(before_0, after_0), (before_1, after_1), ...]

    Note: PyTorch pads in reverse order (last dimensions first).
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.constant_pad_nd")
    require_kwargs(P.kwargs(n), set(), "aten.constant_pad_nd")
    x_node, pad = args[0], args[1]
    value = args[2] if len(args) > 2 else 0

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"aten.constant_pad_nd: constant value must be a scalar, got {type(value)}"
        )

    # Convert PyTorch pad format to MLX pad_width format
    # PyTorch: [left_D, right_D, left_D-1, right_D-1, ...]
    # MLX: [(before_0, after_0), (before_1, after_1), ..., (before_D, after_D)]
    if len(pad) % 2 != 0:
        raise ValueError(
            f"aten.constant_pad_nd: pad length must be even, got {len(pad)}"
        )

    x = P.slot_map([x_node])[0]
    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for constant_pad_nd")

    ndim = len(x_meta.shape)
    num_pad_dims = len(pad) // 2

    if num_pad_dims > ndim:
        raise ValueError(
            f"aten.constant_pad_nd: trying to pad {num_pad_dims} dimensions "
            f"but input has only {ndim} dimensions"
        )

    # Build MLX pad_width: start with zeros for non-padded dims
    pad_width = []
    for _ in range(ndim - num_pad_dims):
        pad_width.extend([0, 0])  # No padding for these dimensions

    # Add padding for the padded dimensions (reverse order)
    for i in range(num_pad_dims - 1, -1, -1):
        left = pad[i * 2]
        right = pad[i * 2 + 1]
        pad_width.extend([left, right])

    out = P.make_or_get_slot(n)
    P.emit(
        PadNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            pad_width=[P.to_int_or_vid(v) for v in pad_width],
            mode="constant",
            constant_value=float(value),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.clamp.default])
def _clamp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.clamp - clamp values to [min, max] range.

    clamp(input, min=None, max=None) -> Tensor

    Clamps all elements in input into the range [min, max].
    If min is None, there is no lower bound. If max is None, there is no upper bound.
    """
    args = P.args(n)
    require_args(args, 1, 3, "aten.clamp")
    require_kwargs(P.kwargs(n), set(), "aten.clamp")

    x = args[0]
    min_val = args[1] if len(args) > 1 else None
    max_val = args[2] if len(args) > 2 else None

    # Get input dtype for creating scalar constants
    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for clamp")
    dtype = x_meta.dtype

    out = P.make_or_get_slot(n)

    # If neither min nor max, just copy (shouldn't happen per PyTorch, but handle it)
    if min_val is None and max_val is None:
        P.emit(
            IdCopyNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
            )
        )
        return out

    # Helper to create a scalar constant slot
    def make_scalar_slot(val):
        _, slot = P.make_tmp_slot()
        P.emit(
            FullNode(
                shape=[],  # Scalar
                v=FloatOrVid.from_literal(float(val)),
                scalar_type=torch_dtype_to_scalar_type(dtype),
                out=P.slot_to_tid(slot),
            )
        )
        return slot

    current = x

    # Apply max constraint first: min(x, max_val)
    if max_val is not None:
        max_slot = make_scalar_slot(max_val)
        if min_val is not None:
            # Need a temp slot since we have both constraints
            _, tmp = P.make_tmp_slot()
            P.emit(
                MinimumNode(
                    a=P.slot_to_tid(current),
                    b=P.slot_to_tid(max_slot),
                    out=P.slot_to_tid(tmp),
                )
            )
            current = tmp
        else:
            # Only max constraint, output directly
            P.emit(
                MinimumNode(
                    a=P.slot_to_tid(current),
                    b=P.slot_to_tid(max_slot),
                    out=P.slot_to_tid(out),
                )
            )
            return out

    # Apply min constraint: max(current, min_val)
    if min_val is not None:
        min_slot = make_scalar_slot(min_val)
        P.emit(
            MaximumNode(
                a=P.slot_to_tid(current),
                b=P.slot_to_tid(min_slot),
                out=P.slot_to_tid(out),
            )
        )

    return out


@REGISTRY.register(
    target=[torch.ops.aten.expand.default, torch.ops.aten.expand_copy.default]
)
def _expand_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle expand: broadcasts dimensions of size 1 to larger sizes."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.expand")
    require_kwargs(P.kwargs(n), set(), "aten.expand")
    x, size = args
    out = P.make_or_get_slot(n)

    shape_iovs = [P.to_int_or_vid(s) for s in size]
    P.emit(
        BroadcastToNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            shape=shape_iovs,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten._native_batch_norm_legit_no_training.default])
def _native_batch_norm_legit_no_training_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle batch norm inference (no training).

    Formula: output = (input - mean) / sqrt(var + eps) * weight + bias

    Args:
        input: [N, C, ...] tensor
        weight: [C] gamma parameter
        bias: [C] beta parameter
        running_mean: [C]
        running_var: [C]
        momentum: float (unused in inference)
        eps: float

    Returns:
        Tuple of (output, empty, empty) - save_mean and save_invstd are empty for no_training
    """
    args = P.args(n)
    require_args(args, 7, 7, "aten._native_batch_norm_legit_no_training")
    require_kwargs(P.kwargs(n), set(), "aten._native_batch_norm_legit_no_training")
    x = args[0]
    weight = args[1]  # gamma [C] - optional (None if affine=False)
    bias = args[2]  # beta [C] - optional (None if affine=False)
    mean = args[3]  # running_mean [C]
    var = args[4]  # running_var [C]
    # momentum = args[5] - not used in inference
    eps = args[6]  # epsilon

    # Get output slots (3 outputs: normalized, save_mean, save_invstd)
    output_slots = P.make_or_get_slots(n)
    out = output_slots[0]  # Main output

    # Get input ndim to determine reshape dimensions
    # For BatchNorm1d: input is [N, C, L] -> reshape params to [1, C, 1]
    # For BatchNorm2d: input is [N, C, H, W] -> reshape params to [1, C, 1, 1]
    input_node = n.args[0]
    input_ndim = len(input_node.meta["val"].shape)

    # Validate input dimensionality (only 3D and 4D supported)
    if input_ndim not in (3, 4):
        raise NotImplementedError(
            f"MLX batch norm handler only supports 3D (BatchNorm1d) and 4D (BatchNorm2d) inputs. "
            f"Got {input_ndim}D input."
        )

    def reshape_for_broadcast(slot, name_suffix):
        """Reshape a [C] tensor for broadcasting with input."""
        _, reshaped = P.make_tmp_slot()
        # Build shape: [1, -1] + [1] * (ndim - 2)
        shape = [P.to_int_or_vid(1), P.to_int_or_vid(-1)]
        for _ in range(input_ndim - 2):
            shape.append(P.to_int_or_vid(1))
        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(slot),
                shape=shape,
                out=P.slot_to_tid(reshaped),
            )
        )
        return reshaped

    mean_reshaped = reshape_for_broadcast(mean, "mean")
    var_reshaped = reshape_for_broadcast(var, "var")

    # Step 1: x_centered = x - mean
    _, tmp_centered = P.make_tmp_slot()
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(mean_reshaped),
            out=P.slot_to_tid(tmp_centered),
        )
    )

    # Step 2: var_eps = var + eps
    # Create eps as a scalar using FullNode (broadcasts correctly with var)
    _, eps_slot = P.make_tmp_slot()
    P.emit(
        FullNode(
            out=P.slot_to_tid(eps_slot),
            shape=[],  # 0-D scalar
            v=FloatOrVid.from_literal(float(eps)),
            scalar_type=torch_dtype_to_scalar_type(torch.float32),
        )
    )
    _, tmp_var_eps = P.make_tmp_slot()
    P.emit(
        AddNode(
            a=P.slot_to_tid(var_reshaped),
            b=P.slot_to_tid(eps_slot),
            out=P.slot_to_tid(tmp_var_eps),
        )
    )

    # Step 3: inv_std = rsqrt(var_eps)
    _, tmp_inv_std = P.make_tmp_slot()
    P.emit(RsqrtNode(x=P.slot_to_tid(tmp_var_eps), out=P.slot_to_tid(tmp_inv_std)))

    # Step 4: x_normalized = x_centered * inv_std
    _, tmp_normalized = P.make_tmp_slot()
    P.emit(
        MultiplyNode(
            a=P.slot_to_tid(tmp_centered),
            b=P.slot_to_tid(tmp_inv_std),
            out=P.slot_to_tid(tmp_normalized),
        )
    )

    # Step 5: x_scaled = x_normalized * weight (skip if weight is None, i.e. affine=False)
    if weight is not None:
        weight_reshaped = reshape_for_broadcast(weight, "weight")
        _, tmp_scaled = P.make_tmp_slot()
        P.emit(
            MultiplyNode(
                a=P.slot_to_tid(tmp_normalized),
                b=P.slot_to_tid(weight_reshaped),
                out=P.slot_to_tid(tmp_scaled),
            )
        )
        current_result = tmp_scaled
    else:
        current_result = tmp_normalized

    # Step 6: out = current_result + bias (skip if bias is None, i.e. affine=False)
    if bias is not None:
        bias_reshaped = reshape_for_broadcast(bias, "bias")
        P.emit(
            AddNode(
                a=P.slot_to_tid(current_result),
                b=P.slot_to_tid(bias_reshaped),
                out=P.slot_to_tid(out),
            )
        )
    else:
        # No bias - just copy the result to output
        P.emit(
            IdCopyNode(
                x=P.slot_to_tid(current_result),
                out=P.slot_to_tid(out),
            )
        )

    # For no_training mode, outputs 1 and 2 (save_mean, save_invstd) are empty
    # They should already be allocated by make_or_get_slots but we don't write to them
    # PyTorch returns empty tensors for these in no_training mode

    return output_slots


@REGISTRY.register(target=[torch.ops.aten.where.self])
def _where_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle where: select from x or y according to condition.

    where(condition, x, y) returns elements from x where condition is True,
    and elements from y where condition is False.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten.where")
    require_kwargs(P.kwargs(n), set(), "aten.where")
    condition, x, y = args
    out = P.make_or_get_slot(n)

    P.emit(
        WhereNode(
            condition=P.slot_to_tid(condition),
            x=P.slot_to_tid(x),
            y=P.slot_to_tid(y),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.full.default])
def _full_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.full - create tensor filled with a value."""
    # aten.full(size, fill_value, *, dtype=None, ...)
    # Use P.args to properly convert Nodes to Slots for dynamic shapes
    args = P.args(n)
    require_args(args, 2, 2, "aten.full")
    kwargs = P.kwargs(n)
    require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.full")
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.full",
    )
    out = P.make_or_get_slot(n)
    shape = args[0]
    shape = args[0]  # List of int or Slot for dynamic dims
    fill_value = args[1]  # Scalar
    dtype = n.kwargs.get("dtype")

    # Convert shape to IntOrVid (supports both static ints and dynamic Slots)
    shape_iovs = [P.to_int_or_vid(d) for d in shape]

    if dtype is None:
        dtype = torch.float32  # default

    P.emit(
        FullNode(
            out=P.slot_to_tid(out),
            shape=shape_iovs,
            v=P.to_float_or_vid(fill_value),
            scalar_type=torch_dtype_to_scalar_type(dtype),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.zeros.default])
def _zeros_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.zeros - create tensor filled with zeros."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.zeros")
    kwargs = P.kwargs(n)
    require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.zeros")
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.zeros",
    )
    out = P.make_or_get_slot(n)

    # aten.zeros(size, *, dtype=None, ...)
    shape = n.args[0]
    shape = n.args[0]  # List[int] or may contain Slots for dynamic dims
    dtype = n.kwargs.get("dtype")

    # Convert shape to IntOrVid (supports both static ints and dynamic Slots)
    shape_iovs = [P.to_int_or_vid(d) for d in shape]

    if dtype is None:
        dtype = torch.float32  # default

    P.emit(
        FullNode(
            out=P.slot_to_tid(out),
            shape=shape_iovs,
            v=FloatOrVid.from_literal(0.0),
            scalar_type=torch_dtype_to_scalar_type(dtype),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.ones.default])
def _ones_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.ones - create tensor filled with ones."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.ones")
    kwargs = P.kwargs(n)
    require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.ones")
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.ones",
    )
    out = P.make_or_get_slot(n)

    # aten.ones(size, *, dtype=None, ...)
    shape = n.args[0]
    shape = n.args[0]  # List[int] or may contain Slots for dynamic dims
    dtype = n.kwargs.get("dtype")

    # Convert shape to IntOrVid (supports both static ints and dynamic Slots)
    shape_iovs = [P.to_int_or_vid(d) for d in shape]

    if dtype is None:
        dtype = torch.float32  # default

    P.emit(
        FullNode(
            out=P.slot_to_tid(out),
            shape=shape_iovs,
            v=FloatOrVid.from_literal(1.0),
            scalar_type=torch_dtype_to_scalar_type(dtype),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.zeros_like.default])
def _zeros_like_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.zeros_like - create zero-filled tensor with same shape as input."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.zeros_like")
    require_kwargs(
        kwargs,
        {"dtype", "layout", "device", "pin_memory", "memory_format"},
        "aten.zeros_like",
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        memory_format=kwargs.get("memory_format"),
        op_name="aten.zeros_like",
    )
    x = args[0]
    out = P.make_or_get_slot(n)

    # aten.zeros_like(input, *, dtype=None, ...)
    # If dtype is None, don't pass it - the C++ will use input's dtype
    dtype = n.kwargs.get("dtype")

    P.emit(
        FullLikeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            v=FloatOrVid.from_literal(0.0),
            scalar_type=(
                torch_dtype_to_scalar_type(dtype) if dtype is not None else None
            ),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.ones_like.default])
def _ones_like_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.ones_like - create one-filled tensor with same shape as input."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.ones_like")
    require_kwargs(
        kwargs,
        {"dtype", "layout", "device", "pin_memory", "memory_format"},
        "aten.ones_like",
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        memory_format=kwargs.get("memory_format"),
        op_name="aten.ones_like",
    )
    x = args[0]
    out = P.make_or_get_slot(n)

    # aten.ones_like(input, *, dtype=None, ...)
    # If dtype is None, don't pass it - the C++ will use input's dtype
    dtype = n.kwargs.get("dtype")

    P.emit(
        FullLikeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            v=FloatOrVid.from_literal(1.0),
            scalar_type=(
                torch_dtype_to_scalar_type(dtype) if dtype is not None else None
            ),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.full_like.default])
def _full_like_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.full_like - create tensor filled with value with same shape."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 2, 2, "aten.full_like")
    require_kwargs(
        kwargs,
        {"dtype", "layout", "device", "pin_memory", "memory_format"},
        "aten.full_like",
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        memory_format=kwargs.get("memory_format"),
        op_name="aten.full_like",
    )
    x = args[0]
    fill_value = args[1]
    out = P.make_or_get_slot(n)

    # aten.full_like(input, fill_value, *, dtype=None, ...)
    # If dtype is None, don't pass it - the C++ will use input's dtype
    dtype = n.kwargs.get("dtype")

    P.emit(
        FullLikeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            v=P.to_float_or_vid(fill_value),
            scalar_type=(
                torch_dtype_to_scalar_type(dtype) if dtype is not None else None
            ),
        )
    )
    return out


# =============================================================================
# Comparison Ops
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar])
def _less_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.lt - less than comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.lt")
    require_kwargs(P.kwargs(n), set(), "aten.lt")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        LessNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar])
def _less_equal_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.le - less than or equal comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.le")
    require_kwargs(P.kwargs(n), set(), "aten.le")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        LessEqualNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar])
def _greater_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.gt - greater than comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.gt")
    require_kwargs(P.kwargs(n), set(), "aten.gt")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        GreaterNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar])
def _greater_equal_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.ge - greater than or equal comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.ge")
    require_kwargs(P.kwargs(n), set(), "aten.ge")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        GreaterEqualNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.eq.Tensor, torch.ops.aten.eq.Scalar])
def _equal_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.eq - equality comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.eq")
    require_kwargs(P.kwargs(n), set(), "aten.eq")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        EqualNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.ne.Tensor, torch.ops.aten.ne.Scalar])
def _not_equal_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.ne - not equal comparison."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.ne")
    require_kwargs(P.kwargs(n), set(), "aten.ne")
    a, b = args[0], args[1]
    if not isinstance(b, Slot):
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32
        b = P.make_or_get_constant(f"_scalar_{b}", torch.tensor([b], dtype=dtype))
    out = P.make_or_get_slot(n)
    P.emit(
        NotEqualNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


# =============================================================================
# Logical Ops
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.logical_not.default])
def _logical_not_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logical_not - element-wise logical NOT."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.logical_not")
    require_kwargs(P.kwargs(n), set(), "aten.logical_not")
    out = P.make_or_get_slot(n)
    P.emit(
        LogicalNotNode(
            a=P.slot_to_tid(args[0]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.bitwise_not.default])
def _bitwise_not_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.bitwise_not - for boolean tensors, dispatch to logical_not."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.bitwise_not")
    require_kwargs(P.kwargs(n), set(), "aten.bitwise_not")
    x_meta = n.args[0].meta.get("val")

    if x_meta is not None and x_meta.dtype == torch.bool:
        # For boolean tensors, bitwise_not is equivalent to logical_not
        out = P.make_or_get_slot(n)
        P.emit(
            LogicalNotNode(
                a=P.slot_to_tid(args[0]),
                out=P.slot_to_tid(out),
            )
        )
        return out
    else:
        raise NotImplementedError(
            f"aten.bitwise_not is only supported for boolean tensors. "
            f"Got dtype={x_meta.dtype if x_meta else 'unknown'}"
        )


@REGISTRY.register(target=[torch.ops.aten.logical_and.default])
def _logical_and_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logical_and - element-wise logical AND."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.logical_and")
    require_kwargs(P.kwargs(n), set(), "aten.logical_and")
    out = P.make_or_get_slot(n)
    P.emit(
        LogicalAndNode(
            a=P.slot_to_tid(args[0]),
            b=P.slot_to_tid(args[1]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.logical_or.default])
def _logical_or_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logical_or - element-wise logical OR."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.logical_or")
    require_kwargs(P.kwargs(n), set(), "aten.logical_or")
    out = P.make_or_get_slot(n)
    P.emit(
        LogicalOrNode(
            a=P.slot_to_tid(args[0]),
            b=P.slot_to_tid(args[1]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.scalar_tensor.default])
def _scalar_tensor_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.scalar_tensor - create a 0-D tensor from a scalar value.

    scalar_tensor(scalar, *, dtype=None, layout=None, device=None, pin_memory=None) -> Tensor

    This is equivalent to torch.full([], scalar, dtype=dtype).
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.scalar_tensor")
    require_kwargs(
        kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.scalar_tensor"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.scalar_tensor",
    )
    scalar_value = args[0]

    out = P.make_or_get_slot(n)

    # Get dtype from kwargs, default to float32
    dtype = n.kwargs.get("dtype")
    if dtype is None:
        # Infer dtype from scalar type
        if isinstance(scalar_value, bool):
            dtype = torch.bool
        elif isinstance(scalar_value, int):
            dtype = torch.int64
        else:
            dtype = torch.float32

    P.emit(
        FullNode(
            out=P.slot_to_tid(out),
            shape=[],  # 0-D tensor (scalar)
            v=P.to_float_or_vid(scalar_value),
            scalar_type=torch_dtype_to_scalar_type(dtype),
        )
    )
    return out


# =============================================================================
# Triangular Matrix Ops
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.tril.default])
def _tril_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.tril - extract lower triangular part of matrix.

    tril(input, diagonal=0) -> Tensor

    Returns the lower triangular part of the matrix, with all elements above
    the diagonal set to zero. The diagonal parameter controls which diagonal
    to consider: 0 = main diagonal, positive = above main, negative = below main.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.tril")
    require_kwargs(P.kwargs(n), set(), "aten.tril")
    x = args[0]
    diagonal = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)
    P.emit(
        TrilNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            k=diagonal,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.triu.default])
def _triu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.triu - extract upper triangular part of matrix.

    triu(input, diagonal=0) -> Tensor

    Returns the upper triangular part of the matrix, with all elements below
    the diagonal set to zero. The diagonal parameter controls which diagonal
    to consider: 0 = main diagonal, positive = above main, negative = below main.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.triu")
    require_kwargs(P.kwargs(n), set(), "aten.triu")
    x = args[0]
    diagonal = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)
    P.emit(
        TriuNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            k=diagonal,
        )
    )
    return out


# Note: TriNode is available in the schema for creating triangular matrices directly
# (without needing an input tensor). There's no direct PyTorch aten.tri op - the typical
# pattern is torch.ones(n, m).tril(k). A fusion pass could optimize this to use TriNode.
# For now, TriNode can be used directly via the serialization API if needed.


# =============================================================================
# Math Ops - Unary Element-wise
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.floor.default])
def _floor_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.floor - floor of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.floor")
    require_kwargs(P.kwargs(n), set(), "aten.floor")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(FloorNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.ceil.default])
def _ceil_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.ceil - ceiling of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.ceil")
    require_kwargs(P.kwargs(n), set(), "aten.ceil")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(CeilNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.square.default])
def _square_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.square - square of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.square")
    require_kwargs(P.kwargs(n), set(), "aten.square")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(SquareNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.exp.default])
def _exp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.exp - exponential of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.exp")
    require_kwargs(P.kwargs(n), set(), "aten.exp")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ExpNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.sin.default])
def _sin_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sin - sine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.sin")
    require_kwargs(P.kwargs(n), set(), "aten.sin")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(SinNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.cos.default])
def _cos_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.cos - cosine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.cos")
    require_kwargs(P.kwargs(n), set(), "aten.cos")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(CosNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.tan.default])
def _tan_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.tan - tangent of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.tan")
    require_kwargs(P.kwargs(n), set(), "aten.tan")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(TanNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.asin.default])
def _asin_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.asin - arc sine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.asin")
    require_kwargs(P.kwargs(n), set(), "aten.asin")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArcsinNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.acos.default])
def _acos_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.acos - arc cosine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.acos")
    require_kwargs(P.kwargs(n), set(), "aten.acos")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArccosNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.atan.default])
def _atan_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.atan - arc tangent of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.atan")
    require_kwargs(P.kwargs(n), set(), "aten.atan")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArctanNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.sinh.default])
def _sinh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sinh - hyperbolic sine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.sinh")
    require_kwargs(P.kwargs(n), set(), "aten.sinh")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(SinhNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.cosh.default])
def _cosh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.cosh - hyperbolic cosine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.cosh")
    require_kwargs(P.kwargs(n), set(), "aten.cosh")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(CoshNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.asinh.default])
def _asinh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.asinh - inverse hyperbolic sine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.asinh")
    require_kwargs(P.kwargs(n), set(), "aten.asinh")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArcsinhNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.acosh.default])
def _acosh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.acosh - inverse hyperbolic cosine of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.acosh")
    require_kwargs(P.kwargs(n), set(), "aten.acosh")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArccoshNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.atanh.default])
def _atanh_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.atanh - inverse hyperbolic tangent of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.atanh")
    require_kwargs(P.kwargs(n), set(), "aten.atanh")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ArctanhNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.log.default])
def _log_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.log - natural logarithm of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.log")
    require_kwargs(P.kwargs(n), set(), "aten.log")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(LogNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.log2.default])
def _log2_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.log2 - base-2 logarithm of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.log2")
    require_kwargs(P.kwargs(n), set(), "aten.log2")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(Log2Node(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.log10.default])
def _log10_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.log10 - base-10 logarithm of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.log10")
    require_kwargs(P.kwargs(n), set(), "aten.log10")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(Log10Node(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.log1p.default])
def _log1p_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.log1p - natural logarithm of (1 + x)."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.log1p")
    require_kwargs(P.kwargs(n), set(), "aten.log1p")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(Log1pNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.erf.default])
def _erf_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.erf - error function of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.erf")
    require_kwargs(P.kwargs(n), set(), "aten.erf")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ErfNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.expm1.default])
def _expm1_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.expm1 - exp(x) - 1 of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.expm1")
    require_kwargs(P.kwargs(n), set(), "aten.expm1")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(Expm1Node(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.round.default])
def _round_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.round - round elements to nearest integer.

    Note: round.decimals variant is not supported as it's not in Core ATen.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.round")
    require_kwargs(P.kwargs(n), set(), "aten.round")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(RoundNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), decimals=0))
    return out


@REGISTRY.register(target=[torch.ops.aten.reciprocal.default])
def _reciprocal_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.reciprocal - 1/x of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.reciprocal")
    require_kwargs(P.kwargs(n), set(), "aten.reciprocal")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(ReciprocalNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.sqrt.default])
def _sqrt_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sqrt - square root of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.sqrt")
    require_kwargs(P.kwargs(n), set(), "aten.sqrt")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(SqrtNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.abs.default])
def _abs_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.abs - absolute value of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.abs")
    require_kwargs(P.kwargs(n), set(), "aten.abs")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(AbsNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.neg.default])
def _neg_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.neg - negation of elements."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.neg")
    require_kwargs(P.kwargs(n), set(), "aten.neg")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(NegNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
    return out


# =============================================================================
# Math Ops - Binary Element-wise
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.atan2.default])
def _atan2_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.atan2 - arc tangent of y/x."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.atan2")
    require_kwargs(P.kwargs(n), set(), "aten.atan2")
    a, b = args[0], args[1]
    out = P.make_or_get_slot(n)
    P.emit(Atan2Node(a=P.slot_to_tid(a), b=P.slot_to_tid(b), out=P.slot_to_tid(out)))
    return out


@REGISTRY.register(target=[torch.ops.aten.logaddexp.default])
def _logaddexp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logaddexp - log(exp(a) + exp(b))."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.logaddexp")
    require_kwargs(P.kwargs(n), set(), "aten.logaddexp")
    a, b = args[0], args[1]
    out = P.make_or_get_slot(n)
    P.emit(
        LogAddExpNode(a=P.slot_to_tid(a), b=P.slot_to_tid(b), out=P.slot_to_tid(out))
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.floor_divide.default])
def _floor_divide_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.floor_divide - floor(a / b)."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.floor_divide")
    require_kwargs(P.kwargs(n), set(), "aten.floor_divide")
    a, b = args[0], args[1]
    out = P.make_or_get_slot(n)
    P.emit(
        FloorDivideNode(a=P.slot_to_tid(a), b=P.slot_to_tid(b), out=P.slot_to_tid(out))
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.pow.Tensor_Tensor, torch.ops.aten.pow.Tensor_Scalar]
)
def _pow_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.pow - a raised to the power of b."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.pow")
    require_kwargs(P.kwargs(n), set(), "aten.pow")
    a = args[0]
    b = args[1]

    # Handle scalar exponent by creating a scalar full tensor that will broadcast
    if not isinstance(b, Slot):
        # Get dtype from input tensor's meta
        input_meta = n.args[0].meta.get("val")
        dtype = input_meta.dtype if input_meta is not None else torch.float32

        # Create a scalar (0-D) tensor for the exponent
        _, b_slot = P.make_tmp_slot()
        P.emit(
            FullNode(
                out=P.slot_to_tid(b_slot),
                shape=[],  # 0-D scalar - broadcasts correctly
                v=FloatOrVid.from_literal(float(b)),
                scalar_type=torch_dtype_to_scalar_type(dtype),
            )
        )
        b = b_slot

    out = P.make_or_get_slot(n)
    P.emit(PowerNode(a=P.slot_to_tid(a), b=P.slot_to_tid(b), out=P.slot_to_tid(out)))
    return out


# =============================================================================
# Math Ops - Reduction
# =============================================================================


@REGISTRY.register(target=[torch.ops.aten.logsumexp.default])
def _logsumexp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logsumexp - log(sum(exp(x))) along axes."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.logsumexp")
    require_kwargs(P.kwargs(n), set(), "aten.logsumexp")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    # Normalize dim to list
    if dim is None:
        axes = []
    elif isinstance(dim, int):
        axes = [dim]
    else:
        axes = list(dim)

    out = P.make_or_get_slot(n)
    P.emit(
        LogSumExpNode(
            x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default])
def _sum_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sum - sum of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 4, "aten.sum")
    require_kwargs(P.kwargs(n), set(), "aten.sum")
    x = args[0]
    axes, keepdim = normalize_reduction_dim(args)

    out = P.make_or_get_slot(n)
    P.emit(
        SumNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim)
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.mean.dim, torch.ops.aten.mean.default])
def _mean_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.mean - mean of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 4, "aten.mean")
    require_kwargs(P.kwargs(n), set(), "aten.mean")
    x = args[0]
    axes, keepdim = normalize_reduction_dim(args)

    out = P.make_or_get_slot(n)
    P.emit(
        MeanNode(
            x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.var.correction, torch.ops.aten.var.dim])
def _var_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.var - variance of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 2, "aten.var")
    require_kwargs(P.kwargs(n), {"correction", "keepdim"}, "aten.var")
    x = args[0]
    axes, _ = normalize_reduction_dim(args)

    # Get correction/ddof and keepdim from kwargs
    correction = n.kwargs.get("correction", None)
    keepdim = n.kwargs.get("keepdim", False)
    ddof = int(correction) if correction is not None else 1

    out = P.make_or_get_slot(n)
    P.emit(
        VarNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axes=axes,
            keepdims=keepdim,
            ddof=ddof,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.std.correction])
def _std_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.std - standard deviation of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 2, "aten.std")
    require_kwargs(P.kwargs(n), {"correction", "keepdim"}, "aten.std")
    x = args[0]
    axes, _ = normalize_reduction_dim(args)

    correction = n.kwargs.get("correction", None)
    keepdim = n.kwargs.get("keepdim", False)
    ddof = int(correction) if correction is not None else 1

    out = P.make_or_get_slot(n)
    P.emit(
        StdNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axes=axes,
            keepdims=keepdim,
            ddof=ddof,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.prod.dim_int, torch.ops.aten.prod.default])
def _prod_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.prod - product of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 4, "aten.prod")
    require_kwargs(P.kwargs(n), set(), "aten.prod")
    x = args[0]
    axes, keepdim = normalize_reduction_dim(args)

    out = P.make_or_get_slot(n)
    P.emit(
        ProdNode(
            x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.amax.default])
def _amax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.amax - max of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.amax")
    require_kwargs(P.kwargs(n), set(), "aten.amax")
    x = args[0]
    axes, keepdim = normalize_reduction_dim(args)

    out = P.make_or_get_slot(n)
    P.emit(
        MaxNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim)
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.max.default])
def _max_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.max.default - global max (reduce all axes)."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.max")
    require_kwargs(P.kwargs(n), set(), "aten.max")
    x = args[0]

    out = P.make_or_get_slot(n)
    P.emit(MaxNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=[], keepdims=False))
    return out


@REGISTRY.register(target=[torch.ops.aten.amin.default])
def _amin_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.amin - min of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.amin")
    require_kwargs(P.kwargs(n), set(), "aten.amin")
    x = args[0]
    axes, keepdim = normalize_reduction_dim(args)

    out = P.make_or_get_slot(n)
    P.emit(
        MinNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim)
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.min.default])
def _min_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.min.default - global min (reduce all axes)."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.min")
    require_kwargs(P.kwargs(n), set(), "aten.min")
    x = args[0]

    out = P.make_or_get_slot(n)
    P.emit(MinNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=[], keepdims=False))
    return out


@REGISTRY.register(target=[torch.ops.aten.argmax.default])
def _argmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.argmax - index of max element along axis."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.argmax")
    require_kwargs(P.kwargs(n), set(), "aten.argmax")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    out = P.make_or_get_slot(n)

    if dim is None:
        # argmax without dim: flatten tensor to 1D, then argmax over axis 0
        # Result is a scalar index into the flattened tensor
        _, flat_slot = P.make_tmp_slot()

        # Get total number of elements from input shape
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise ValueError("Input tensor metadata not found for argmax")
        numel = x_meta.numel()

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(flat_slot),
                shape=[P.to_int_or_vid(numel)],
            )
        )
        P.emit(
            ArgmaxNode(
                x=P.slot_to_tid(flat_slot),
                out=P.slot_to_tid(out),
                axis=0,
                keepdims=False,
            )
        )
    else:
        P.emit(
            ArgmaxNode(
                x=P.slot_to_tid(x), out=P.slot_to_tid(out), axis=dim, keepdims=keepdim
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten.argmin.default])
def _argmin_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.argmin - index of min element along axis."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.argmin")
    require_kwargs(P.kwargs(n), set(), "aten.argmin")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    out = P.make_or_get_slot(n)

    if dim is None:
        # argmin without dim: flatten tensor to 1D, then argmin over axis 0
        # Result is a scalar index into the flattened tensor
        _, flat_slot = P.make_tmp_slot()

        # Get total number of elements from input shape
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise ValueError("Input tensor metadata not found for argmin")
        numel = x_meta.numel()

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(flat_slot),
                shape=[P.to_int_or_vid(numel)],
            )
        )
        P.emit(
            ArgminNode(
                x=P.slot_to_tid(flat_slot),
                out=P.slot_to_tid(out),
                axis=0,
                keepdims=False,
            )
        )
    else:
        P.emit(
            ArgminNode(
                x=P.slot_to_tid(x), out=P.slot_to_tid(out), axis=dim, keepdims=keepdim
            )
        )
    return out


# =============================================================================
# Pooling ops (MaxPool / AvgPool 1d/2d/3d)
# =============================================================================


def _parse_pool_args(args, ndim, op_name):
    """Parse pooling op arguments, normalizing scalars to lists.

    ATen pooling signatures:
      max_pool{N}d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
      avg_pool{N}d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    We only support the first 4 args (input, kernel_size, stride, padding).
    Unsupported args (dilation, ceil_mode, count_include_pad, divisor_override)
    are rejected by require_args.

    Returns (kernel_size, stride, padding) as lists of length ndim.
    """
    require_args(args, 2, 4, op_name)

    kernel_size = args[1]
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    stride = args[2] if len(args) > 2 and args[2] else kernel_size
    if isinstance(stride, int):
        stride = [stride] * ndim
    if not stride:  # empty list means default to kernel_size
        stride = list(kernel_size)

    padding = args[3] if len(args) > 3 else [0] * ndim
    if isinstance(padding, int):
        padding = [padding] * ndim

    return list(kernel_size), list(stride), list(padding)


def _emit_pool_nd(
    P: MLXProgramBuilder,
    n: Node,
    ndim: int,
    reduce_node_cls: type,
    padding_value: float,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
) -> Slot:
    """Emit IR nodes for N-dimensional pooling.

    Decomposes pooling into:
      Transpose (channels-first -> channels-last)
      -> Pad (if needed)
      -> Reshape+Transpose (fast path) or AsStrided (general path)
      -> Max/Mean reduction over kernel dims
      -> Transpose (channels-last -> channels-first)

    Works for 1D, 2D, and 3D pooling uniformly.

    Args:
        P: Program builder.
        n: FX graph node for the pooling op.
        ndim: Spatial dimensionality (1, 2, or 3).
        reduce_node_cls: MaxNode or MeanNode.
        padding_value: Padding fill value (-inf for max, 0 for avg).
        kernel_size: Kernel size per spatial dim, length ndim.
        stride: Stride per spatial dim, length ndim.
        padding: Padding per spatial dim, length ndim.

    Returns:
        Output Slot with shape [N, C, *out_spatial].
    """
    x_node = P.args(n)[0]
    (x,) = P.slot_map([x_node])
    x_meta = n.args[0].meta["val"]
    shape = list(x_meta.shape)  # [N, C, *spatial]

    N = shape[0]
    C = shape[1]
    spatial = shape[2:]  # length == ndim

    # 1. Transpose: channels-first [N, C, *spatial] -> channels-last [N, *spatial, C]
    to_cl = [0] + list(range(2, ndim + 2)) + [1]
    _, cur = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(cur),
            perm=to_cl,
        )
    )

    # 2. Pad spatial dims if needed
    spatial_padded = [s + 2 * p for s, p in zip(spatial, padding)]
    if any(p > 0 for p in padding):
        pad_width = [0, 0]  # batch dim: no pad
        for p in padding:
            pad_width += [p, p]
        pad_width += [0, 0]  # channel dim: no pad
        P.emit(
            PadNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                pad_width=[P.to_int_or_vid(v) for v in pad_width],
                mode="constant",
                constant_value=padding_value,
            )
        )

    # 3. Sliding windows -> [N, *out_spatial, *kernel_size, C]
    out_spatial = [
        (sp - k) // s + 1 for sp, k, s in zip(spatial_padded, kernel_size, stride)
    ]

    can_fast_path = all(
        k == s and sp % k == 0 for k, s, sp in zip(kernel_size, stride, spatial_padded)
    )

    if can_fast_path:
        # Fast path: reshape + transpose (no AsStridedNode needed).
        # [N, *spatial_padded, C]
        #   -> reshape [N, sp0//k0, k0, sp1//k1, k1, ..., C]
        #   -> transpose to gather output-spatial dims, then kernel dims, then C
        reshape_shape = [N]
        for sp, k in zip(spatial_padded, kernel_size):
            reshape_shape += [sp // k, k]
        reshape_shape += [C]

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                shape=[IntOrVid.from_literal(d) for d in reshape_shape],
            )
        )

        # Transpose: gather output-spatial (odd indices), then kernel (even indices after batch)
        # Reshaped tensor axes: [0=batch, 1=out0, 2=k0, 3=out1, 4=k1, ..., last=C]
        last = 2 * ndim + 1
        out_spatial_axes = list(range(1, last, 2))  # [1, 3, 5, ...]
        kernel_axes = list(range(2, last, 2))  # [2, 4, 6, ...]
        perm = [0] + out_spatial_axes + kernel_axes + [last]

        P.emit(
            TransposeNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                perm=perm,
            )
        )
    else:
        # General path: as_strided to create sliding window view.
        # Input layout: [N, *spatial_padded, C] (channels-last, row-major)
        dims = [N] + spatial_padded + [C]
        elem_strides = []
        acc = 1
        for d in reversed(dims):
            elem_strides.append(acc)
            acc *= d
        elem_strides.reverse()

        # as_strided shape: [N, *out_spatial, *kernel_size, C]
        as_shape = [N] + out_spatial + kernel_size + [C]

        # as_strided strides:
        #   batch:          elem_strides[0]
        #   out_spatial[i]: elem_strides[i+1] * stride[i]  (skip by pool stride)
        #   kernel[i]:      elem_strides[i+1]               (consecutive rows/cols)
        #   channel:        1
        as_strides = [elem_strides[0]]
        for i in range(ndim):
            as_strides.append(elem_strides[i + 1] * stride[i])
        for i in range(ndim):
            as_strides.append(elem_strides[i + 1])
        as_strides.append(1)

        P.emit(
            AsStridedNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                shape=[IntOrVid.from_literal(d) for d in as_shape],
                strides=[IntOrVid.from_literal(d) for d in as_strides],
                offset=0,
            )
        )

    # 4. Reduce over kernel dims (axes [ndim+1 .. 2*ndim])
    reduce_axes = list(range(ndim + 1, 2 * ndim + 1))
    _, reduced = P.make_tmp_slot()
    P.emit(
        reduce_node_cls(
            x=P.slot_to_tid(cur),
            out=P.slot_to_tid(reduced),
            axes=reduce_axes,
            keepdims=False,
        )
    )

    # 5. Transpose: channels-last [N, *out_spatial, C] -> channels-first [N, C, *out_spatial]
    to_cf = [0, ndim + 1] + list(range(1, ndim + 1))
    output_slots = P.make_or_get_slots(n)
    out = output_slots[0]
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(reduced),
            out=P.slot_to_tid(out),
            perm=to_cf,
        )
    )
    return out


# --- MaxPool handlers ---


@REGISTRY.register(target=[torch.ops.aten.max_pool2d_with_indices.default])
def _max_pool2d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.max_pool2d_with_indices.

    Returns a tuple (output, indices). Indices are unused during inference;
    make_or_get_slots in _emit_pool_nd allocates both output slots so
    getitem[0] works.
    """
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(
        args, 2, "aten.max_pool2d_with_indices"
    )

    _emit_pool_nd(P, n, 2, MaxNode, float("-inf"), kernel_size, stride, padding)


@REGISTRY.register(target=[torch.ops.aten.max_pool1d.default])
def _max_pool1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.max_pool1d (returns tensor, not tuple)."""
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(args, 1, "aten.max_pool1d")

    return _emit_pool_nd(P, n, 1, MaxNode, float("-inf"), kernel_size, stride, padding)


@REGISTRY.register(target=[torch.ops.aten.max_pool3d_with_indices.default])
def _max_pool3d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.max_pool3d_with_indices.

    Returns a tuple (output, indices). Indices are unused during inference;
    make_or_get_slots in _emit_pool_nd allocates both output slots.
    """
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(
        args, 3, "aten.max_pool3d_with_indices"
    )

    _emit_pool_nd(P, n, 3, MaxNode, float("-inf"), kernel_size, stride, padding)


# --- AvgPool handlers ---


@REGISTRY.register(target=[torch.ops.aten.avg_pool1d.default])
def _avg_pool1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.avg_pool1d."""
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(args, 1, "aten.avg_pool1d")

    return _emit_pool_nd(P, n, 1, MeanNode, 0.0, kernel_size, stride, padding)


@REGISTRY.register(target=[torch.ops.aten.avg_pool2d.default])
def _avg_pool2d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.avg_pool2d."""
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(args, 2, "aten.avg_pool2d")

    return _emit_pool_nd(P, n, 2, MeanNode, 0.0, kernel_size, stride, padding)


@REGISTRY.register(target=[torch.ops.aten.avg_pool3d.default])
def _avg_pool3d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.avg_pool3d."""
    args = P.args(n)
    kernel_size, stride, padding = _parse_pool_args(args, 3, "aten.avg_pool3d")

    return _emit_pool_nd(P, n, 3, MeanNode, 0.0, kernel_size, stride, padding)


# =============================================================================
# Standalone dequantize (torchao.dequantize_affine)
# =============================================================================


@REGISTRY.register(target=[torch.ops.torchao.dequantize_affine.default])
def _dequantize_affine_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle standalone torchao.dequantize_affine (not fused with linear/embedding).

    MLX's dequantize always operates along the last axis.  When the quantized
    dimension is not last (e.g. Conv2d with block_size=[1,32,1,1]), we permute
    the constant weight/scale/zero_point tensors at compile time so the
    quantized dim becomes last, emit the DequantizeNode, then emit a
    TransposeNode with the inverse permutation to restore the original layout.
    """
    parsed = parse_dequant_node(n)
    if parsed is None:
        raise NotImplementedError(
            f"dequantize_affine: unsupported quantization config at {n}"
        )
    (
        qdata_node,
        scale_node,
        zero_point_node,
        group_size,
        bits,
        out_dtype,
        quantized_dim,
    ) = parsed

    qdata_target, qdata = P.get_placeholder_target_and_tensor(qdata_node)
    zero_point_target, zero_point = P.get_placeholder_target_and_tensor(zero_point_node)
    scale_target, scale = P.get_placeholder_target_and_tensor(scale_node)

    if out_dtype is None:
        out_dtype = scale_node.meta["val"].dtype
    out_scalar_type = torch_dtype_to_scalar_type(out_dtype)

    ndim = qdata.ndim
    needs_permute = quantized_dim != ndim - 1

    if needs_permute:
        perm = list(range(ndim))
        perm.remove(quantized_dim)
        perm.append(quantized_dim)
        qdata = qdata.permute(perm).contiguous()
        scale = scale.permute(perm).contiguous()
        zero_point = zero_point.permute(perm).contiguous()

    # to_mlx_qparams expects 2D tensors; flatten N-D to 2D for packing,
    # then restore the (possibly permuted) leading dimensions afterward.
    permuted_shape = qdata.shape
    qdata_2d = qdata.reshape(-1, qdata.shape[-1])
    scale_2d = scale.reshape(-1, scale.shape[-1])
    zero_point_2d = zero_point.reshape(-1, zero_point.shape[-1])

    Q, B = to_mlx_qparams(qdata_2d, scale_2d, zero_point_2d, bits)

    leading_dims = permuted_shape[:-1]
    Q = Q.reshape(*leading_dims, Q.shape[-1])
    scale_nd = scale_2d.reshape(*leading_dims, scale_2d.shape[-1])
    if B is not None:
        B = B.reshape(*leading_dims, B.shape[-1])

    w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
    biases = P.make_or_get_constant(f"{zero_point_target}_to_biases", B)
    scale_const = P.make_or_get_constant(f"{scale_target}_scale", scale_nd)

    if needs_permute:
        _, dequant_tmp = P.make_tmp_slot()
    else:
        dequant_tmp = P.make_or_get_slot(n)

    P.emit(
        DequantizeNode(
            w=P.slot_to_tid(w),
            scales=P.slot_to_tid(scale_const),
            out=P.slot_to_tid(dequant_tmp),
            biases=P.slot_to_tid(biases),
            group_size=group_size,
            bits=bits,
            mode="affine",
            out_scalar_type=out_scalar_type,
        )
    )

    if needs_permute:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = P.make_or_get_slot(n)
        P.emit(
            TransposeNode(
                x=P.slot_to_tid(dequant_tmp),
                out=P.slot_to_tid(out),
                perm=inv_perm,
            )
        )
    else:
        out = dequant_tmp

    return out
