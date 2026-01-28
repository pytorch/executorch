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
from typing import Tuple

import torch
from executorch.backends.apple.mlx.program_builder import (
    _torch_dtype_to_dtypeid,
    emit_stop_position,
    MLXProgramBuilder,
    REGISTRY,
    Slot,
)
from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
    AddmmNode,
    AddNode,
    AddScalarNode,
    ARangeNode,
    ContiguousNode,
    Conv1DNode,
    ExpandDimsNode,
    GatherNode,
    GeluNode,
    IdCopyNode,
    IntOrVid,
    ItemIntNode,
    LayerNormNode,
    LinearNode,
    MulNode,
    ReshapeNode,
    RMSNormNode,
    RopeNode,
    SiluNode,
    SliceNode,
    SliceUpdateNode,
    SymSizeNode,
    TakeAlongAxisNode,
    TileNode,
    TransposeNode,
)
from torch.fx.node import Node


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
    P._emit(
        SliceUpdateNode(
            dst=P._slot_to_tid(cache_slot),
            update=P._slot_to_tid(value_slot),
            axis=IntOrVid.from_literal(1),  # S dimension in [B, S, H, D]
            start=P._to_int_or_vid(start_slot),
            stop=P._to_int_or_vid(stop_slot),
        )
    )

    # Return tuple of (token, updated_cache)
    # - token_slot: create a placeholder (token is not actually used)
    # - cache_slot: the cache that was updated in-place by SliceUpdateNode
    _, token_slot = P.slot_manager.make_tmp_slot()

    # The token is a dummy value that's not used. We emit an IdCopyNode
    # from value to token just to have something valid there.
    P._emit(
        IdCopyNode(
            x=P._slot_to_tid(value_slot),  # Copy from value as a placeholder
            out=P._slot_to_tid(token_slot),
        )
    )

    return (token_slot, cache_slot)


@REGISTRY.register(target=[torch.ops.aten.linear.default])
def _linear_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    x, w = args[0], args[1]
    b = args[2] if len(args) > 2 else None
    out = P.make_or_get_slot(n)

    P._emit(
        LinearNode(
            x=P._slot_to_tid(x),
            weight=P._slot_to_tid(w),
            out=P._slot_to_tid(out),
            bias=P._slot_to_tid(b) if b else None,
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
    bias, mat1, mat2 = args[0], args[1], args[2]

    # Get kwargs for beta and alpha (default to 1)
    kwargs = P.kwargs(n)
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    out = P.make_or_get_slot(n)

    # For now, only support the common case where beta=1 and alpha=1
    # This is equivalent to: mat1 @ mat2 + bias
    if beta != 1 or alpha != 1:
        raise ValueError(
            f"addmm with beta={beta}, alpha={alpha} not yet supported, only beta=1, alpha=1"
        )

    # Emit AddmmNode: computes mat1 @ mat2 + bias using matmul directly
    P._emit(
        AddmmNode(
            mat1=P._slot_to_tid(mat1),
            mat2=P._slot_to_tid(mat2),
            out=P._slot_to_tid(out),
            bias=P._slot_to_tid(bias),
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.view.default, torch.ops.aten.view_copy.default]
)
def _view_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, shape = P.args(n)
    out = P.make_or_get_slot(n)

    shape_iovs = [P._to_int_or_vid(s) for s in shape]
    P._emit(
        ReshapeNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            shape=shape_iovs,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.clone.default, torch.ops.aten.alias.default])
def _clone_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    (x,) = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        ContiguousNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
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
        x = args[0]
        out = P.make_or_get_slot(n)
        P._emit(
            ContiguousNode(
                x=P._slot_to_tid(x),
                out=P._slot_to_tid(out),
            )
        )
        return out

except ImportError:
    # Edge IR ops not available (e.g., when building from ATen dialect)
    pass


@REGISTRY.register(target=[torch.ops.aten.embedding.default])
def _embedding_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    w, x = args[0], args[1]
    out = P.make_or_get_slot(n)
    P._emit(
        GatherNode(
            table_=P._slot_to_tid(w),
            ids=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.add.Tensor])
def _add_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)

    # Check if both inputs are scalars (not tensors)
    # We can't support scalar + scalar because:
    # 1. Scalars get lifted to tensors during export
    # 2. But the return type would be a tensor, not a scalar
    # 3. ExecuTorch would expect a scalar return value
    a_is_scalar = not isinstance(a, Slot)
    b_is_scalar = not isinstance(b, Slot)
    if a_is_scalar and b_is_scalar:
        raise ValueError(
            "aten.add.Tensor with both scalar inputs is not supported. "
            "Use operator.add for scalar arithmetic."
        )

    out = P.make_or_get_slot(n)
    P._emit(
        AddNode(
            a=P._slot_to_tid(a),
            b=P._slot_to_tid(b),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.add])
def _add_scalar_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        AddScalarNode(
            a=P._to_int_or_vid(a),
            b=P._to_int_or_vid(b),
            out=P._slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.mul.Tensor])
def _mul_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)
    out = P.make_or_get_slot(n)

    # Check if both inputs are scalars (not tensors)
    # We can't support scalar * scalar because:
    # 1. Scalars get lifted to tensors during export
    # 2. But the return type would be a tensor, not a scalar
    # 3. ExecuTorch would expect a scalar return value
    a_is_scalar = not isinstance(a, Slot)
    b_is_scalar = not isinstance(b, Slot)
    if a_is_scalar and b_is_scalar:
        raise ValueError(
            "aten.mul.Tensor with both scalar inputs is not supported. "
            "Use operator.mul for scalar arithmetic."
        )

    # Handle scalar multiplication by creating constants
    if isinstance(a, float):
        a = P.make_or_get_constant(
            f"_scalar_{a}", torch.tensor([a], dtype=n.meta["val"].dtype)
        )
    if isinstance(b, float):
        b = P.make_or_get_constant(
            f"_scalar_{b}", torch.tensor([b], dtype=n.meta["val"].dtype)
        )

    P._emit(
        MulNode(
            a=P._slot_to_tid(a),
            b=P._slot_to_tid(b),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.silu.default])
def _silu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    (x,) = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        SiluNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.gelu.default])
def _gelu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    (x,) = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        GeluNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default]
)
def _permute_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, dims = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        TransposeNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            perm=list(dims),
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.transpose.int, torch.ops.aten.transpose_copy.int]
)
def _transpose_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, dim0, dim1 = P.args(n)
    perm = list(range(len(n.meta["val"].shape)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    out = P.make_or_get_slot(n)
    P._emit(
        TransposeNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            perm=perm,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor]
)
def _slice_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, dim, start, stop = P.args(n)
    if start is None:
        start = 0
    out = P.make_or_get_slot(n)
    P._emit(
        SliceNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            axis=P._to_int_or_vid(dim),
            start=P._to_int_or_vid(start),
            stop=P._to_int_or_vid(stop),
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
    x, dim, start, length = P.args(n)
    out = P.make_or_get_slot(n)

    # Convert narrow (start, length) to slice (start, end)
    # The end is start + length
    start_iov = P._to_int_or_vid(start)
    length_iov = P._to_int_or_vid(length)

    # For stop = start + length, we need to emit an ADD_SCALAR if either is a Vid
    if isinstance(start_iov, IntOrVid) and start_iov.vid is not None:
        # start is a Vid, need to add at runtime
        if isinstance(length_iov, IntOrVid) and length_iov.vid is not None:
            # Both are Vids - emit add to compute stop
            stop_vid = P.make_tmp_vid()
            P._emit(
                AddScalarNode(
                    a=start_iov.vid,
                    b=length_iov.vid,
                    out=stop_vid,
                )
            )
            stop_iov = IntOrVid(int64=None, vid=stop_vid)
        else:
            # start is Vid, length is int - emit add scalar
            stop_vid = P.make_tmp_vid()
            P._emit(
                AddScalarNode(
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
        stop_vid = P.make_tmp_vid()
        P._emit(
            AddScalarNode(
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

    P._emit(
        SliceNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            axis=P._to_int_or_vid(dim),
            start=start_iov,
            stop=stop_iov,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.unsqueeze.default, torch.ops.aten.unsqueeze_copy.default]
)
def _unsqueeze_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, axis = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        ExpandDimsNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            axis=axis,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.repeat.default])
def _repeat_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, reps = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        TileNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            reps=list(reps),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.index.Tensor])
def _index_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x, idx_list = P.args(n)
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
    P._emit(
        TakeAlongAxisNode(
            x=P._slot_to_tid(x),
            indices=P._slot_to_tid(idx_list[0]),
            out=P._slot_to_tid(out),
            axis=0,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sym_size.int])
def _sym_size_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    a, dim = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        SymSizeNode(
            a=P._slot_to_tid(a),
            dim=dim,
            out=P._slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.item.default])
def _item_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    if not isinstance(n.meta["val"], torch.SymInt):
        raise ValueError("item only supported if it returns a SymInt")
    (x,) = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        ItemIntNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_vid(out),
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
    a, idx = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        IdCopyNode(
            x=P._slot_to_tid(a[idx]),
            out=P._slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.layer_norm.default])
def _layer_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    x, shape = args[0:2]
    if len(shape) > 1:
        raise ValueError(
            "LayerNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    out = P.make_or_get_slot(n)
    P._emit(
        LayerNormNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            weight=P._slot_to_tid(w) if w else None,
            bias=P._slot_to_tid(bias) if bias else None,
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.arange.default])
def _arange_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with just stop, or (start, stop) or (start, stop, step).

    Supports both static (literal int) and dynamic (Slot from item()) values.
    """
    args = P.args(n)
    if len(args) == 1:
        start = 0
        stop = args[0]
    else:
        start, stop = args[0:2]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = n.kwargs.get("dtype", torch.int64)
    dtype_id = _torch_dtype_to_dtypeid(dtype)

    out = P.make_or_get_slot(n)
    P._emit(
        ARangeNode(
            out=P._slot_to_tid(out),
            start=P._to_int_or_vid(start),
            stop=P._to_int_or_vid(stop),
            step=P._to_int_or_vid(step),
            dtype=dtype_id,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.arange.start_step])
def _arange_start_step_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with start, end, and step arguments.

    Supports both static (literal int) and dynamic (Slot from item()) start/stop/step.
    """
    args = P.args(n)
    start = args[0]
    stop = args[1]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = n.kwargs.get("dtype", torch.int64)
    dtype_id = _torch_dtype_to_dtypeid(dtype)

    out = P.make_or_get_slot(n)
    P._emit(
        ARangeNode(
            out=P._slot_to_tid(out),
            start=P._to_int_or_vid(start),
            stop=P._to_int_or_vid(stop),
            step=P._to_int_or_vid(step),
            dtype=dtype_id,
        )
    )
    return out


# =============================================================================
# Custom MLX ops
# =============================================================================


@REGISTRY.register(target=[torch.ops.mlx.rms_norm.default])
def _rms_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    x, w = args[0], args[1]
    eps = args[2] if len(args) >= 3 else 1e-5
    out = P.make_or_get_slot(n)
    P._emit(
        RMSNormNode(
            x=P._slot_to_tid(x),
            weight=P._slot_to_tid(w),
            out=P._slot_to_tid(out),
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.mlx.rope.default])
def _rope_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    x, head_dim, pos = args[0], args[1], args[2]
    traditional = args[3] if len(args) > 3 else False
    base = args[4] if len(args) > 4 else 500000.0
    scale = args[5] if len(args) > 5 else 1.0
    freqs = args[6] if len(args) > 6 else None
    out = P.make_or_get_slot(n)

    # pos must be a Slot (SymInt) from input_pos.item() during tracing
    # The schema only supports Vid for pos, not literal int
    if not isinstance(pos, Slot):
        raise ValueError(
            f"RopeNode.pos must be a SymInt (traced via tensor.item()), got {type(pos)}. "
            "Make sure input_pos is a tensor and you call input_pos.item() to get a SymInt."
        )

    P._emit(
        RopeNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
            head_dim=head_dim,
            pos=P._slot_to_vid(pos),
            freqs=P._slot_to_tid(freqs) if freqs else None,
            traditional=traditional,
            base=base,
            scale=scale,
        )
    )

    return out


@REGISTRY.register(target=[torch.ops.aten.conv1d.default])
def _conv1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    stride = n.args[3] if len(n.args) > 3 else 1
    if isinstance(stride, list):
        assert len(stride) == 1
        stride = stride[0]
    padding = n.args[4] if len(n.args) > 4 else 0
    if isinstance(padding, list):
        assert len(padding) == 1
        padding = padding[0]
    dilation = n.args[5] if len(n.args) > 5 else 1
    groups = n.args[6] if len(n.args) > 6 else 1

    # Weight needs to be transposed: [O, I/G, K] -> [O, K, I]
    w_target, w_tensor = P.get_placeholder_target_and_tensor(w_node)
    w = P.make_or_get_constant(
        f"{w_target}_channel_last", w_tensor.permute([0, 2, 1]).contiguous()
    )

    x, bias = P.slot_map([x_node, bias_node])

    # Transpose input: (N, C_in, W) -> (N, W, C_in)
    tmp_name, tmp = P.slot_manager.make_tmp_slot()
    P._emit(
        TransposeNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(tmp),
            perm=[0, 2, 1],
        )
    )

    # Conv1D
    P._emit(
        Conv1DNode(
            x=P._slot_to_tid(tmp),
            w=P._slot_to_tid(w),
            out=P._slot_to_tid(tmp),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    )

    # Add bias if present
    if bias is not None:
        tmp2_name, tmp2 = P.slot_manager.make_tmp_slot()
        P._emit(
            ReshapeNode(
                x=P._slot_to_tid(bias),
                out=P._slot_to_tid(tmp2),
                shape=[
                    IntOrVid.from_literal(1),
                    IntOrVid.from_literal(1),
                    IntOrVid.from_literal(-1),
                ],
            )
        )
        P._emit(
            AddNode(
                a=P._slot_to_tid(tmp),
                b=P._slot_to_tid(tmp2),
                out=P._slot_to_tid(tmp),
            )
        )

    # Transpose output: (N, W, C_out) -> (N, C_out, W)
    out = P.make_or_get_slot(n)
    P._emit(
        TransposeNode(
            x=P._slot_to_tid(tmp),
            out=P._slot_to_tid(out),
            perm=[0, 2, 1],
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.clamp.default])
def _clamp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    # TODO: This is a hack that removes clamp from the graph
    # It's to address torch inserting clamps for fp16
    x, _min, _max = P.args(n)
    out = P.make_or_get_slot(n)
    P._emit(
        IdCopyNode(
            x=P._slot_to_tid(x),
            out=P._slot_to_tid(out),
        )
    )
    return out
