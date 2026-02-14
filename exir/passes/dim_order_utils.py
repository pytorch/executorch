# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Set

import torch

from executorch.exir.tensor import dim_order_from_stride

try:
    from executorch.exir.dialects._ops import ops as exir_ops
except ImportError:
    exir_ops = None  # type: ignore[assignment]


def _is_dim_order_op_with_explicit_arg(op: object) -> bool:
    """True if the op takes an explicit dim_order kwarg (e.g. _clone_dim_order, _to_dim_order_copy)."""
    if exir_ops is None:
        return False
    return op in (
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    ) or (
        hasattr(exir_ops.edge.dim_order_ops._clone_dim_order, "out")
        and op == exir_ops.edge.dim_order_ops._clone_dim_order.out
    ) or (
        hasattr(exir_ops.edge.dim_order_ops._to_dim_order_copy, "out")
        and op == exir_ops.edge.dim_order_ops._to_dim_order_copy.out
    )


def _format_preserving_ops() -> Set[object]:
    """Build set of format-preserving ops (aten and edge dialect)."""
    ops: Set[object] = {
        torch.ops.aten.clone.out,
        torch.ops.aten.clone.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
    }
    if hasattr(torch.ops.aten.clone, "memory_format"):
        ops.add(torch.ops.aten.clone.memory_format)
    if exir_ops is not None:
        ops.add(exir_ops.edge.aten.clone.default)
        ops.add(exir_ops.edge.dim_order_ops._clone_dim_order.default)
        if hasattr(exir_ops.edge.dim_order_ops._clone_dim_order, "out"):
            ops.add(exir_ops.edge.dim_order_ops._clone_dim_order.out)
    return ops


# Format-preserving ops: output layout must match primary input. Include out-variants
# because when SpecPropPass runs, OutVarPass has already converted e.g. clone.default
# to clone.out.
FORMAT_PRESERVING_OPS: Set[object] = _format_preserving_ops()


def dim_order_from_fake_tensor(t: torch.Tensor) -> Optional[List[int]]:
    """
    Derive ExecuTorch dim_order from a tensor's strides (e.g. contiguous -> [0,1,2,3],
    channels_last -> [0,2,3,1]). Returns None if layout cannot be expressed (e.g. 0 in strides).
    """
    try:
        st = t.stride()
        result = dim_order_from_stride(st)
        return list(result)
    except ValueError:
        return None


def get_explicit_output_dim_order(
    node: "torch.fx.Node",
) -> Optional[List[int]]:
    """
    If the node is a dim_order op (_clone_dim_order, _to_dim_order_copy) with
    an explicit dim_order in kwargs, return it. Otherwise return None so the
    caller can propagate from the primary input (format-preserving).
    """
    if not _is_dim_order_op_with_explicit_arg(node.target):
        return None
    dim_order_val = node.kwargs.get("dim_order") if node.kwargs else None
    if dim_order_val is None:
        return None
    if isinstance(dim_order_val, (list, tuple)) and len(dim_order_val) > 0:
        if all(isinstance(i, int) for i in dim_order_val):
            return list(dim_order_val)
    return None


def should_propagate_dim_order(op: object) -> bool:
    """True if the op is format-preserving and we should propagate primary input dim_order to out."""
    return op in FORMAT_PRESERVING_OPS
