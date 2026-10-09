# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


_CAT = exir_ops.edge.aten.cat.default
_SLICE = exir_ops.edge.aten.slice_copy.Tensor


class RewriteCatSlicePass(ArmPass):
    """Replace concat slices with concats of their overlapping inputs.

    Rewrites static unit-stride slices along a concat dimension without
    materializing the source concat. Applies only when every source user is a
    compatible slice and the replacements do not increase the operation count.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if _try_rewrite_cat_slices(node):
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)


def _is_cat(node: Node) -> bool:
    return node.op == "call_function" and node.target == _CAT


def _is_slice(node: Node) -> bool:
    return node.op == "call_function" and node.target == _SLICE


def _cat_inputs(node: Node) -> list[Node] | None:
    if not node.args or not isinstance(node.args[0], (list, tuple)):
        return None
    inputs = list(node.args[0])
    return (
        inputs if all(isinstance(input_node, Node) for input_node in inputs) else None
    )


def _cat_dim(node: Node) -> int | None:
    if len(node.args) > 1 and isinstance(node.args[1], int):
        return node.args[1]
    dim = node.kwargs.get("dim")
    return dim if isinstance(dim, int) else None


def _node_shape(node: Node) -> tuple[int, ...] | None:
    shape = getattr(node.meta.get("val"), "shape", None)
    if shape is None or not all(isinstance(size, int) for size in shape):
        return None
    return tuple(shape)


def _slice_args(node: Node, rank: int) -> tuple[int, int, int] | None:
    if len(node.args) < 4:
        return None
    dim_arg, start_arg, end_arg = node.args[1:4]
    step_arg = node.args[4] if len(node.args) > 4 else 1
    if not all(
        isinstance(value, int) for value in (dim_arg, start_arg, end_arg, step_arg)
    ):
        return None
    dim = cast(int, dim_arg)
    start = cast(int, start_arg)
    end = cast(int, end_arg)
    step = cast(int, step_arg)
    dim = dim + rank if dim < 0 else dim
    return (dim, start, end) if 0 <= dim < rank and step == 1 else None


def _slice_range(start: int, end: int, dim_size: int) -> tuple[int, int] | None:
    start = max(0, min(dim_size, start + dim_size if start < 0 else start))
    end = max(0, min(dim_size, end + dim_size if end < 0 else end))
    return (start, end) if start < end else None


def _overlapping_inputs(
    inputs: list[Node], dim: int, start: int, end: int, rank: int
) -> list[tuple[Node, int, int, int, int]] | None:
    overlaps: list[tuple[Node, int, int, int, int]] = []
    offset = 0
    for input_index, input_node in enumerate(inputs):
        input_shape = _node_shape(input_node)
        if input_shape is None or len(input_shape) != rank:
            return None
        input_size = input_shape[dim]
        input_end = offset + input_size
        overlap_start = max(start, offset)
        overlap_end = min(end, input_end)
        if overlap_start < overlap_end:
            overlaps.append(
                (
                    input_node,
                    input_index,
                    overlap_start - offset,
                    overlap_end - offset,
                    input_size,
                )
            )
        offset = input_end
    return overlaps


def _create_slice(
    graph: torch.fx.Graph,
    input_node: Node,
    dim: int,
    start: int,
    end: int,
    from_node: Node,
    input_qparams: Any | None,
) -> Node:
    with graph.inserting_before(from_node):
        slice_node = create_node(
            graph,
            _SLICE,
            args=(input_node, dim, start, end, 1),
            from_node=from_node,
            inherit_qparams=True,
        )
    val = input_node.meta.get("val")
    if val is not None and hasattr(val, "new_empty") and hasattr(val, "shape"):
        shape = list(val.shape)
        shape[dim] = end - start
        slice_node.meta["val"] = val.new_empty(tuple(shape))
    if input_qparams is not None:
        slice_node.meta["input_qparams"] = {0: input_qparams}
    return slice_node


def _replacement_inputs(
    node: Node,
    overlaps: list[tuple[Node, int, int, int, int]],
    dim: int,
    input_qparams: Any | None,
) -> list[Node]:
    inputs: list[Node] = []
    for input_node, _, start, end, input_size in overlaps:
        if (
            start == 0
            and end == input_size
            and (len(overlaps) > 1 or input_qparams is None)
        ):
            inputs.append(input_node)
        else:
            inputs.append(
                _create_slice(
                    node.graph,
                    input_node,
                    dim,
                    start,
                    end,
                    node,
                    input_qparams,
                )
            )
    return inputs


def _replacement_op_count(
    overlaps: list[tuple[Node, int, int, int, int]], input_qparams: Any | None
) -> int:
    slice_count = sum(
        not (
            start == 0
            and end == input_size
            and (len(overlaps) > 1 or input_qparams is None)
        )
        for _, _, start, end, input_size in overlaps
    )
    return slice_count + (len(overlaps) > 1)


def _source_replacement_op_count(
    source: Node,
    inputs: list[Node],
    dim: int,
    rank: int,
    dim_size: int,
    input_qparams: Any | None,
) -> int | None:
    op_count = 0
    for user in source.users:
        slice_args = _slice_args(user, rank)
        if slice_args is None or slice_args[0] != dim:
            return None
        slice_range = _slice_range(slice_args[1], slice_args[2], dim_size)
        if slice_range is None:
            return None
        overlaps = _overlapping_inputs(inputs, dim, *slice_range, rank)
        if not overlaps:
            return None
        op_count += _replacement_op_count(overlaps, input_qparams)
    return op_count


def _can_fuse_slice(node: Node, source: Node) -> bool:
    if not _is_slice(node) or not node.args or node.args[0] is not source:
        return False

    inputs = _cat_inputs(source)
    dim = _cat_dim(source)
    source_shape = _node_shape(source)
    if inputs is None or dim is None or source_shape is None:
        return False
    rank = len(source_shape)
    dim = dim + rank if dim < 0 else dim
    slice_args = _slice_args(node, rank)
    if not 0 <= dim < rank or slice_args is None or slice_args[0] != dim:
        return False
    slice_range = _slice_range(slice_args[1], slice_args[2], source_shape[dim])
    if slice_range is None:
        return False

    return bool(_overlapping_inputs(inputs, dim, *slice_range, rank))


def _try_rewrite_cat_slices(source: Node) -> bool:
    if not _is_cat(source) or not source.users:
        return False

    users = list(source.users)
    if not all(_can_fuse_slice(user, source) for user in users):
        return False

    inputs = _cat_inputs(source)
    dim = _cat_dim(source)
    source_shape = _node_shape(source)
    if inputs is None or dim is None or source_shape is None:
        return False
    rank = len(source_shape)
    dim = dim + rank if dim < 0 else dim
    if not 0 <= dim < rank:
        return False

    source_input_qparams = source.meta.get("input_qparams")
    source_input_qparams = (
        source_input_qparams if isinstance(source_input_qparams, dict) else None
    )
    input_qparams = (
        source_input_qparams.get(0) if source_input_qparams is not None else None
    )
    replacement_op_count = _source_replacement_op_count(
        source,
        inputs,
        dim,
        rank,
        source_shape[dim],
        input_qparams,
    )
    if replacement_op_count is None or replacement_op_count > 1 + len(users):
        return False

    for node in users:
        slice_args = _slice_args(node, rank)
        if slice_args is None:
            return False
        slice_range = _slice_range(slice_args[1], slice_args[2], source_shape[dim])
        if slice_range is None:
            return False
        overlaps = _overlapping_inputs(inputs, dim, *slice_range, rank)
        if not overlaps:
            return False
        replacement_inputs = _replacement_inputs(node, overlaps, dim, input_qparams)
        if len(replacement_inputs) == 1:
            replacement = replacement_inputs[0]
        else:
            with node.graph.inserting_before(node):
                replacement = create_node(
                    node.graph,
                    _CAT,
                    args=(replacement_inputs, dim),
                    from_node=node,
                    inherit_qparams=True,
                )
            if input_qparams is not None:
                replacement.meta["input_qparams"] = {0: input_qparams}
        node.replace_all_uses_with(replacement)

    return True
