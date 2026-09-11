# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence, Set

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase


_RAW_SPLIT_TARGETS: frozenset[Callable[..., object]] = frozenset(
    {
        torch.ops.aten.chunk.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
    }
)
_EDGE_SPLIT_TARGETS: frozenset[Callable[..., object]] = frozenset(
    {exir_ops.edge.aten.split_with_sizes_copy.default}
)


def _normalize_dim(dim: int, rank: int) -> int:
    normalized = dim + rank if dim < 0 else dim
    assert 0 <= normalized < rank
    return normalized


def _ordered_split_inputs(
    cat_node: Node,
    split_targets: Set[Callable[..., object]],
) -> tuple[Node, list[Node]] | None:
    cat_inputs = get_arg(cat_node, "tensors")
    if not isinstance(cat_inputs, Sequence) or not cat_inputs:
        return None

    getitem_nodes: list[Node] = []
    for inp in cat_inputs:
        if not isinstance(inp, Node) or inp.target != operator.getitem:
            return None
        getitem_nodes.append(inp)

    split_node = getitem_nodes[0].args[0]
    if not isinstance(split_node, Node) or split_node.target not in split_targets:
        return None

    split_outputs = split_node.meta["val"]
    if not isinstance(split_outputs, (tuple, list)) or len(getitem_nodes) != len(
        split_outputs
    ):
        return None
    for index, getitem_node in enumerate(getitem_nodes):
        if getitem_node.args[0] != split_node or getitem_node.args[1] != index:
            return None
    return split_node, getitem_nodes


def _is_view_equivalent(
    cat_node: Node,
    split_node: Node,
    getitem_nodes: Sequence[Node],
) -> bool:
    split_input = get_arg(split_node, "input", Node)
    input_val = split_input.meta["val"]
    output_val = cat_node.meta["val"]
    if not isinstance(input_val, torch.Tensor) or not isinstance(
        output_val, torch.Tensor
    ):
        return False
    if not input_val.is_contiguous() or input_val.numel() != output_val.numel():
        return False

    rank = input_val.ndim
    split_dim = _normalize_dim(get_arg(split_node, "dim", int), rank)
    cat_dim = _normalize_dim(get_arg(cat_node, "dim", int), rank)

    first_moved_dim, last_moved_dim = sorted((split_dim, cat_dim))
    # Moving split parts across axes preserves flattened storage order only
    # when every crossed axis is singleton.
    for getitem_node in getitem_nodes:
        value = getitem_node.meta["val"]
        if not isinstance(value, torch.Tensor) or any(
            size != 1 for size in value.shape[first_moved_dim:last_moved_dim]
        ):
            return False
    return True


class MergeSplitConcatChainPass(PassBase):
    """Replace view-equivalent split/getitem/cat chains with a view.

    For example, splitting a contiguous ``[2, 1, 6, 4, 4]`` tensor into
    three parts along dimension 2 and concatenating them along dimension 1
    is equivalent to a view with shape ``[2, 3, 2, 4, 4]``.
    """

    def _replace_cats(
        self,
        graph_module: GraphModule,
        cat_target: Callable[..., object],
        split_targets: Set[Callable[..., object]],
        view_target: Callable[..., object],
    ) -> bool:
        modified = False
        for cat_node in graph_module.graph.find_nodes(
            op="call_function", target=cat_target
        ):
            match = _ordered_split_inputs(cat_node, split_targets)
            if match is None:
                continue
            split_node, getitem_nodes = match
            if not _is_view_equivalent(cat_node, split_node, getitem_nodes):
                continue

            output_val = cat_node.meta["val"]
            assert isinstance(output_val, torch.Tensor)
            split_input = get_arg(split_node, "input", Node)
            with graph_module.graph.inserting_before(cat_node):
                replacement_view = graph_module.graph.call_function(
                    view_target,
                    (split_input, list(output_val.shape)),
                )
                replacement_view.meta = cat_node.meta.copy()
            cat_node.replace_all_uses_with(replacement_view)
            modified = True
        return modified

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = self._replace_cats(
            graph_module,
            torch.ops.aten.cat.default,
            _RAW_SPLIT_TARGETS,
            torch.ops.aten.view_copy.default,
        )
        modified |= self._replace_cats(
            graph_module,
            exir_ops.edge.aten.cat.default,
            _EDGE_SPLIT_TARGETS,
            exir_ops.edge.aten.view_copy.default,
        )
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
