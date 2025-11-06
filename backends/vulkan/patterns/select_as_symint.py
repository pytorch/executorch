# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


class SelectAsSymIntMatch(PatternMatch):
    def __init__(self, local_scalar_dense_node: torch.fx.Node) -> None:
        self.anchor_node = local_scalar_dense_node
        self.match_found = False

        # Check if the input to local_scalar_dense is a select_copy node
        if len(local_scalar_dense_node.args) < 1:
            return

        select_node = local_scalar_dense_node.args[0]
        if not isinstance(select_node, torch.fx.Node):
            return

        if (
            select_node.op != "call_function"
            or select_node.target != exir_ops.edge.aten.select_copy.int
        ):
            return

        # select_copy.int has signature: select_copy(Tensor self, int dim, int index)
        if len(select_node.args) < 3:
            return

        self.select_node = select_node

        self.tensor_node = select_node.args[0]
        self.dim_node = select_node.args[1]
        self.index_node = select_node.args[2]

        self.all_nodes = [
            self.anchor_node,
            self.select_node,
            self.tensor_node,
            self.dim_node,
            self.index_node,
        ]

        self.match_found = True


@register_pattern_detector("select_as_symint")
def find_select_as_symint_patterns(
    node: torch.fx.Node,
) -> Optional[SelectAsSymIntMatch]:
    if node.target != torch.ops.aten._local_scalar_dense.default:
        return None

    matched_pattern = SelectAsSymIntMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


@register_pattern_replacement("select_as_symint")
def replace_select_local_scalar_dense_with_select_as_symint(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: SelectAsSymIntMatch,
):
    with graph_module.graph.inserting_before(match.anchor_node):
        new_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.select_as_symint.default,
            args=(
                match.tensor_node,
                match.dim_node,
                match.index_node,
            ),
        )

    new_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(new_node)

    # # Remove both the local_scalar_dense and select_copy nodes
    # graph_module.graph.erase_node(match.anchor_node)
    # # Only erase select_node if it has no other users
    # if len(match.select_node.users) == 0:
    #     graph_module.graph.erase_node(match.select_node)
