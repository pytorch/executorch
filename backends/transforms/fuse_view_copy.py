# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Set, Type

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.node import map_arg


class FuseViewCopyTransform(ExportPass):
    """Fuse redundant ``view_copy`` nodes through supported unary elementwise
    chains.

    Example:
        let view(sN) be a view node with output shape sN

        Input:
        view(s0) -> view(s1) -> abs -> view(s2) -> sqrt -> view(s3) -> user0
                                                               `-----> user1
        Output:
        view(s3) -> abs -> sqrt -> user0
                             `---> user1

        Note that all view nodes except the first one have been elimininated.
        That first node now outputs the final view shape s3 instead. user0 and
        user1 have been reconnected to the last retained node sqrt because the
        former connection has been broken with the removal of the view nodes.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    VIEW_OP: EdgeOpOverload = exir_ops.edge.aten.view_copy.default

    UNARY_ELEMENTWISE_OPS: list[EdgeOpOverload] = [
        exir_ops.edge.aten.alias_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.ceil.default,
        exir_ops.edge.aten.floor.default,
        exir_ops.edge.aten.neg.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.round.default,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.silu.default,
        exir_ops.edge.aten.sqrt.default,
        exir_ops.edge.aten.tanh.default,
        exir_ops.edge.aten.sign.default,
        exir_ops.edge.aten.reciprocal.default,
        exir_ops.edge.aten.gelu.default,
        exir_ops.edge.aten.rsqrt.default,
        exir_ops.edge.aten.exp.default,
        exir_ops.edge.aten.log.default,
    ]

    def _find_view_copy_chain(
        self, start_node: torch.fx.Node, ops: list[EdgeOpOverload]
    ) -> list[torch.fx.Node]:
        """Collect removable view nodes in a fusible chain."""

        end_node = start_node
        view_nodes: list[torch.fx.Node] = []
        while (
            end_node.op == "call_function"
            and end_node.target in ops
            and len(end_node.users) == 1
            and (end_node_user := next(iter(end_node.users))).target in ops
        ):
            if end_node_user.target == self.VIEW_OP:
                view_nodes.append(end_node_user)

            end_node = end_node_user

        return view_nodes

    def _shape_nodes(self, shape: Any) -> set[torch.fx.Node]:
        nodes: set[torch.fx.Node] = set()

        def collect_node(node: torch.fx.Node) -> torch.fx.Node:
            nodes.add(node)
            return node

        map_arg(shape, collect_node)
        return nodes

    def _topologically_valid_shape(
        self, graph: torch.fx.Graph, view_node: torch.fx.Node, shape: Any
    ) -> bool:
        shape_nodes = self._shape_nodes(shape)
        if len(shape_nodes) == 0:
            return True

        seen_nodes: set[torch.fx.Node] = set()
        for node in graph.nodes:
            if node is view_node:
                return shape_nodes.issubset(seen_nodes)
            seen_nodes.add(node)

        return False

    def merge_view_copy_chains(
        self, graph: torch.fx.Graph
    ) -> tuple[torch.fx.Graph, bool]:
        """Merge redundant view nodes in linear chains.

        For each view node, search forward through a single-user chain of view
        and supported unary elementwise ops. If the chain contains later view
        nodes, update the first view to the final view shape and bypass each
        later view by replacing its uses with its input. The view nodes are
        then eliminated.

        Args:
            graph (torch.fx.Graph): Graph to rewrite.

        Returns:
            tuple[torch.fx.Graph, bool]: The rewritten graph and whether it was
                modified.
        """
        modified = False
        ops: list[EdgeOpOverload] = self.UNARY_ELEMENTWISE_OPS + [self.VIEW_OP]
        for node in graph.find_nodes(op="call_function", target=self.VIEW_OP):
            view_nodes_to_remove = self._find_view_copy_chain(node, ops)

            # Fuse the longest prefix whose final shape is available at the
            # first view. A later shape producer must remain after that view.
            while view_nodes_to_remove and not self._topologically_valid_shape(
                graph, node, view_nodes_to_remove[-1].args[1]
            ):
                view_nodes_to_remove.pop()

            if len(view_nodes_to_remove) == 0:
                continue

            final_shape = view_nodes_to_remove[-1].args[1]

            modified = True

            # Set the first view node in the chain to have the final shape.
            node.args = (node.args[0], final_shape)

            # Redirect output edges from removed view nodes to bypass them.
            for view_node in view_nodes_to_remove:
                view_node.replace_all_uses_with(view_node.args[0])

        if modified:
            graph.eliminate_dead_code()

        return graph, modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph, modified = self.merge_view_copy_chains(graph_module.graph)
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
