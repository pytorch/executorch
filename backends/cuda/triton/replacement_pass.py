# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Triton Kernel Replacement.

This pass replaces ATen operators with optimized Triton kernels in the graph.
"""

import logging

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)
triton = torch.ops.triton

# Global mapping from edge dialect operators to Triton kernel functions
EDGE_TO_TRITON_KERNELS = {
    exir_ops.edge.aten.scaled_dot_product_attention.default: triton.sdpa,
}


class ReplaceEdgeOpWithTritonOpPass(PassBase):
    """
    Pass to replace ATen operators with Triton kernels.

    This pass scans the graph for Edge operators that have registered Triton
    replacements using EDGE_TO_TRITON_KERNELS and replaces them with the
    optimized Triton implementations.
    """

    def __init__(self):
        """Initialize the pass."""
        super().__init__()
        self._replacement_count = 0

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Execute the pass on the graph module.

        Args:
            graph_module: The graph module to transform

        Returns:
            PassResult indicating success/failure and the modified graph module
        """
        self._replacement_count = 0
        modified = False

        if not EDGE_TO_TRITON_KERNELS:
            return PassResult(graph_module, False)

        # Iterate through all nodes in the graph
        for node in graph_module.graph.nodes:
            if self._should_replace_node(node):
                try:
                    self._replace_node_with_triton(graph_module, node)
                    modified = True
                    self._replacement_count += 1
                except Exception as e:
                    logger.warning(f"Failed to replace node {node.name}: {e}")
                    # Continue with other replacements even if one fails

        if modified:
            # Recompile the graph module after modifications
            graph_module.recompile()

        # logger.info(f"Replaced {self._replacement_count} nodes with Triton kernels")
        print(f"Replaced {self._replacement_count} nodes with Triton kernels")

        return PassResult(graph_module, modified)

    def _should_replace_node(self, node: Node) -> bool:
        """
        Check if a node should be replaced with a Triton kernel.

        Args:
            node: The node to check

        Returns:
            True if the node should be replaced
        """
        # Only consider call_function nodes
        if node.op != "call_function":
            return False

        return node.target in EDGE_TO_TRITON_KERNELS

    def _replace_node_with_triton(self, graph_module: GraphModule, node: Node) -> None:
        """
        Replace an edge dialect node with a Triton kernel call.

        Args:
            graph_module: The graph module containing the node
            node: The node to replace
        """
        # Get the target operator (should be an exir_ops edge dialect op)
        target = node.target

        # Get the replacement kernel
        if target not in EDGE_TO_TRITON_KERNELS:
            raise ValueError(f"No replacement kernel found for {target}")

        triton_kernel_fn = EDGE_TO_TRITON_KERNELS[target]

        # Create a new node with the Triton kernel
        with graph_module.graph.inserting_before(node):
            # The triton_kernel_fn is already registered as a custom op via @triton_op
            # We can call it directly
            new_node = graph_module.graph.call_function(
                triton_kernel_fn,
                args=node.args,
                kwargs=node.kwargs,
            )

            # Copy metadata from original node
            new_node.meta = node.meta.copy()

        # Replace all uses of the old node with the new node
        node.replace_all_uses_with(new_node)

        # Remove the old node
        graph_module.graph.erase_node(node)
