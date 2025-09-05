# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.backends.nxp.neutron_partitioner import QDQClusterRecognizer
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassResult


def insert_qdq_pair_after_node(
    graph: torch.fx.Graph, anchor: torch.fx.Node, q_params: tuple
):
    # Insert a Quantize node.
    with graph.inserting_after(anchor):
        quantize_op = graph.create_node(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(),  # Will be added later.
        )
        quantize_op.meta = anchor.meta

    # Insert a Dequantize node.
    with graph.inserting_after(quantize_op):
        dequantize_op = graph.create_node(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(quantize_op,) + q_params,
        )
        dequantize_op.meta = quantize_op.meta
    anchor.replace_all_uses_with(dequantize_op)

    # Add this at the end, so the `anchor.replace_all_uses_with(dequantize_op)` does not replace the first use of the
    #  `quantize_op`.
    quantize_op.args = (anchor,) + q_params


def _is_dequantize(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target
        == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    )


def _is_quantize(node_: Node) -> bool:
    return (
        node_.op == "call_function"
        and node_.target
        == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    )


class MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass(NeutronEdgePass):
    """
                                                           │
                                                     ┌─────▼──────┐
                │                                    │ dequantize │
          ┌─────▼──────┐                             └─────┬──────┘
          │ dequantize │                             ┌─────▼──────┐
          └─────┬──────┘                             │ <aux_node> │
          ┌─────▼──────┐                             └─────┬──────┘
          │ <aux_node> │                              ┌────▼─────┐            ┐
          └─────┬──────┘                              │ quantize │            │
     ┌──────────▼──────────┐       replaced with      └────┬─────┘            │
    ⋯┤ <main_cluster_node> ├⋯     ──────────────►          │                  │ newly added nodes
     └──────────┬──────────┘                         ┌─────▼──────┐           │
                ▼                                    │ dequantize │           │
                ⋮                                    └─────┬──────┘           ┘
           ┌────▼─────┐                         ┌──────────▼──────────┐
           │ quantize │                        ⋯┤ <main_cluster_node> ├⋯
           └────┬─────┘                         └──────────┬──────────┘
                ▼                                          ▼
                                                           ⋮
                                                      ┌────▼─────┐
                                                      │ quantize │
                                                      └────┬─────┘
                                                           ▼
    """

    allowed_auxiliary_nodes = [exir_ops.edge.aten.view_copy.default]

    # List of approved nodes to which the <aux_node> can be connected in order for the pass to make the modification.
    allowed_main_cluster_nodes = [
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.mm.default,
    ]

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for aux_node in graph_module.graph.nodes:
            if (
                aux_node.op != "call_function"
                or aux_node.target not in self.allowed_auxiliary_nodes
            ):
                continue

            dequantize_node = aux_node.args[0]
            if not _is_dequantize(dequantize_node):
                # Not the intended use case.
                continue

            users = list(aux_node.users.keys())
            if len(users) != 1:
                # Not the intended use case.
                continue

            main_cluster_node = users[0]
            if (
                main_cluster_node.op != "call_function"
                or main_cluster_node.target not in self.allowed_main_cluster_nodes
            ):
                # Unsupported `main_cluster_node`.
                continue

            # Make sure the nodes are part of the same QDQ cluster.
            cluster = QDQClusterRecognizer().get_qdq_cluster(main_cluster_node)
            if any(
                node_ not in cluster
                for node_ in [dequantize_node, aux_node, main_cluster_node]
            ):
                continue

            # ---- The nodes follow the pattern described in the header. ----

            q_params = dequantize_node.args[1:]
            insert_qdq_pair_after_node(graph_module.graph, aux_node, q_params)

            # The graph has now changed, and we shouldn't keep iterating through it. Return the new graph and the parent
            #  class will call this pass again.
            return PassResult(graph_module, True)

        # Nothing was changed.
        return PassResult(graph_module, False)


class MoveTrailingAuxiliaryOperatorIntoSeparateQDQClusterPass(NeutronEdgePass):
    """
                                                            │
                                                      ┌─────▼──────┐
                │                                     │ dequantize │
          ┌─────▼──────┐                              └─────┬──────┘
          │ dequantize │                                    ⋮
          └─────┬──────┘                         ┌──────────▼──────────┐
                ▼                               ⋯┤ <main_cluster_node> ├⋯
                ⋮                                └──────────┬──────────┘
     ┌──────────▼──────────┐       replaced with       ┌────▼─────┐            ┐
    ⋯┤ <main_cluster_node> ├⋯     ──────────────►      │ quantize │            │
     └──────────┬──────────┘                           └────┬─────┘            │
          ┌─────▼──────┐                                    │                  │ newly added nodes
          │ <aux_node> │                              ┌─────▼──────┐           │
          └─────┬──────┘                              │ dequantize │           │
           ┌────▼─────┐                               └─────┬──────┘           ┘
           │ quantize │                               ┌─────▼──────┐
           └────┬─────┘                               │ <aux_node> │
                ▼                                     └─────┬──────┘
                                                       ┌────▼─────┐
                                                       │ quantize │
                                                       └────┬─────┘
                                                            ▼
    """

    allowed_auxiliary_nodes = [exir_ops.edge.aten.view_copy.default]

    # List of approved nodes to which the `<aux_node>` can be connected in order for the pass to make the modification.
    allowed_main_cluster_nodes = [
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.mm.default,
    ]

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:

        for aux_node in graph_module.graph.nodes:
            if (
                aux_node.op != "call_function"
                or aux_node.target not in self.allowed_auxiliary_nodes
            ):
                continue

            main_cluster_node = aux_node.args[0]
            if (
                main_cluster_node.op != "call_function"
                or main_cluster_node.target not in self.allowed_main_cluster_nodes
            ):
                # Unsupported `main_cluster_node`.
                continue

            users = list(aux_node.users.keys())
            if len(users) != 1:
                # Not the intended use case.
                continue

            quantize_node = users[0]
            if not _is_quantize(quantize_node):
                # Not the intended use case.
                continue

            # Make sure the nodes are part of the same QDQ cluster.
            cluster = QDQClusterRecognizer().get_qdq_cluster(main_cluster_node)
            if any(
                node_ not in cluster
                for node_ in [quantize_node, aux_node, main_cluster_node]
            ):
                continue

            # ---- The nodes follow the pattern described in the header. ----

            q_params = quantize_node.args[1:]
            insert_qdq_pair_after_node(graph_module.graph, main_cluster_node, q_params)

            # The graph has now changed, and we shouldn't keep iterating through it. Return the new graph and the parent
            #  class will call this pass again.
            return PassResult(graph_module, True)

        # Nothing was changed.
        return PassResult(graph_module, False)
