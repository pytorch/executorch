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

# Operator aliases for better readability.
AddMM = exir_ops.edge.aten.addmm.default
ViewCopy = exir_ops.edge.aten.view_copy.default
MM = exir_ops.edge.aten.mm.default
Conv = exir_ops.edge.aten.convolution.default
HardTanh = exir_ops.edge.aten.hardtanh.default
Relu = exir_ops.edge.aten.relu.default
Sigmoid = exir_ops.edge.aten.sigmoid.default
Tanh = exir_ops.edge.aten.tanh.default
Clone = exir_ops.edge.aten.clone.default
CloneDimOrder = exir_ops.edge.dim_order_ops._clone_dim_order.default


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
        hasattr(node_, "op")
        and node_.op == "call_function"
        and node_.target
        == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    )


def _is_quantize(node_: Node) -> bool:
    return (
        hasattr(node_, "op")
        and node_.op == "call_function"
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
    ...┤ <main_cluster_node> ├...   ──────────────►          │                  │ newly added nodes
       └──────────┬──────────┘                         ┌─────▼──────┐           │
                  ▼                                    │ dequantize │           │
                  .                                    └─────┬──────┘           ┘
             ┌────▼─────┐                         ┌──────────▼──────────┐
             │ quantize │                      ...┤ <main_cluster_node> ├...
             └────┬─────┘                         └──────────┬──────────┘
                  ▼                                          ▼
                                                             .
                                                        ┌────▼─────┐
                                                        │ quantize │
                                                        └────┬─────┘
                                                             ▼
    """

    # Dictionary mapping main cluster nodes to auxiliary nodes, for which this optimization will be applied.
    main_cluster_node_to_auxiliary_nodes = {
        AddMM: [
            ViewCopy,
        ],
        MM: [
            ViewCopy,
        ],
        ViewCopy: [Clone, CloneDimOrder],
    }

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for aux_node in graph_module.graph.nodes:
            if aux_node.op != "call_function":
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
            if main_cluster_node.op != "call_function":
                continue

            if aux_node.target not in self.main_cluster_node_to_auxiliary_nodes.get(
                main_cluster_node.target, []
            ):
                # Unsupported main cluster node and auxiliary node pair.
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
            │ dequantize │                                    .
            └─────┬──────┘                         ┌──────────▼──────────┐
                  ▼                             ...┤ <main_cluster_node> ├...
                  .                                └──────────┬──────────┘
       ┌──────────▼──────────┐       replaced with       ┌────▼─────┐            ┐
    ...┤ <main_cluster_node> ├...   ──────────────►      │ quantize │            │
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

    # Dictionary mapping main cluster nodes to auxiliary nodes, for which this optimization will be applied.
    main_cluster_node_to_auxiliary_nodes = {
        AddMM: [
            ViewCopy,
            HardTanh,
            Relu,
            Sigmoid,
            Tanh,
        ],
        MM: [
            ViewCopy,
            HardTanh,
            Relu,
            Sigmoid,
            Tanh,
        ],
        Conv: [
            HardTanh,
            Relu,
            Sigmoid,
            Tanh,
        ],
        ViewCopy: [Clone, CloneDimOrder],
    }

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:

        for aux_node in graph_module.graph.nodes:
            if aux_node.op != "call_function":
                continue

            main_cluster_node = aux_node.args[0]
            if not (
                hasattr(main_cluster_node, "op")
                and main_cluster_node.op == "call_function"
            ):
                continue

            if aux_node.target not in self.main_cluster_node_to_auxiliary_nodes.get(
                main_cluster_node.target, []
            ):
                # Unsupported main cluster node and auxiliary node pair.
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
