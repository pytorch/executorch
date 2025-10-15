# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_helper import get_quantization_parameters_for
from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.backends.nxp.neutron_partitioner import QDQClusterRecognizer
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult


class RemoveAdditionalQDQClustersPass(NeutronEdgePass):
    """
    After delegation of partitions, there may be additional dequantize quantize nodes for QDQ clusters that were
    not delegated. If dequantize quantize nodes are quantized per tensor and quantization parameters of dequantize
    and quantize nodes in a QDQ cluster are equal, the nodes can be removed and thus the inner nodes computed in int8.

                                         │
                            ┌────────────▼──────────┐
                            │ dequantize_per_tensor │
                            └────────────┬──────────┘
                                         │                                    │
                                     ┌───▼──┐        replace with         ┌───▼──┐
                                     │ node │       ──────────────►       │ node │
                                     └───┬──┘                             └───┬──┘
                                         │                                    ▼
                             ┌───────────▼─────────┐
                             │ quantize_per_tensor │
                             └───────────┬─────────┘
                                         ▼

    """

    qdq_per_channel_nodes = (
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    )

    qdq_per_tensor_nodes = (
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    )

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        nodes = list(graph_module.graph.nodes)
        qdq_clusterer = QDQClusterRecognizer()
        qdq_clusterer.tag_qdq_clusters(nodes)

        for cluster in qdq_clusterer.cluster_map.values():
            # For now, enable only permute_copy and cat.
            if cluster.compute_node.target not in [
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.aten.cat.default,
            ]:
                continue

            # Ensure cluster doesn't contain dequantize/quantize per channel nodes.
            if any(
                node
                for node in cluster.ops
                if node.target in self.qdq_per_channel_nodes
            ):
                continue

            qdq_nodes = [
                node for node in cluster.ops if node.target in self.qdq_per_tensor_nodes
            ]

            qdq_nodes_quant_params = [
                get_quantization_parameters_for(node) for node in qdq_nodes
            ]

            equal_quant_scales = [
                np.allclose(
                    qdq_nodes_quant_params[idx][0], qdq_nodes_quant_params[idx + 1][0]
                )
                for idx in range(len(qdq_nodes_quant_params[:-1]))
            ]

            equal_quant_zero_points = [
                np.allclose(
                    qdq_nodes_quant_params[idx][1], qdq_nodes_quant_params[idx + 1][1]
                )
                for idx in range(len(qdq_nodes_quant_params[:-1]))
            ]

            # Check if all quantization params are equal to ensure that QDQ cluster can be removed.
            if not all(equal_quant_scales + equal_quant_zero_points):
                continue

            # Replace the uses of each dequantize/quantize node with its arg node.
            for qdq_node in qdq_nodes:
                qdq_node.replace_all_uses_with(qdq_node.args[0])
                graph_module.graph.erase_node(qdq_node)

            # Remove compute node cluster info from node meta.
            cluster.compute_node.meta.pop("cluster")

            graph_module = self.recompile_module(graph_module)

            # The graph has now changed, and we cannot keep iterating through it. Return the new graph and the parent
            #  class will call this pass again.
            return PassResult(graph_module, True)

        return PassResult(graph_module, False)
