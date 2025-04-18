# Copyright (c) 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Partitioner for the NXP Neutron NPU

import logging
import operator
from dataclasses import dataclass
from typing import final, List, Dict, Mapping

import torch
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.nn import Parameter

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.converter.node_converter import Target
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import *
from executorch.backends.nxp.nxp_backend import NeutronBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops


class QDQClusterRecognizer:
    """
    Implementation of the Quantize - Dequantize clustering.
    The quantization is captured in the ExecuTorch program using the QDQ (Quantize - DeQuantize) representation. Here
    the inputs to a node comes from some dequantize nodes and outputs goes to some quantize nodes.
    The QDQClusterRecognizer identifies operator performing the quantized arithmetic represented in QDQ form, and the
    corresponding QDQ cluster. The QDQ cluster consists of the:
    - dequantize nodes producing the inputs to the compute node
    - compute node (e.g. conv)
    - auxiliary nodes, like getitem, view_copy, ... which does not perform a core computation
    - quantize nodes processing the output of the compute node.
    """

    @dataclass
    class QDQCluster:
        """
        Dataclass to hold the QDQ cluster instance. For the purpose of Partitioner we hold the list of operators,
        in the QDQ cluster (`ops`) and the compute node what the QDQ cluster is built around.
        The compute node is what is represented in the Neutron IR. the rest of nodes are helpers for data transformation,
        and defines the quantization parameters. This gives the partitioner the ability to:
            - identify if the node is part of a QDQ cluster
            - reference the compute node in the QDQ cluster
        """
        compute_node: torch.fx.Node
        ops: List[torch.fx.Node]

    QUANTIZE_OPERATORS = [
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    ]

    DEQUANTIZE_OPERATORS = [
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    ]

    AUXILIARY_OPS = [
        operator.getitem,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
    ]

    def __init__(self):
        self.cluster_map: dict[str, QDQClusterRecognizer.QDQCluster] = {}

    @staticmethod
    def is_quant_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.QUANTIZE_OPERATORS

    @staticmethod
    def is_dequant_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.DEQUANTIZE_OPERATORS

    @staticmethod
    def is_auxiliary_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.AUXILIARY_OPS

    def get_qdq_cluster_input_part(self, node: torch.fx.Node) -> List[torch.fx.Node]:
        """
        Return the list of nodes representing the input part of the QDQ cluster of the node `node`.
        Those are various dequantization nodes (see DEQUANTIZE_OPERATORS) optionally followed by auxiliary
        nodes.
        If the `node` not meets the QDQ cluster schema, returns empty list.
        """

        # Iterative search for input nodes of the QDQ Cluster:
        nodes_to_check = [node]
        qdq_cluster = []
        while len(nodes_to_check) > 0:
            n = nodes_to_check.pop()
            qdq_cluster.append(n)
            if self.is_dequant_node(n):
                continue
            input_nodes_from_dequant_or_helper = [(self.is_dequant_node(i) or self.is_auxiliary_node(i))
                                                  for i in n.all_input_nodes]
            if all(input_nodes_from_dequant_or_helper):
                nodes_to_check.extend(n.all_input_nodes)
            else:
                return []

        logging.debug(f"Dequant Cluster for {node} is: {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster_output_part(self, node: torch.fx.Node) -> List[torch.fx.Node]:
        """
        Returns the list of nodes representing the output part of the QDQ cluster of the `node`.
        Those are various quantize nodes (see QUANTIZE_OPERATORS) preceded by auxiliary nodes.
        If the `node` not meets the QDQ cluster schema, returns empty list.
        """

        # Iterative search for output nodes of the QDQ Cluster:
        nodes_to_check = [node]
        qdq_cluster = []
        while len(nodes_to_check) > 0:
            n = nodes_to_check.pop()
            qdq_cluster.append(n)
            if self.is_quant_node(n):
                continue
            consumers = [ngn for ngn in list(node.graph.nodes) if n in ngn.all_input_nodes]
            logging.debug(f"\t Users for node {n} are: {consumers}")
            output_nodes_to_quant_or_helper = [(self.is_quant_node(i) or self.is_auxiliary_node(i))
                                               for i in consumers]
            if all(output_nodes_to_quant_or_helper):
                nodes_to_check.extend(consumers)
            else:
                return []

        logging.debug(f"Quant Cluster for {node} is {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster(self, node: torch.fx.Node) -> List[torch.fx.Node]:
        """
        Returns the QDQ cluster of the operator, if quantized. If operator is not quantized, returns empty list.
        """
        logging.debug(node)
        input_qdq_cluster = self.get_qdq_cluster_input_part(node)
        output_qdq_cluster = self.get_qdq_cluster_output_part(node)
        if input_qdq_cluster and output_qdq_cluster:
            return list(set(input_qdq_cluster).union(output_qdq_cluster))
        else:
            return []

    def tag_nodes(self, nodes: List[torch.fx.Node], cluster_name: str) -> None:
        """
        Tags a node and its related dequant and quant nodes with a specified cluster name
        """
        for node in nodes:
            logging.info(f"Tagging node {node} as {cluster_name}")
            node.meta["cluster"] = cluster_name

    def tag_qdq_clusters(self, nodes: List[torch.fx.Node]):
        """
        Identifies QDQ clusters and tag them based on compute operation inside.
        """

        for node in nodes:
            if (node.op == "call_function" and
                not self.is_quant_node(node) and
                not self.is_dequant_node(node)):
                cluster = self.get_qdq_cluster(node)
                if cluster:
                    cluster_name = f"{node.name}_cluster"
                    self.tag_nodes(cluster, cluster_name)
                    self.cluster_map[cluster_name] = self.QDQCluster(node, cluster)


supported_ops = {
    exir_ops.edge.aten.addmm.default: AddMMConverter,
    exir_ops.edge.aten.avg_pool2d.default: AvgPool2dConverter,
    exir_ops.edge.aten.constant_pad_nd.default: ConstantPadNDConverter,
    exir_ops.edge.aten.convolution.default: ConvolutionConverter,
    exir_ops.edge.aten.max_pool2d.default: MaxPool2dConverter,
    exir_ops.edge.aten.max_pool2d_with_indices.default: MaxPool2dConverter,
    exir_ops.edge.aten.mm.default: MMConverter,
    exir_ops.edge.aten.relu.default: ReLUConverter,
    exir_ops.edge.aten._softmax.default: SoftmaxConverter,
    exir_ops.edge.aten.view_copy.default: ViewCopyConverter,
}


class NeutronSupportedOperators(OperatorSupportBase):

    def __init__(self, qdq_clusters: Dict[str, QDQClusterRecognizer.QDQCluster], target: Target,
                 operators_not_to_delegate: List[str], parameters_mapping: dict[str, Parameter]):
        self.qdq_clusters = qdq_clusters
        self.target = target
        self.operators_not_to_delegate = operators_not_to_delegate
        self.parameters_mapping = parameters_mapping

    def _is_node_quantized(self, node: torch.fx.node.Node):
        return "cluster" in node.meta

    def _is_node_call_function(self, node: torch.fx.node.Node):
        return node.op == "call_function"

    def is_node_delegatable(self, node: torch.fx.node.Node):
        if self.operators_not_to_delegate != ['']:
            any_non_delegatable = any(x in node.name for x in self.operators_not_to_delegate)
            return not any_non_delegatable
        return True

    def _is_node_supported_compute(self, node: torch.fx.node.Node) -> bool:
        """
        Operator checking function for compute nodes.
        """
        if not self.is_node_delegatable(node):
            return False

        if (node_converter := supported_ops.get(node.target, None)) is None:
            # There is no `NodeConverter` for this `node`.
            return False

        return (
            self._is_node_call_function(node) and
            self._is_node_quantized(node) and

            # TODO: `view_copy` node should be delegated only if it's not the only operator in the cluster.
            node_converter.is_supported(node, self.target, self.parameters_mapping)
        )

    def _is_node_supported_non_compute(self, node: torch.fx.node.Node) -> bool:
        """
        If the node is a quantize, dequantize or auxiliary node inside a QDQ cluster, the support on Neutron
        is determined by the support of the compute operator.
        """
        return (self._is_node_quantized(node) and
                self._is_node_supported_compute(self.qdq_clusters[node.meta["cluster"]].compute_node))

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        """
        Check if the Edge operator is supported on Neutron.
        """

        if (QDQClusterRecognizer.is_quant_node(node) or
            QDQClusterRecognizer.is_dequant_node(node) or
            QDQClusterRecognizer.is_auxiliary_node(node)):
            return self._is_node_supported_non_compute(node)
        else:
            return self._is_node_supported_compute(node)


@final
class NeutronPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(NeutronBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("NeutronPartitioner::partition")
        partition_tags = {}

        graph_module = exported_program.graph_module
        nodes = list(graph_module.graph.nodes)

        qdq_clusterer = QDQClusterRecognizer()

        qdq_clusterer.tag_qdq_clusters(nodes)

        graph_module.recompile()
        target = self.delegation_spec[1][2].value
        target = Target(target.decode())

        operators_not_to_delegate = self.delegation_spec[1][3].value.decode().split(',')
        logging.info(f"Operators not to delegate: {operators_not_to_delegate}")

        parameters_mapping = EdgeProgramToIRConverter.map_inputs_to_parameters(exported_program)
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NeutronSupportedOperators(qdq_clusterer.cluster_map, target, operators_not_to_delegate, parameters_mapping),
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
