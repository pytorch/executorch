# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Partitioner for the NXP Neutron NPU

import logging
import operator
from dataclasses import dataclass
from typing import final, Mapping

import torch

from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.edge_helper import (
    DEQUANTIZE_OPERATORS,
    is_no_op_on_neutron,
    QUANTIZE_OPERATORS,
)
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import is_not_qdq_node
from torch.export.exported_program import ExportedProgram
from torch.fx import Graph
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.nn import Parameter
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import *  # noqa F403
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.backend.node_format_inference import NodeFormatInference
from executorch.backends.nxp.nxp_backend import NeutronBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops

NXP_DO_NOT_DELEGATE = "NXP_DO_NOT_DELEGATE"
NXP_DELEGATION_TAG = "delegation_tag"


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
        ops: list[torch.fx.Node]

    AUXILIARY_OPS = [
        operator.getitem,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.tanh.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    ]

    def __init__(self):
        self.cluster_map: dict[str, QDQClusterRecognizer.QDQCluster] = {}

    @staticmethod
    def is_quant_node(node: torch.fx.Node) -> bool:
        return node.target in QUANTIZE_OPERATORS

    @staticmethod
    def is_dequant_node(node: torch.fx.Node) -> bool:
        return node.target in DEQUANTIZE_OPERATORS

    @staticmethod
    def is_auxiliary_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.AUXILIARY_OPS

    def get_qdq_cluster_input_part(self, node: torch.fx.Node) -> list[torch.fx.Node]:
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
            input_nodes_from_dequant_or_helper = [
                (self.is_dequant_node(i) or self.is_auxiliary_node(i))
                for i in n.all_input_nodes
            ]
            if all(input_nodes_from_dequant_or_helper):
                nodes_to_check.extend(n.all_input_nodes)
            else:
                return []

        logging.debug(f"Dequant Cluster for {node} is: {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster_output_part(self, node: torch.fx.Node) -> list[torch.fx.Node]:
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
            consumers = [
                ngn for ngn in list(node.graph.nodes) if n in ngn.all_input_nodes
            ]
            logging.debug(f"\t Users for node {n} are: {consumers}")
            output_nodes_to_quant_or_helper = [
                (self.is_quant_node(i) or self.is_auxiliary_node(i)) for i in consumers
            ]
            if all(output_nodes_to_quant_or_helper):
                nodes_to_check.extend(consumers)
            else:
                return []

        logging.debug(f"Quant Cluster for {node} is {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster(self, node: torch.fx.Node) -> list[torch.fx.Node]:
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

    def tag_nodes(self, nodes: list[torch.fx.Node], cluster_name: str) -> None:
        """
        Tags a node and its related dequant and quant nodes with a specified cluster name
        """
        for node in nodes:
            logging.info(f"Tagging node {node} as {cluster_name}")
            node.meta["cluster"] = cluster_name

    def tag_qdq_clusters(self, nodes: list[torch.fx.Node]):
        """
        Identifies QDQ clusters and tag them based on compute operation inside.
        """

        for node in nodes:
            if (
                node.op == "call_function"
                and not self.is_quant_node(node)
                and not self.is_dequant_node(node)
            ):
                cluster = self.get_qdq_cluster(node)
                if cluster:
                    cluster_name = f"{node.name}_cluster"
                    self.tag_nodes(cluster, cluster_name)
                    self.cluster_map[cluster_name] = self.QDQCluster(node, cluster)


supported_ops = {
    exir_ops.edge.aten.abs.default: AbsConverter,  # noqa F405
    exir_ops.edge.aten._adaptive_avg_pool2d.default: AdaptiveAvgPool2dConverter,  # noqa F405
    exir_ops.edge.aten.addmm.default: AddMMConverter,  # noqa F405
    exir_ops.edge.aten.add.Tensor: AddTensorConverter,  # noqa F405
    exir_ops.edge.aten.avg_pool2d.default: AvgPool2dConverter,  # noqa F405
    exir_ops.edge.aten.cat.default: CatConverter,  # noqa F405
    exir_ops.edge.aten.clamp.default: ClampConverter,  # noqa F405
    exir_ops.edge.aten.clone.default: CloneConverter,  # noqa F405
    exir_ops.edge.dim_order_ops._clone_dim_order.default: CloneConverter,  # noqa F405
    exir_ops.edge.aten.constant_pad_nd.default: ConstantPadNDConverter,  # noqa F405
    exir_ops.edge.aten.convolution.default: ConvolutionConverter,  # noqa F405
    exir_ops.edge.aten.hardtanh.default: HardTanhConverter,  # noqa F405
    exir_ops.edge.aten.max_pool2d.default: MaxPool2dConverter,  # noqa F405
    exir_ops.edge.aten.max_pool2d_with_indices.default: MaxPool2dConverter,  # noqa F405
    exir_ops.edge.aten.mean.dim: MeanDimConverter,  # noqa F405
    exir_ops.edge.aten.mm.default: MMConverter,  # noqa F405
    exir_ops.edge.aten.mul.Tensor: MulTensorConverter,  # noqa F405
    exir_ops.edge.aten.neg.default: NegConverter,  # noqa F405
    exir_ops.edge.aten.permute_copy.default: PermuteCopyConverter,  # noqa F405
    exir_ops.edge.aten.relu.default: ReLUConverter,  # noqa F405
    exir_ops.edge.aten.sigmoid.default: SigmoidConverter,  # noqa F405
    exir_ops.edge.aten.slice_copy.Tensor: SliceTensorConverter,  # noqa F405
    exir_ops.edge.aten._softmax.default: SoftmaxConverter,  # noqa F405
    exir_ops.edge.aten.sub.Tensor: SubTensorConverter,  # noqa F405
    exir_ops.edge.aten.tanh.default: TanhConverter,  # noqa F405
    exir_ops.edge.aten.upsample_bilinear2d.vec: UpsampleBilinear2DConverter,  # noqa F405
    exir_ops.edge.aten.upsample_nearest2d.vec: UpsampleNearest2DConverter,  # noqa F405
    exir_ops.edge.aten.view_copy.default: ViewCopyConverter,  # noqa F405
}


class NeutronSupportedOperators(OperatorSupportBase):

    def __init__(
        self,
        qdq_clusters: dict[str, QDQClusterRecognizer.QDQCluster],
        neutron_target_spec: NeutronTargetSpec,
        operators_not_to_delegate: list[str],
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ):
        self.qdq_clusters = qdq_clusters
        self.neutron_target_spec = neutron_target_spec
        self.operators_not_to_delegate = operators_not_to_delegate
        self.parameters_mapping = parameters_mapping
        self.custom_delegation_options = custom_delegation_options

    def _is_node_quantized(self, node: torch.fx.node.Node):
        return "cluster" in node.meta

    def _is_node_call_function(self, node: torch.fx.node.Node):
        return node.op == "call_function"

    def is_node_delegatable(self, node: torch.fx.node.Node):
        if self.operators_not_to_delegate != [""]:
            any_non_delegatable = any(
                x in node.name for x in self.operators_not_to_delegate
            )
            return not any_non_delegatable
        return True

    def _is_node_supported_compute(self, node: torch.fx.node.Node) -> bool:
        """
        Operator checking function for compute nodes.
        """

        if hasattr(node, "meta") and node.meta.get(NXP_DO_NOT_DELEGATE, False):
            # The delegation of this node has been prohibited.
            return False

        if not self.is_node_delegatable(node):
            return False

        if (node_converter := supported_ops.get(node.target, None)) is None:
            # There is no `NodeConverter` for this `node`.
            return False

        # noinspection PyUnresolvedReferences
        return (
            self._is_node_call_function(node)
            and self._is_node_quantized(node)
            and node_converter.is_supported(
                node,
                self.neutron_target_spec,
                self.parameters_mapping,
                self.custom_delegation_options,
            )
        )

    def _is_node_supported_non_compute(self, node: torch.fx.node.Node) -> bool:
        """
        If the node is a quantize, dequantize or auxiliary node inside a QDQ cluster, the support on Neutron
        is determined by the support of the compute operator.
        """
        return self._is_node_quantized(node) and self._is_node_supported_compute(
            self.qdq_clusters[node.meta["cluster"]].compute_node
        )

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Check if the Edge operator is supported on Neutron.
        """

        if (
            QDQClusterRecognizer.is_quant_node(node)
            or QDQClusterRecognizer.is_dequant_node(node)
            or QDQClusterRecognizer.is_auxiliary_node(node)
        ):
            return self._is_node_supported_non_compute(node)
        else:
            return self._is_node_supported_compute(node)


@final
class NeutronPartitioner(Partitioner):
    def __init__(
        self,
        compile_spec: list[CompileSpec],
        neutron_target_spec: NeutronTargetSpec,
        custom_delegation_options: CustomDelegationOptions | None = None,
        post_quantization_state_dict: dict[str, Parameter] | None = None,
    ) -> None:
        """Class responsible for identifying partitions suited for delegation to the eIQ Neutron NPU.

        :param compile_spec: A list of compile specifications for Neutron.
        :param neutron_target_spec: Object for querying the target platform to retrieve its properties.
        :param custom_delegation_options: Custom user options which affect node delegation.
        :param post_quantization_state_dict: State-dict of the model right after quantization. During partitioning, the
                                              `edge_program` only contains fake tensors without any data. In this case,
                                              this state dict is used instead (if provided). Notice: It may potentially
                                              contain outdated data,
        """
        super().__init__()
        self.delegation_spec = DelegationSpec(NeutronBackend.__name__, compile_spec)
        self.custom_delegation_options = (
            custom_delegation_options or CustomDelegationOptions()
        )
        self.neutron_target_spec = neutron_target_spec
        self.post_quantization_state_dict = post_quantization_state_dict

    @staticmethod
    def _partition_contains_compute_nodes(
        partition: Partition, parameters_mapping: dict[str, Parameter]
    ) -> bool:
        non_q_dq_partition_nodes = list(filter(is_not_qdq_node, partition.nodes))

        if all(
            is_no_op_on_neutron(node, parameters_mapping)
            for node in non_q_dq_partition_nodes
        ):
            # All operators in the partition are no-ops from the perspective of Neutron.
            return False

        return True

    def validate_partitioning_result(
        self,
        graph: Graph,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        """Verify that the result of the partitioning can be safely used for delegation.

        :param graph: Original graph that was partitioned.
        :param partition_list: List of the resulting partitions.
        :param custom_delegation_options: Custom user options which affect node delegation.
        :param parameters_mapping: Dict mapping tensor names to their static data. Should be inferred from the
                                   `state_dict` attribute of an edge program.
        :return: True, if the partitioning result is valid.
        """
        if len(partition_list) == 0:
            # If there are no partitions, the partitioning is trivially valid.
            return True

        partitioning_valid = True

        if not custom_delegation_options.allow_no_op_partitions:
            # Make sure all every partition contains at least 1 compute node. A partition with just no-op nodes will
            #  result in an empty Neutron delegated partition, causing an exception.
            for partition in partition_list:
                if not self._partition_contains_compute_nodes(
                    partition, parameters_mapping
                ):
                    # None of the nodes in this partition are compute nodes. Make sure they will not be delegated.
                    partitioning_valid = False
                    for node in partition.nodes:
                        node.meta[NXP_DO_NOT_DELEGATE] = True

        # Verify that all node-specific partitioning requirements are satisfied.
        all_delegated_nodes = {
            node for partition in partition_list for node in partition.nodes
        }
        for node in graph.nodes:
            if (
                node in all_delegated_nodes
                and hasattr(node, "target")
                and node.target in supported_ops
            ):
                if not supported_ops[node.target].supports_partitioning_result(
                    node,
                    partition_list,
                    custom_delegation_options,
                    self.neutron_target_spec,
                    parameters_mapping,
                ):
                    # This node is not supported within its partition. Exclude it from delegation in the future.
                    partitioning_valid = False
                    node.meta[NXP_DO_NOT_DELEGATE] = True

        return partitioning_valid

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("NeutronPartitioner::partition")
        partition_tags = {}
        partition_list = []

        graph_module = exported_program.graph_module
        nodes = list(graph_module.graph.nodes)

        qdq_cluster_recognizer = QDQClusterRecognizer()
        qdq_cluster_recognizer.tag_qdq_clusters(nodes)

        graph_module.recompile()

        operators_not_to_delegate = self.delegation_spec[1][4].value.decode().split(",")
        logging.info(f"Operators not to delegate: {operators_not_to_delegate}")

        parameters_mapping = EdgeProgramToIRConverter.map_inputs_to_parameters(
            exported_program, self.post_quantization_state_dict
        )
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NeutronSupportedOperators(
                qdq_cluster_recognizer.cluster_map,
                self.neutron_target_spec,
                operators_not_to_delegate,
                parameters_mapping,
                self.custom_delegation_options,
            ),
            allows_single_node_partition=True,
        )

        # Identify the format (NCHW/NHWC/...) for all nodes in the graph, and store it in the `node.meta`.
        # This format will be used by the `CapabilityBasedPartitioner` to determine which nodes will be delegated.
        NodeFormatInference(exported_program).identify_node_formats()

        parameters_mapping = EdgeProgramToIRConverter.map_inputs_to_parameters(
            exported_program, self.post_quantization_state_dict
        )

        iteration_limit = len(exported_program.graph.nodes)
        for _ in range(iteration_limit):
            # Run the partitioning.
            partition_list = capability_partitioner.propose_partitions()

            # Check if the nodes support the partitioning result. Mark the problematic nodes with `NXP_DO_NOT_DELEGATE`.
            partitioning_valid = self.validate_partitioning_result(
                exported_program.graph,
                partition_list,
                self.custom_delegation_options,
                parameters_mapping,
            )
            if partitioning_valid:
                # The result of the partitioning is fine
                break

        # Mark the partitions in the node `meta` attribute.
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta[NXP_DELEGATION_TAG] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
