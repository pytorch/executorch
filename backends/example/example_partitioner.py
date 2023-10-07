# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final

import torch
from executorch.backends.example.example_backend import ExampleBackend
from executorch.backends.example.example_operators.ops import module_to_annotator
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase


@final
class ExamplePartitioner(Partitioner):
    """
    Partitions all add/mul nodes regardless of order
    """

    def __init__(self) -> None:
        self.patterns = module_to_annotator.keys()
        self.delegation_spec = DelegationSpec(ExampleBackend.__name__, [])

        class DequantQuantOperatorSupport(OperatorSupportBase):
            def is_node_supported(self, _submodules, node: torch.fx.Node) -> bool:
                return node.op == "call_function" and node.target in [
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                ]

        self.dequant_quant_support = DequantQuantOperatorSupport()

    def _partition_graph_module(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_nodes = []
        for pattern in self.patterns:
            fused_partitions = find_sequential_partitions(
                edge_graph_module,
                pattern,
            )

            for fused_partition in fused_partitions:
                for partition in fused_partition:
                    partition_nodes.append(partition.nodes)

        partitions = generate_partitions_from_list_of_nodes(
            edge_graph_module, partition_nodes, self.dequant_quant_support
        )

        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                if node.op == "call_function":
                    for arg_node in node.args:
                        if (
                            isinstance(arg_node, torch.fx.Node)
                            and arg_node.op == "get_attr"
                        ):
                            arg_node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        for _, submodule, _ in get_control_flow_submodules(edge_graph_module):
            submodule_partition_tags = self._partition_graph_module(submodule)
            partition_tags.update(submodule_partition_tags)

        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tag = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tag
        )
