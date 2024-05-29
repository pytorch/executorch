# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, final, List, Optional, Tuple

import torch
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class AddOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
        ]


class MatmulOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.mm.default,
        ]


@final
class AddMulPartitionerDemo(Partitioner):
    """
    Partitions all add/mul nodes regardless of order
    """

    def __init__(self) -> None:
        self.op_support = any_chain(AddOperatorSupport(), MatmulOperatorSupport())
        self.delegation_spec = DelegationSpec(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([4]))],
        )

    def _partition_graph_module(
        self,
        graph_module: torch.fx.GraphModule,
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        for _, submodule, _ in get_control_flow_submodules(graph_module):
            ret_partition_tags = self._partition_graph_module(submodule)
            partition_tags.update(ret_partition_tags)

        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


@final
class AddAttributePartitionerDemo(Partitioner):
    """
    Partitions all add and get_attr nodes
    """

    def __init__(self) -> None:
        self.op_support = AddOperatorSupport()

        self.delegation_spec = DelegationSpec(BackendWithCompilerDemo.__name__, [])

    def partition(self, edge_exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = {}
        partition_list = generate_pattern_op_partitions(
            edge_exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                partition_tags[delegation_tag] = self.delegation_spec

                # Tag the add nodes
                node.meta["delegation_tag"] = delegation_tag

                for arg_node in node.args:
                    if not isinstance(arg_node, torch.fx.Node):
                        continue

                    is_get_attr = arg_node.op == "get_attr"
                    is_param_buffer = arg_node.op == "placeholder" and (
                        is_param(edge_exported_program, arg_node)
                        or is_buffer(edge_exported_program, arg_node)
                        or is_lifted_tensor_constant(edge_exported_program, arg_node)
                    )
                    if is_get_attr or is_param_buffer:
                        arg_node.meta["delegation_tag"] = delegation_tag
                    # Add to the list of partitioned nodes.

        return PartitionResult(
            tagged_exported_program=edge_exported_program, partition_tags=partition_tags
        )


ops_not_to_decompose = [
    torch.ops.aten.linear.default,
    torch.ops.aten.scaled_dot_product_attention.default,
]

edge_ops_non_decomposed = [
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.aten.scaled_dot_product_attention.default,
]


class OpsToNotDecomposeOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in edge_ops_non_decomposed


@final
class NonDecompTestPartitioner(Partitioner):
    """
    Partitions all add/mul nodes regardless of order
    """

    def __init__(self) -> None:
        self.op_support = any_chain(OpsToNotDecomposeOperatorSupport())
        self.delegation_spec = DelegationSpec(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([4]))],
        )

    def ops_to_not_decompose(
        self,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        def filter_ops(node: torch.fx.Node) -> bool:
            if node.op == "call_function" and node.target in ops_not_to_decompose:
                if len(node.args) == 3:
                    # This means that linear has a bias which is the only linear we support in this
                    # demo partitioner.
                    return True
                else:
                    return False

            return True

        return (ops_not_to_decompose, filter_ops)

    def _partition_graph_module(
        self,
        graph_module: torch.fx.GraphModule,
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        for _, submodule, _ in get_control_flow_submodules(graph_module):
            ret_partition_tags = self._partition_graph_module(submodule)
            partition_tags.update(ret_partition_tags)
        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
