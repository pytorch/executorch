# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

import torch
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
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

        self.partition_tags = {}

    def partition(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        partition_list = generate_pattern_op_partitions(
            edge_graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

        for _, submodule, _ in get_control_flow_submodules(edge_graph_module):
            self.partition(submodule)

        return edge_graph_module


@final
class AddAttributePartitionerDemo(Partitioner):
    """
    Partitions all add and get_attr nodes
    """

    def __init__(self) -> None:
        self.op_support = AddOperatorSupport()

        self.delegation_spec = DelegationSpec(BackendWithCompilerDemo.__name__, [])
        self.partition_tags = {}

    def partition(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        partition_list = generate_pattern_op_partitions(
            edge_graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                self.partition_tags[delegation_tag] = self.delegation_spec

                # Tag the add nodes
                node.meta["delegation_tag"] = delegation_tag

                # Tag the get_attr nodes that are arguments to the add nodes.
                # This will add the attributes into the lowered submodule.
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                        arg.meta["delegation_tag"] = delegation_tag

        return edge_graph_module
