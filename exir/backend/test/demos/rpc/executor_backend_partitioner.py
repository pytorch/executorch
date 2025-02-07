# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import final

import torch
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class AnyOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function"


class AnyDelegateSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op == "call_method":
            assert isinstance(
                node.args[0], torch.fx.Node
            ), "the first argument is not an fx Node, it's not a valid graph with delgates"
            lowered_name = typing.cast(torch.fx.Node, node.args[0]).name
            lowered_module = submodules[lowered_name]
            return lowered_module.backend_id is BackendWithCompilerDemo.__name__
        return False


@final
class ExecutorBackendPartitioner(Partitioner):
    """
    Partitions all add/mul nodes regardless of order
    """

    def __init__(self) -> None:
        self.op_support = any_chain(AnyOperatorSupport(), AnyDelegateSupport())
        self.delegation_spec = DelegationSpec(ExecutorBackend.__name__, [])

    def partition(self, edge_exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = {}
        partition_list = generate_pattern_op_partitions(
            edge_exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

                # Tag the delegate submodules
                if node.args[0].op == "get_attr":
                    node.args[0].meta["delegation_tag"] = delegation_tag

        return PartitionResult(
            tagged_exported_program=edge_exported_program,
            partition_tags=partition_tags,
        )
