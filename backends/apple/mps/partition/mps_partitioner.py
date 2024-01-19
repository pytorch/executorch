#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging
from typing import Any, Dict, List, Union

import torch
from executorch.backends.apple.mps.mps_preprocess import MPSBackend
from executorch.backends.apple.mps.operators.node_visitor import get_node_visitors
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch._export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase


class MPSOperatorSupport(OperatorSupportBase):
    def __init__(self, edge_program: torch.export.ExportedProgram, compiler_specs):
        self.node_visitors = get_node_visitors(edge_program)

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target.__name__ not in self.node_visitors:
            return False

        return True


class MPSPartitioner(Partitioner):
    compile_spec: List[CompileSpec] = []

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(MPSBackend.__name__, self.compile_spec)
        self.partition_tags: Dict[str, DelegationSpec] = {}

    def generate_partitions(self, edge_program: ExportedProgram) -> List[Any]:
        self.supported_ops = MPSOperatorSupport(
            edge_program=edge_program, compiler_specs=self.delegation_spec.compile_specs
        )
        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.supported_ops,
        )

    def tag_nodes(self, partitions: List[Partition]) -> None:
        for partition in partitions:
            for node in partition.nodes:
                delegation_tag = f"mps_{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

    @staticmethod
    def check_partitions(partitions: Union[dict, list]) -> bool:
        pl = len(partitions)
        if pl == 0:
            logging.warning("Nothing can be partitioned!")
        else:
            logging.info(f"Found {pl} subgraphs to be partitioned.")
        return pl != 0

    # override
    def partition(self, edge_program: ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program=edge_program)
        if self.check_partitions(partitions):
            self.tag_nodes(partitions)
        x = PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

        return x
