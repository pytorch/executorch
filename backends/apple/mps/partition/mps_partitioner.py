#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import logging

import torch
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)

from torch._export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class OperatorsSupportedForMpsBackend(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported_mps_ops = [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mm.default,
            torch.ops.aten.div.default,
        ]
        ret_val = (
            (node.op == "call_function" and node.target in supported_mps_ops)
            or node.op == "get_attr"
            or node.op == "output"
        )
        return ret_val


# TODO MPSPartitioner is work in progress currently.
# Use whole graph delegation instead when lowering to MPS.
class MPSPartitioner(Partitioner):
    compile_spec = []

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec("MPSBackend", self.compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("MpsPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            OperatorsSupportedForMpsBackend(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
