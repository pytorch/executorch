# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import logging
from typing import List

import torch
from torch._export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.backends.apple.coreml.compiler.coreml_preprocess import CoreMLBackend

from coremltools.converters.mil.frontend.torch.torch_op_registry import is_torch_fx_node_supported

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class OperatorsSupportedForCoreMLBackend(OperatorSupportBase):
    def __init__(self, skip_ops: List[str] = []) -> None:
        super().__init__()
        self.skip_ops = skip_ops

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # get_attr node can always be supported on any backend
        if node.op == "get_attr":
            return True
        # check if the PyTorch op get called is supported in Core ML
        elif node.op == "call_function":
            return is_torch_fx_node_supported(node, self.skip_ops)
        # cowardly refuse to support all other types of node
        else:
            return False


class CoreMLPartitioner(Partitioner):
    compile_spec = []

    def __init__(self, skip_ops: List[str] = []) -> None:
        self.skip_ops = skip_ops
        self.delegation_spec = DelegationSpec("CoreMLBackend", self.compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logger.info("CoreMLPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            OperatorsSupportedForCoreMLBackend(self.skip_ops),
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
