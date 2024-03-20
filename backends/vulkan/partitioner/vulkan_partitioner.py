# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import final, List, Optional

import torch
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase


class VulkanSupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            # Binary arithmetic operators
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.div.Tensor_mode,
            exir_ops.edge.aten.pow.Tensor_Tensor,
            # Activation operators
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.relu.default,
            # Matrix multiplication operators
            exir_ops.edge.aten.mm.default,
            # Other
            operator.getitem,
        ]
        return supported


@final
class VulkanPartitioner(Partitioner):
    def __init__(self, compile_spec: Optional[List[CompileSpec]] = None) -> None:
        if compile_spec is None:
            compile_spec = []
        self.delegation_spec = DelegationSpec(VulkanBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            VulkanSupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
