# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, final, List

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend  # usort: skip
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


class AOTISupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        supported = node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        ]

        return supported

    def is_node_supported_custom(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.edge.aten.mean.dim:
            keep_dim = node.args[2] if len(node.args) > 2 else False
            return cast(bool, keep_dim)
        if node.target == exir_ops.edge.aten.var.correction:
            keep_dim = node.kwargs.get("keepdim", False)
            return cast(bool, keep_dim)
        return True


@final
class AotiPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(AotiBackend.__name__, compile_spec)
        print(self.delegation_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        # logger.info("AotiPartitioner::partition")
        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            AOTISupportedOperators(),
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
