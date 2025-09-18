# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import Callable, cast, Dict, final, List, Optional, Set, Tuple

import torch
from executorch.backends.apple.metal.metal_backend import MetalBackend  # usort: skip
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
        # supported = node.op == "call_function" and (
        #     node.target == operator.getitem
        #     or str(node.target._op) not in inductor_fallback_ops
        #     or str(node.target._op) in supported_fallback_operators
        # )

        supported = node.op == "call_function"

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
class MetalPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(MetalBackend.__name__, compile_spec)
        print(self.delegation_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        # logger.info("MetalPartitioner::partition")
        print("entering partitioner...")

        partition_tags = {}

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            AOTISupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        assert len(partition_list) == 1, "Graph break is not supported yet"

        print(f"graph breaks into {len(partition_list)} parts")

        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Return a list of operations that should not be decomposed and let the AOT compiler handle them.
        """
        do_not_decompose = set()
        op_support = AOTISupportedOperators()

        for node in ep.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(node.target, torch._ops.OpOverload)
                and op_support.is_node_supported(None, node)
            ):
                do_not_decompose.add(node.target)
        return list(do_not_decompose), None
