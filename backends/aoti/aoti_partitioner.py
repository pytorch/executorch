# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export.exported_program import ExportedProgram


@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class AotiPartitioner(Partitioner):
    """
    Base partitioner for AOTInductor-driven backend integration.

    This partitioner creates a single partition containing all operators from the input graph.
    It skips core ATen decomposition, allowing the backend to handle decomposition using
    AOTInductor's backend-specific decomposition table.

    Only operators that cannot be handled by the aoti library will be excluded from
    the partition and fall back to ExecuTorch's default or custom handling.
    """

    def __init__(self, backend_name: str, compile_spec: List[CompileSpec]) -> None:
        """
        Initialize the AOTI partitioner.

        Args:
            backend_name: The name of the backend (e.g., "CudaBackend", "MetalBackend")
            compile_spec: List of compilation specifications
        """
        self.delegation_spec = DelegationSpec(backend_name, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Fully delegate the graph to AOTInductor by tagging all nodes as a single partition.
        """

        partition_tags: Dict[str, DelegationSpec] = {}
        tag = "tag0"

        # Tag torch.cond and other control flow operations
        def is_control_flow(node: torch.fx.Node) -> bool:
            return node.op == "call_function" and node.target in [
                torch.ops.higher_order.cond,
                torch.ops.higher_order.map_impl,
                torch.ops.higher_order.while_loop,
            ]

        for node in exported_program.graph.nodes:
            if node.op == "call_function":
                node.meta["delegation_tag"] = tag
            # Tag get_attr nodes that are used by control flow operations
            elif node.op == "get_attr":
                # Check if any user is a control flow operation
                for user in node.users:
                    if is_control_flow(user):
                        node.meta["delegation_tag"] = tag
                        break

        partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        # Tag constant placeholders that have no users
        # tag_constant_data only tags constants that have users with delegation_tag
        # but we need to tag all constants for this partition
        for node in exported_program.graph.nodes:
            if node.op == "placeholder" and (
                is_param(exported_program, node)
                or is_buffer(exported_program, node)
                or is_lifted_tensor_constant(exported_program, node)
            ):
                if "delegation_tag" not in node.meta:
                    node.meta["delegation_tag"] = tag

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Return a list of operations that should not be decomposed and let the AOT compiler handle them.
        Currently we skip ATen decompositon for all ops, and let the backend handle them.
        """
        do_not_decompose = set()

        for node in ep.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                do_not_decompose.add(node.target)
        return list(do_not_decompose), None
