# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import (
    get_non_lowered_nodes,
    tag_constant_data,
    tag_mutated_buffer,
)
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


@experimental(
    "This API and all of cuda backend related functionality are experimental."
)
class AotiPartitioner(Partitioner):
    """
    Base partitioner for AOTInductor-driven backend integration.

    Delegates the non-lowered operators to AOTInductor as one or more convex
    partitions (a single partition when nothing else has claimed part of the
    graph). It skips core ATen decomposition, letting the backend decompose via
    AOTInductor's backend-specific decomposition table.
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
        """Delegate the non-lowered ops to AOTInductor.

        Uses CapabilityBasedPartitioner rather than a single tag because a
        delegated submodule must be convex: if a node that is not delegated sits
        between the delegated ops, one tag would span a non-convex set and fusion
        would fail with a dependency cycle.
        """
        # Only nodes not already lowered are candidates for this backend.
        non_lowered_nodes = set(get_non_lowered_nodes(exported_program.graph))

        control_flow_targets = [
            torch.ops.higher_order.cond,
            torch.ops.higher_order.map_impl,
            torch.ops.higher_order.while_loop,
            torch.ops.higher_order.scan,
        ]

        class AotiOperatorSupport(OperatorSupportBase):
            def is_node_supported(
                self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
            ) -> bool:
                return node.op == "call_function" and node in non_lowered_nodes

        partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            AotiOperatorSupport(),
            allows_single_node_partition=True,
        )

        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitioner.propose_partitions():
            tag = f"aoti_{partition.id}"
            partition_tags[tag] = self.delegation_spec
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag

        # A control-flow op carries its branch GraphModules as get_attr operands;
        # they must share the op's tag so they land inside the same submodule. A
        # branch module feeds a single control-flow op, so first match wins.
        for node in exported_program.graph.nodes:
            if node.op != "get_attr":
                continue
            for user in node.users:
                if (
                    user.op == "call_function"
                    and user.target in control_flow_targets
                    and "delegation_tag" in user.meta
                ):
                    node.meta["delegation_tag"] = user.meta["delegation_tag"]
                    break

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        # tag_constant_data only tags constants that have users; tag the
        # genuinely unused ones too so none are left dangling.
        if partition_tags:
            fallback_tag = next(iter(partition_tags))
            for node in exported_program.graph.nodes:
                if (
                    node.op == "placeholder"
                    and not node.users
                    and "delegation_tag" not in node.meta
                    and (
                        is_param(exported_program, node)
                        or is_buffer(exported_program, node)
                        or is_lifted_tensor_constant(exported_program, node)
                    )
                ):
                    node.meta["delegation_tag"] = fallback_tag

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
