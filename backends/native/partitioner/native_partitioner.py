# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Native Backend Partitioner

The native partitioner marks ALL nodes as supported. The NativeBackend
runtime has a CPU fallback that can execute any portable op; runtime
partitioning across accelerators (Metal, etc.) happens in C++ based on
has_op() queries against the per-runtime op registries.
"""

from typing import Any, Callable, Dict, final, List, Mapping, Optional, Tuple

import torch

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer

from torch.export.exported_program import ExportedProgram
from torch.fx import Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


# Canonical list of ops the native backend preserves (does NOT decompose)
# during edge lowering when used with `to_edge_transform_and_lower`. Part
# of the universal-IR specification of NativeBackend; not user-configurable
# per call.
#
# Add an op here when you ship a dedicated C++ handler for it on the Metal
# (or other accelerator) side; otherwise the default decomposition pass at
# edge lowering will break it apart and we'll just dispatch the pieces.
_DEFAULT_PRESERVED_OPS = [
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.baddbmm.default,
    torch.ops.aten.scaled_dot_product_attention.default,
]


class NativeSupportedOperators(OperatorSupportBase):
    """
    Operator support checker for the Native backend.

    NativeBackend has a CPU fallback, so it supports ALL operators. The
    actual runtime partitioning across accelerators happens in C++.
    """

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        # Skip placeholder and output nodes - they shouldn't be partitioned
        if node.op in ("placeholder", "output", "get_attr"):
            return False

        # NativeBackend supports all call_function ops via CPU fallback
        return node.op == "call_function"


@final
class NativePartitioner(Partitioner):
    """
    Partitioner for the Native Backend.

    Unlike accelerator-specific partitioners that only claim ops they can
    accelerate, the native partitioner claims ALL ops since:
    1. CPU fallback can execute any portable op.
    2. Runtime partitioning in C++ handles dispatch to accelerators.

    Two usage paths:

    A) Classic to_edge + to_backend (default decomposition runs first):
        edge_program = to_edge(exported_program)
        edge_program = edge_program.to_backend(NativePartitioner())

    B) to_edge_transform_and_lower (preserves listed ops from decomposition):
        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[NativePartitioner()],
        )
        # The ops in _DEFAULT_PRESERVED_OPS are kept intact (no decomposition)
        # so accelerator kernels see them whole. The list is maintained
        # by this partitioner — extend `_DEFAULT_PRESERVED_OPS` (in code) when
        # you ship a new dedicated handler. Per-call overrides are NOT
        # exposed: NativeBackend is a universal IR and the preserve list
        # is part of its specification.
    """

    def __init__(
        self,
        compile_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.options = compile_options or {}
        compile_spec = self._parse_compile_options(self.options)
        # Import here to avoid circular dependency
        from executorch.backends.native.preprocess import NativeBackend

        self.delegation_spec = DelegationSpec(NativeBackend.__name__, compile_spec)

    def _parse_compile_options(self, options: Dict[str, Any]) -> List[CompileSpec]:
        """Convert compile options dict to CompileSpec list."""
        compile_specs = []

        for key, value in options.items():
            if isinstance(value, bool):
                value_bytes = value.to_bytes(1, byteorder="little")
                compile_specs.append(CompileSpec(key, value_bytes))
            elif isinstance(value, int):
                value_bytes = value.to_bytes(4, byteorder="little")
                compile_specs.append(CompileSpec(key, value_bytes))

        return compile_specs

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[Node], bool]]]:
        """
        Return ops that should NOT be decomposed during edge lowering.

        Called by `to_edge_transform_and_lower` BEFORE partitioning. The ops
        returned here are kept whole (skipped by the default decomposition
        pass), so backend kernels see them in their original form.

        The preserve list (`_DEFAULT_PRESERVED_OPS`) is the canonical list
        maintained by NativeBackend — part of the universal-IR specification
        and not user-configurable. Extend the constant in this file when
        you ship a new dedicated handler.

        Behavior:
          - If the graph is already partitioned (contains lowered_module
            get_attr nodes), return empty — partitioning has run, decomposition
            decisions are settled.
          - Otherwise return the preserve list, intersected with ops that
            actually appear in `ep` (no point listing ops the graph doesn't
            have).

        The second tuple element (filter callable) is None: the rule applies
        uniformly to every node whose target is in the preserve list.
        """
        # Already-partitioned graph -> nothing to preserve.
        for node in ep.graph.nodes:
            if node.op == "get_attr" and "lowered_module" in node.name:
                return ([], None)

        # Intersect preserve list with ops actually present in the graph.
        present: List[torch._ops.OpOverload] = []
        seen = set()
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if node.target in _DEFAULT_PRESERVED_OPS and node.target not in seen:
                present.append(node.target)
                seen.add(node.target)

        return (present, None)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Partition the exported program for the native backend.

        Since native supports everything, this partitions the entire graph
        into a single delegation block.
        """
        partition_tags = {}

        # Use CapabilityBasedPartitioner with our "support everything" checker
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NativeSupportedOperators(),
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()

        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        # Tag constant data for proper handling
        tag_constant_data(exported_program)
        # Tag mutated buffer placeholders so they're owned by our delegate
        # (KV-cache style state flows through the delegated subgraph; the
        # writeback copy_ collapses out of the top-level chain).
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
