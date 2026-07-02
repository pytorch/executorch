# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Native backend partitioner.

Claims core ATen ops (torch.Tag.core) plus an explicit opt-in set.
"""

import json

from typing import Callable, final, List, Mapping, Optional, Tuple

import executorch.backends.native.specializations  # noqa: F401 — register recipes

import torch

from executorch.exir.backend.backend_details import CompileSpec
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

# Non-core ops the native backend supports. Also preserved (not decomposed).
_SUPPORTED_NON_CORE_OPS = [
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.scaled_dot_product_attention.default,
]


class NativeSupportedOperators(OperatorSupportBase):
    _NON_CORE = set(_SUPPORTED_NON_CORE_OPS)

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op in ("placeholder", "output", "get_attr"):
            return False
        if node.op != "call_function":
            return False
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            return False

        from executorch.exir.dialects.edge._ops import EdgeOpOverload

        target = node.target
        if isinstance(target, EdgeOpOverload):
            target = target._op
        if isinstance(target, torch._ops.OpOverload):
            if target in self._NON_CORE:
                return True
            return torch.Tag.core in target.tags or torch.Tag.view_copy in target.tags
        return False


@final
class NativePartitioner(Partitioner):
    _SPECIALIZATIONS_KEY = "native_specializations"

    def __init__(
        self,
        specializations: Optional[List[str]] = None,
    ) -> None:
        self.specializations = specializations
        from executorch.backends.native.preprocess import NativeBackend

        compile_specs = []
        if specializations:
            compile_specs.append(
                CompileSpec(
                    self._SPECIALIZATIONS_KEY,
                    json.dumps(specializations).encode("utf-8"),
                )
            )
        self.delegation_spec = DelegationSpec(NativeBackend.__name__, compile_specs)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[Node], bool]]]:
        # Already-partitioned graph -> nothing to preserve.
        from executorch.exir.lowered_backend_module import executorch_call_delegate

        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is executorch_call_delegate:
                return ([], None)

        present: List[torch._ops.OpOverload] = []
        seen = set()
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if node.target in _SUPPORTED_NON_CORE_OPS and node.target not in seen:
                present.append(node.target)
                seen.add(node.target)

        return (present, None)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = {}

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

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
