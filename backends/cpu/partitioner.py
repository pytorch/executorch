# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Mapping

import torch
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.utils._pytree import tree_leaves


# Serialized as a little-endian uint32. Bump for incompatible artifact contracts.
CPU_DELEGATE_VERSION = 1


class CPUSemanticOperators(OperatorSupportBase):
    """Pure tensor operations representable by the initial static FP32
    delegate version.

    Runtime providers determine implementation coverage. An unresolved operation
    is reported by name when the CPU runtime prepares the graph.
    Nodes with missing value metadata or non-tensor outputs are unsupported.
    """

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op != "call_function":
            return False
        if node.target is operator.getitem:
            if len(node.args) != 2:
                return False
            producer = node.args[0]
            if not isinstance(producer, torch.fx.Node) or not self.is_node_supported(
                submodules, producer
            ):
                return False
        else:
            target = node.target
            if isinstance(target, EdgeOpOverload):
                target = target._op
            if not isinstance(target, torch._ops.OpOverload):
                return False
            if target._schema.is_mutable or any(
                result.alias_info is not None for result in target._schema.returns
            ):
                return False
        outputs = tree_leaves(node.meta.get("val"))
        if not outputs or any(not isinstance(value, torch.Tensor) for value in outputs):
            return False
        related_nodes = [node, *node.all_input_nodes]
        if any(related_node.meta.get("val") is None for related_node in related_nodes):
            return False
        return all(
            value.dtype == torch.float32
            and all(isinstance(size, int) for size in value.shape)
            and value.is_contiguous()
            for related_node in related_nodes
            for value in tree_leaves(related_node.meta["val"])
            if isinstance(value, torch.Tensor)
        )


class CPUPartitioner(Partitioner):
    """Retain the static FP32 semantic graph for runtime provider selection."""

    def ops_to_not_decompose(self, ep: torch.export.ExportedProgram):
        support = CPUSemanticOperators()
        return [torch.ops.aten.linear.default], lambda node: support.is_node_supported(
            {}, node
        )

    def partition(
        self, exported_program: torch.export.ExportedProgram
    ) -> PartitionResult:
        from executorch.backends.cpu.preprocess import CpuBackend

        partitions = CapabilityBasedPartitioner(
            exported_program.graph_module,
            CPUSemanticOperators(),
            allows_single_node_partition=True,
        ).propose_partitions()
        tags = {}
        for partition in partitions:
            tag = f"cpu_{partition.id}"
            tags[tag] = DelegationSpec(
                CpuBackend.__name__,
                [
                    CompileSpec(
                        "cpu_delegate_version",
                        CPU_DELEGATE_VERSION.to_bytes(4, "little"),
                    )
                ],
            )
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
        tag_constant_data(exported_program)
        return PartitionResult(exported_program, tags)
