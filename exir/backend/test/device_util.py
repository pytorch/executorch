# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared device-aware test partitioners for ExecuTorch backend tests.

Provides ``DeviceAwarePartitioner`` (delegates add ops to a configurable
target device) and ``CpuOnlyPartitioner`` (delegates add ops without any
device annotation).  Both use ``AddOperatorSupport`` to select
``aten.add.Tensor`` nodes for delegation via ``BackendWithCompilerDemo``.
"""

from typing import Dict, final

import torch
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class AddOperatorSupport(OperatorSupportBase):
    """Marks ``aten.add.Tensor`` nodes as supported for delegation."""

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
        ]


@final
class DeviceAwarePartitioner(Partitioner):
    """Partitions add ops for delegation with a ``target_device`` CompileSpec.

    The ``target_device`` string (e.g. ``"cuda:0"``) is encoded into the
    delegation compile specs so that ``PropagateDevicePass`` can later
    annotate tensor specs with the correct device information.
    """

    def __init__(self, target_device: str = "cuda:0") -> None:
        super().__init__()
        self.op_support = any_chain(AddOperatorSupport())
        self.delegation_spec = DelegationSpec(
            BackendWithCompilerDemo.__name__,
            [
                CompileSpec("max_value", bytes([4])),
                CompileSpec(
                    TARGET_DEVICE_COMPILE_SPEC_KEY,
                    target_device.encode("utf-8"),
                ),
            ],
        )

    def partition(self, exported_program) -> PartitionResult:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )


@final
class CpuOnlyPartitioner(Partitioner):
    """Partitions add ops for delegation *without* a ``target_device`` spec.

    Useful as a control: since no device annotation is present, the
    ``PropagateDevicePass`` should leave all tensor specs on CPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.op_support = any_chain(AddOperatorSupport())
        self.delegation_spec = DelegationSpec(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([4]))],
        )

    def partition(self, exported_program) -> PartitionResult:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
