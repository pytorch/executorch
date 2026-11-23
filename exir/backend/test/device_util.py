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
``BuffersOnDevicePlanningPass`` plans chosen buffers in device memory.
"""

from typing import Dict, final, Optional, Set

import torch
from executorch.exir import memory
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
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import (
    _copy_output_spec,
    _replace_spec,
    TARGET_DEVICE_COMPILE_SPEC_KEY,
)
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
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


class BuffersOnDevicePlanningPass(MemoryPlanningPass):
    """Plans the buffers named in ``buffer_fqns`` in ``cuda:device_index`` memory.

    Host ops then read such a buffer through an et_copy._d2h_copy, and its new
    value is written back with an et_copy._h2d_copy, so the methods run with any
    device allocator. With ``insert_copies=False`` host ops use the device
    memory directly, which export refuses.
    """

    def __init__(
        self,
        buffer_fqns: Set[str],
        device_index: int = 0,
        insert_copies: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.buffer_fqns = buffer_fqns
        self.device_index = device_index
        self.insert_copies = insert_copies

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ):
        assert graph_signature is not None
        for node in list(graph_module.graph.nodes):
            if (
                node.op == "placeholder"
                and graph_signature.inputs_to_buffers.get(node.target)
                in self.buffer_fqns
            ):
                if self.insert_copies:
                    _copy_across_device_boundary(graph_module, node)
                node.meta["spec"].device = DeviceType.CUDA
                node.meta["spec"].device_index = self.device_index
        graph_module.recompile()
        return super().run(graph_module, graph_signature)


def _copy_across_device_boundary(
    graph_module: torch.fx.GraphModule, buffer: torch.fx.Node
) -> None:
    """Makes the write-back of ``buffer`` an H2D copy and its readers read a D2H copy."""
    graph = graph_module.graph
    val = buffer.meta["val"]
    for user in list(buffer.users):
        # A delegate on the device, or the copy that stages it for one, reads
        # the buffer where it is.
        if user.target in (executorch_call_delegate, torch.ops.et_copy._h2d_copy.out):
            continue
        # A view aliases the buffer, so its own readers need the copies.
        if user.target == memory.view:
            _copy_across_device_boundary(graph_module, user)
            continue
        if user.target == torch.ops.aten.copy_.default and user.args[0] is buffer:
            with graph.inserting_before(user):
                write = graph.call_function(
                    torch.ops.et_copy._h2d_copy.out, (user.args[1],), {"out": buffer}
                )
            write.meta = user.meta.copy()
            user.replace_all_uses_with(write)
            graph.erase_node(user)
            continue
        with graph.inserting_before(user):
            host = graph.call_function(memory.alloc, ((tuple(val.shape), val.dtype),))
            read = graph.call_function(
                torch.ops.et_copy._d2h_copy.out, (buffer,), {"out": host}
            )
        # The copy owns its memory, so it cannot keep a view's link to the base.
        spec = (
            TensorSpec.from_tensor(val)
            if buffer.target == memory.view
            else buffer.meta["spec"]
        )
        host.meta["spec"] = _copy_output_spec(spec, DeviceType.CPU)
        host.meta["val"] = read.meta["val"] = val
        read.meta["spec"] = host.meta["spec"]
        user.replace_input_with(buffer, read)
        _replace_spec([user], buffer.meta["spec"], read.meta["spec"])
