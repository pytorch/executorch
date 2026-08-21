# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import torch
from executorch.exir import memory
from executorch.exir.delegate import (
    executorch_call_delegate,
    NUM_DELEGATE_SCRATCH_ARGS_KEY,
)
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.memory_planning import get_node_tensor_specs
from executorch.exir.passes.propagate_device_pass import _get_lowered_module
from executorch.exir.tensor import TensorSpec
from torch.fx.passes.infra.pass_base import PassBase, PassResult


def _delegate_calls(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is executorch_call_delegate
    ]


def strip_delegate_scratch_pass(graph_module: torch.fx.GraphModule) -> PassResult:
    """Drops scratch allocations left over from a previous ``to_executorch()``.

    ``to_executorch()`` writes its lowered graph back into the edge program, so
    a second call would otherwise retrace a delegate call whose extra arguments
    the lowered module's signature does not have.
    """
    modified = False
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        stripped = False
        for node in _delegate_calls(module):
            num_scratch = node.meta.pop(NUM_DELEGATE_SCRATCH_ARGS_KEY, 0)
            if num_scratch == 0:
                continue
            keep = len(node.args) - num_scratch
            scratch_nodes = node.args[keep:]
            node.args = tuple(node.args[:keep])
            for scratch in scratch_nodes:
                module.graph.erase_node(scratch)
            stripped = True
        if stripped:
            module.recompile()
            modified = True
    return PassResult(graph_module, modified)


class InsertDelegateScratchPass(PassBase):
    """Materializes the scratch buffers a backend asked for in ``preprocess``.

    Each ``DelegateScratchSpec`` becomes a ``memory.alloc`` node appended to the
    delegate call's arguments, which the memory planner then allocates in the
    arena like any other intermediate. The count is recorded on the delegate
    node so the emitter can place them last, after the delegate's outputs, and
    so the inputs and outputs keep the positions the backend already expects.

    This runs after the last interpreter-based pass: retracing a delegate call
    with extra arguments fails against the lowered module's original signature.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if isinstance(module, torch.fx.GraphModule):
                modified |= self._insert_in_graph(module)
        return PassResult(graph_module, modified)

    def _insert_in_graph(self, graph_module: torch.fx.GraphModule) -> bool:
        modified = False
        for node in _delegate_calls(graph_module):
            lowered_module = _get_lowered_module(graph_module, node)
            if lowered_module is None:
                raise RuntimeError(
                    f"Delegate call {node.name} does not reference a LoweredBackendModule."
                )
            if not lowered_module.scratch_specs:
                continue
            node.args = tuple(node.args) + tuple(
                self._make_scratch_nodes(graph_module, node, lowered_module)
            )
            node.meta[NUM_DELEGATE_SCRATCH_ARGS_KEY] = len(lowered_module.scratch_specs)
            modified = True

        if modified:
            graph_module.recompile()
        return modified

    def _make_scratch_nodes(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        lowered_module: LoweredBackendModule,
    ) -> List[torch.fx.Node]:
        delegate_specs = get_node_tensor_specs(node)
        scratch_nodes = []
        for scratch_spec in lowered_module.scratch_specs:
            with graph_module.graph.inserting_before(node):
                alloc = graph_module.graph.call_function(
                    memory.alloc, (((scratch_spec.nbytes,), torch.uint8),)
                )

            spec = TensorSpec(
                dtype=torch.uint8, shape=torch.Size([scratch_spec.nbytes])
            )
            if scratch_spec.mem_id is not None:
                spec.mem_id = scratch_spec.mem_id
            if delegate_specs:
                # PropagateDevicePass has already run, so take the delegate's
                # device rather than defaulting the scratch to CPU.
                spec.device = delegate_specs[0].device
                spec.device_index = delegate_specs[0].device_index
            alloc.meta["spec"] = spec
            scratch_nodes.append(alloc)
        return scratch_nodes
