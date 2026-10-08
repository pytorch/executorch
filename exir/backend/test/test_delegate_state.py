# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, final, List

import torch
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    DelegateScratchSpec,
    DelegateStateSpec,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.memory_planning import get_node_tensor_specs
from executorch.exir.passes.delegate_scratch_pass import DelegateScratchSpecPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.program._program import _transform
from executorch.exir.tensor import TensorSpec
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase

# Not all zero, so that an image dropped on the way to the program would not
# be mistaken for a zero-filled buffer.
INITIAL_STATE = bytes(range(1, 49))
ZERO_FILLED_STATE_BYTES = 100
SCRATCH_BYTES = 64


@final
class DelegateStateTestBackend(BackendDetails):
    """Declares one state buffer with an initial image and one without."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(
            processed_bytes=b"state-backend-blob",
            state_specs=[
                DelegateStateSpec(
                    nbytes=len(INITIAL_STATE),
                    label="history",
                    initial_data=INITIAL_STATE,
                ),
                DelegateStateSpec(nbytes=ZERO_FILLED_STATE_BYTES, label="counters"),
            ],
        )


@final
class StateAndScratchTestBackend(BackendDetails):
    """Declares both kinds, so that each can be checked to stay in its own list."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(
            processed_bytes=b"state-and-scratch-backend-blob",
            scratch_specs=[DelegateScratchSpec(nbytes=SCRATCH_BYTES)],
            state_specs=[DelegateStateSpec(nbytes=ZERO_FILLED_STATE_BYTES)],
        )


class AddSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function" and node.target is exir_ops.edge.aten.add.Tensor
        )


class OneDelegatePerAddPartitioner(Partitioner):
    """Tags every add node separately, so each becomes its own delegate call."""

    def __init__(
        self, backend_id: str, compile_specs: List[CompileSpec] | None = None
    ) -> None:
        self.delegation_spec = DelegationSpec(backend_id, compile_specs or [])
        self._next_tag = 0

    def _tag_graph(
        self, graph_module: torch.fx.GraphModule
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        support = AddSupport()
        for node in graph_module.graph.nodes:
            if not support.is_node_supported({}, node):
                continue
            tag = f"tag{self._next_tag}"
            self._next_tag += 1
            node.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec
        for _, submodule, _ in get_control_flow_submodules(graph_module):
            partition_tags.update(self._tag_graph(submodule))
        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=self._tag_graph(exported_program.graph_module),
        )


class SingleDelegateModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.sin(x + y)


class ChainedDelegateModule(torch.nn.Module):
    """The first delegate's output is consumed before the second delegate runs.

    Their calls are therefore disjoint, so scratch could share bytes between
    them, and state must not.
    """

    def forward(self, x, y):
        return torch.sin(torch.sin(x + y) + y)


class CondModule(torch.nn.Module):
    def forward(self, pred, x, y):
        return torch.cond(pred, lambda a, b: a + b, lambda a, b: a * b, (x, y))


def _edge(module: torch.nn.Module, inputs, backend_id: str, compile_specs=None):
    exported = torch.export.export(module.eval(), inputs)
    return to_edge_transform_and_lower(
        exported,
        partitioner=[OneDelegatePerAddPartitioner(backend_id, compile_specs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def _lower(module: torch.nn.Module, inputs, backend_id: str, config=None):
    return _edge(module, inputs, backend_id).to_executorch(
        config or ExecutorchBackendConfig()
    )


def _single_delegate_call(lowered) -> torch.fx.Node:
    (call,) = [
        node
        for node in lowered.exported_program().graph.nodes
        if node.target is executorch_call_delegate
    ]
    return call


def _overlaps(lhs: TensorSpec, rhs: TensorSpec) -> bool:
    return (
        lhs.mem_id == rhs.mem_id
        and lhs.mem_offset < rhs.mem_offset + rhs.allocated_memory
        and rhs.mem_offset < lhs.mem_offset + lhs.allocated_memory
    )


def _planned_specs(graph_module: torch.fx.GraphModule) -> List[TensorSpec]:
    specs = []
    for node in graph_module.graph.nodes:
        for spec in get_node_tensor_specs(node):
            if spec.mem_id is not None and spec not in specs:
                specs.append(spec)
    return specs


SINGLE_INPUTS = (torch.randn(3, 4), torch.randn(3, 4))


class TestDelegateState(unittest.TestCase):
    def test_state_lives_for_the_whole_graph(self):
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        graph = lowered.exported_program().graph
        last = len(graph.nodes) - 1

        for spec in memory.delegate_state_specs(_single_delegate_call(lowered)):
            self.assertEqual(spec.lifetime, [0, last])

    def test_state_shares_bytes_with_nothing_else_in_the_plan(self):
        # Anything sharing them would overwrite the state between executions.
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        state = memory.delegate_state_specs(_single_delegate_call(lowered))
        others = [
            spec
            for spec in _planned_specs(lowered.exported_program().graph_module)
            if spec not in state
        ]
        self.assertTrue(others)

        for spec in state:
            for other in others:
                self.assertFalse(_overlaps(spec, other), f"{spec} overlaps {other}")
        self.assertFalse(_overlaps(state[0], state[1]))

    def test_state_of_disjoint_delegate_calls_does_not_share_bytes(self):
        # Scratch of these two calls may share bytes; state, which outlives
        # every execution, may not.
        lowered = _lower(
            ChainedDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        calls = [
            node
            for node in lowered.exported_program().graph.nodes
            if node.target is executorch_call_delegate
        ]
        first, second = (memory.delegate_state_specs(call) for call in calls)

        for lhs in first:
            for rhs in second:
                self.assertFalse(_overlaps(lhs, rhs))

    def test_labels_reach_the_planner(self):
        edge = _edge(SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend")
        program = _transform(
            edge.exported_program(), SpecPropPass(), DelegateScratchSpecPass()
        )
        call = [
            node
            for node in program.graph.nodes
            if node.target is executorch_call_delegate
        ][0]
        self.assertEqual(
            [state.label for state in memory.delegate_state(call)],
            ["history", "counters"],
        )

    def test_state_inside_control_flow_is_rejected(self):
        inputs = (torch.tensor(True), torch.randn(3, 4), torch.randn(3, 4))
        edge = _edge(CondModule(), inputs, "DelegateStateTestBackend")

        with self.assertRaisesRegex(RuntimeError, "control-flow submodule"):
            edge.to_executorch(ExecutorchBackendConfig())

    def test_a_stateful_module_called_twice_is_rejected(self):
        # Each call site would get its own state where the eager model shares
        # one buffer.
        edge = _edge(SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend")
        graph_module = edge.exported_program().graph_module
        call = [
            node
            for node in graph_module.graph.nodes
            if node.target is executorch_call_delegate
        ][0]
        with graph_module.graph.inserting_after(call):
            second = graph_module.graph.call_function(
                executorch_call_delegate, call.args, call.kwargs
            )
            second.meta = dict(call.meta)
        graph_module.recompile()

        with self.assertRaisesRegex(RuntimeError, "called from 2 places"):
            DelegateScratchSpecPass()(graph_module)

    def test_state_on_a_device_delegate_is_rejected(self):
        edge = _edge(
            SingleDelegateModule(),
            SINGLE_INPUTS,
            "DelegateStateTestBackend",
            compile_specs=[CompileSpec("target_device", b"cuda:0")],
        )

        with self.assertRaisesRegex(RuntimeError, "only supported on the CPU"):
            edge.to_executorch(ExecutorchBackendConfig())

    def test_state_with_shared_mutable_buffers_is_rejected(self):
        config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True)
        )

        with self.assertRaisesRegex(RuntimeError, "share_mutable_buffers"):
            _lower(
                SingleDelegateModule(),
                SINGLE_INPUTS,
                "DelegateStateTestBackend",
                config=config,
            )

    def test_unusable_requests_are_rejected_at_declaration(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DelegateStateSpec(nbytes=0)
        with self.assertRaisesRegex(ValueError, "initial data is 3 bytes"):
            DelegateStateSpec(nbytes=4, initial_data=b"abc")


if __name__ == "__main__":
    unittest.main()
