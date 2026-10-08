# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import unittest
from typing import Dict, final, List

import torch
from executorch import exir
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    memory,
    to_edge_transform_and_lower,
)
from executorch.exir._serialize._program import deserialize_pte_binary
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
from executorch.exir.backend.utils import tag_mutated_buffer, tag_owned_mutated_buffers
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.emit import emit_program
from executorch.exir.error import InternalError
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.memory_planning import get_node_tensor_specs
from executorch.exir.passes.delegate_scratch_pass import DelegateScratchSpecPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.print_program import print_program
from executorch.exir.program._program import _transform
from executorch.exir.schema import DelegateCall
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


def _delegate_calls(plan) -> List[DelegateCall]:
    return [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, DelegateCall)
    ]


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

    def test_emitted_state_matches_the_planner(self):
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        (delegate_call,) = _delegate_calls(lowered.executorch_program.execution_plan[0])
        specs = memory.delegate_state_specs(_single_delegate_call(lowered))

        self.assertEqual(
            [
                (s.allocation.memory_id, s.allocation.memory_offset, s.size)
                for s in delegate_call.state
            ],
            [(spec.mem_id, spec.mem_offset, spec.nbytes()) for spec in specs],
        )

    def test_initial_image_lands_in_the_mutable_data_segment(self):
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        pte = deserialize_pte_binary(lowered.buffer)
        (delegate_call,) = _delegate_calls(pte.program.execution_plan[0])
        initialised, zero_filled = delegate_call.state

        self.assertGreater(initialised.initial_data_idx, 0)
        self.assertEqual(
            pte.mutable_data[initialised.initial_data_idx].storage, INITIAL_STATE
        )
        # Zero-filling is the runtime's job, so no image is stored for it.
        self.assertEqual(zero_filled.initial_data_idx, 0)
        self.assertEqual(len(pte.mutable_data), 2)

    def test_scratch_and_state_stay_in_their_own_lists(self):
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "StateAndScratchTestBackend"
        )
        (delegate_call,) = _delegate_calls(lowered.executorch_program.execution_plan[0])
        self.assertEqual([s.size for s in delegate_call.scratch], [SCRATCH_BYTES])
        self.assertEqual(
            [s.size for s in delegate_call.state], [ZERO_FILLED_STATE_BYTES]
        )

        call = _single_delegate_call(lowered)
        (scratch,) = memory.delegate_scratch_specs(call)
        (state,) = memory.delegate_state_specs(call)
        self.assertLess(scratch.lifetime[1] - scratch.lifetime[0], state.lifetime[1])
        self.assertFalse(_overlaps(scratch, state))

    def test_state_survives_a_serde_round_trip(self):
        edge = _edge(SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend")
        buffer = io.BytesIO()
        exir.save(edge.exported_program(), buffer)
        buffer.seek(0)
        reloaded = EdgeProgramManager(exir.load(buffer))

        pte = deserialize_pte_binary(
            reloaded.to_executorch(ExecutorchBackendConfig()).buffer
        )
        (delegate_call,) = _delegate_calls(pte.program.execution_plan[0])
        initialised, zero_filled = delegate_call.state
        self.assertEqual(
            pte.mutable_data[initialised.initial_data_idx].storage, INITIAL_STATE
        )
        self.assertEqual(zero_filled.size, ZERO_FILLED_STATE_BYTES)

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

    def test_emitting_without_the_pass_is_an_error_not_a_silent_drop(self):
        edge = _edge(SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend")
        program = _transform(
            edge.exported_program(), SpecPropPass(), MemoryPlanningPass()
        )

        with self.assertRaisesRegex(InternalError, "declared state buffers"):
            emit_program(program)

    def test_print_program_shows_the_state(self):
        lowered = _lower(
            SingleDelegateModule(), SINGLE_INPUTS, "DelegateStateTestBackend"
        )
        out = io.StringIO()
        print_program(lowered.executorch_program, out=out)
        self.assertIn(f"[{len(INITIAL_STATE)}]", out.getvalue())
        self.assertIn("state=[", out.getvalue())

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


class ReadThenReplaceStateModule(torch.nn.Module):
    """``state`` is read by the add and replaced by the sin, which does not read it."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(3))

    def forward(self, x):
        out = x + self.state
        self.state.copy_(torch.sin(x))
        return out


class TwoReadersStateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(3))

    def forward(self, x):
        out = (x + self.state) * (x - self.state)
        self.state.copy_(torch.sin(x))
        return out


class TagOpsPartitioner(Partitioner):
    """Tags the named edge ops, then the buffers their partitions own outright."""

    def __init__(self, tags_by_op) -> None:
        self.tags_by_op = tags_by_op
        self.delegation_spec = DelegationSpec("DelegateStateTestBackend", [])

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        for node in exported_program.graph.nodes:
            if node.op == "call_function" and node.target in self.tags_by_op:
                node.meta["delegation_tag"] = self.tags_by_op[node.target]
        tag_owned_mutated_buffers(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags={
                tag: self.delegation_spec for tag in set(self.tags_by_op.values())
            },
        )


def _edge_program(module: torch.nn.Module) -> ExportedProgram:
    return exir.to_edge(
        torch.export.export(module, (torch.ones(3),), strict=True)
    ).exported_program()


def _state_placeholder(exported_program: ExportedProgram) -> torch.fx.Node:
    return next(
        node
        for node in exported_program.graph.nodes
        if exported_program.graph_signature.inputs_to_buffers.get(node.name) == "state"
    )


def _tag(exported_program: ExportedProgram, tags_by_op) -> None:
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and node.target in tags_by_op:
            node.meta["delegation_tag"] = tags_by_op[node.target]


ADD = exir_ops.edge.aten.add.Tensor
SUB = exir_ops.edge.aten.sub.Tensor
SIN = exir_ops.edge.aten.sin.default


class TestTagOwnedMutatedBuffers(unittest.TestCase):
    def test_a_buffer_whose_readers_and_writer_share_a_partition_is_tagged(self):
        program = _edge_program(ReadThenReplaceStateModule())
        _tag(program, {ADD: "p", SIN: "p"})

        tag_owned_mutated_buffers(program)

        self.assertEqual(_state_placeholder(program).meta.get("delegation_tag"), "p")

    def test_a_buffer_updated_on_the_cpu_is_left_to_the_graph(self):
        # Every reader is delegated, which is all tag_mutated_buffer checks,
        # but the new value is computed outside the partition.
        program = _edge_program(ReadThenReplaceStateModule())
        _tag(program, {ADD: "p"})

        tag_owned_mutated_buffers(program)
        self.assertNotIn("delegation_tag", _state_placeholder(program).meta)

        tag_mutated_buffer(program)
        self.assertEqual(_state_placeholder(program).meta.get("delegation_tag"), "p")

    def test_a_buffer_read_by_two_partitions_is_left_to_the_graph(self):
        program = _edge_program(TwoReadersStateModule())
        _tag(program, {ADD: "p0", SUB: "p1", SIN: "p0"})

        tag_owned_mutated_buffers(program)

        self.assertNotIn("delegation_tag", _state_placeholder(program).meta)

    def test_lowering_moves_an_owned_buffer_into_the_delegate(self):
        edge = exir.to_edge(
            torch.export.export(
                ReadThenReplaceStateModule(), (torch.ones(3),), strict=True
            )
        ).to_backend(TagOpsPartitioner({ADD: "p", SIN: "p"}))

        self.assertEqual(edge.exported_program().graph_signature.buffers_to_mutate, {})

    def test_lowering_keeps_a_cpu_updated_buffer_in_the_graph(self):
        edge = exir.to_edge(
            torch.export.export(
                ReadThenReplaceStateModule(), (torch.ones(3),), strict=True
            )
        ).to_backend(TagOpsPartitioner({ADD: "p"}))

        self.assertEqual(
            list(edge.exported_program().graph_signature.buffers_to_mutate.values()),
            ["state"],
        )


if __name__ == "__main__":
    unittest.main()
