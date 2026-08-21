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
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import (
    BackendDetails,
    DelegateScratchSpec,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.error import InternalError
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.schema import DelegateCall, ScalarType
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase

# Scratch bytes each partition asks for, per element of its output.
BYTES_PER_ELEMENT = 128


@final
class ScratchBackend(BackendDetails):
    """Declares a scratch requirement derived from the size of its own partition."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        numel = sum(
            out.meta["val"].numel()
            for node in edge_program.graph.nodes
            if node.op == "output"
            for out in node.args[0]
        )
        scratch_specs = (
            [DelegateScratchSpec(nbytes=numel * BYTES_PER_ELEMENT)] if numel else []
        )
        return PreprocessResult(
            processed_bytes=b"scratch-backend-blob",
            scratch_specs=scratch_specs,
        )


@final
class NoScratchBackend(BackendDetails):
    """Declares nothing, to pin down that the graph is untouched in that case."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(processed_bytes=b"no-scratch-backend-blob")


TWO_BUFFER_SIZES = (2048, 512)
POOLED_MEM_ID = 3


@final
class TwoBufferScratchBackend(BackendDetails):
    """Declares two buffers, the first pinned to a specific planned pool."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(
            processed_bytes=b"two-buffer-scratch-backend-blob",
            scratch_specs=[
                DelegateScratchSpec(nbytes=TWO_BUFFER_SIZES[0], mem_id=POOLED_MEM_ID),
                DelegateScratchSpec(nbytes=TWO_BUFFER_SIZES[1]),
            ],
        )


class AddSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function" and node.target is exir_ops.edge.aten.add.Tensor
        )


class OneDelegatePerAddPartitioner(Partitioner):
    """Tags every add node separately, so each becomes its own delegate call.

    Walks control flow submodules too, so that delegates nested in a cond
    branch are covered.
    """

    def __init__(self, backend_id: str) -> None:
        self.delegation_spec = DelegationSpec(backend_id, [])
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


class TwoDelegateModule(torch.nn.Module):
    """The two adds are disjoint, so their delegate calls have disjoint lifetimes."""

    def forward(self, x, y, big):
        return torch.sin(x + y), torch.sin(big + big)


class CondModule(torch.nn.Module):
    """The delegated add lives in a control flow submodule."""

    def forward(self, pred, x, y):
        return torch.cond(pred, lambda a, b: a + b, lambda a, b: a * b, (x, y))


def _lower(module: torch.nn.Module, inputs, backend_id: str):
    exported = torch.export.export(module.eval(), inputs)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[OneDelegatePerAddPartitioner(backend_id)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge.to_executorch(ExecutorchBackendConfig())


def _execution_plan(module: torch.nn.Module, inputs, backend_id: str):
    return _lower(module, inputs, backend_id).executorch_program.execution_plan[0]


def _delegate_calls(plan) -> List[DelegateCall]:
    return [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, DelegateCall)
    ]


def _arena_size(plan) -> int:
    return plan.non_const_buffer_sizes[1]


def _tensor_sizes(plan, value_index: int) -> List[int]:
    return plan.values[value_index].val.sizes


def _offset(plan, value_index: int) -> int:
    return plan.values[value_index].val.allocation_info.memory_offset_low


class TestDelegateScratch(unittest.TestCase):
    def test_backend_that_declares_no_scratch_is_untouched(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(SingleDelegateModule(), inputs, "NoScratchBackend")

        (delegate_call,) = _delegate_calls(plan)
        # Two inputs and one output, and nothing else.
        self.assertEqual(len(delegate_call.args), 3)

    def test_declared_scratch_is_planned_into_the_arena(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected_bytes = 12 * BYTES_PER_ELEMENT

        baseline = _execution_plan(SingleDelegateModule(), inputs, "NoScratchBackend")
        plan = _execution_plan(SingleDelegateModule(), inputs, "ScratchBackend")

        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(len(delegate_call.args), 4)
        # Scratch trails the inputs and outputs, so a runtime that does not
        # know about it cannot misread the arguments that come before.
        scratch_value = delegate_call.args[-1]
        self.assertEqual(_tensor_sizes(plan, scratch_value), [expected_bytes])
        self.assertGreaterEqual(_arena_size(plan), expected_bytes)
        self.assertLessEqual(_arena_size(plan), _arena_size(baseline) + expected_bytes)

    def test_each_delegate_call_gets_its_own_scratch_size(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4), torch.randn(8, 8))
        plan = _execution_plan(TwoDelegateModule(), inputs, "ScratchBackend")

        first, second = _delegate_calls(plan)
        self.assertEqual(_tensor_sizes(plan, first.args[-1]), [12 * BYTES_PER_ELEMENT])
        self.assertEqual(_tensor_sizes(plan, second.args[-1]), [64 * BYTES_PER_ELEMENT])

    def test_scratch_inside_control_flow_is_materialized(self):
        inputs = (torch.tensor(True), torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(CondModule(), inputs, "ScratchBackend")

        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(
            _tensor_sizes(plan, delegate_call.args[-1]), [12 * BYTES_PER_ELEMENT]
        )

    def test_emitting_without_the_pass_is_an_error_not_a_shuffle(self):
        # LoweredBackendModule.program() emits without running the pass. The
        # count must come from the graph, or a real input lands in the scratch
        # slot and the backend writes scratch bytes over a live tensor.
        exported = torch.export.export(
            SingleDelegateModule().eval(), (torch.randn(3, 4), torch.randn(3, 4))
        )
        edge = to_edge_transform_and_lower(
            exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        lowered = to_backend("ScratchBackend", edge.exported_program(), [])

        with self.assertRaisesRegex(InternalError, "InsertDelegateScratchPass"):
            lowered.program()

    def test_to_executorch_twice_is_stable(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("ScratchBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        first = edge.to_executorch(ExecutorchBackendConfig())
        second = edge.to_executorch(ExecutorchBackendConfig())

        first_plan = first.executorch_program.execution_plan[0]
        second_plan = second.executorch_program.execution_plan[0]
        self.assertEqual(_arena_size(first_plan), _arena_size(second_plan))
        self.assertEqual(
            len(_delegate_calls(first_plan)[0].args),
            len(_delegate_calls(second_plan)[0].args),
        )

    def test_to_executorch_twice_is_stable_without_scratch(self):
        # strip_delegate_scratch_pass runs for every export, not just for
        # scratch users, so it must not disturb the common case.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("NoScratchBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        first = edge.to_executorch(ExecutorchBackendConfig())
        second = edge.to_executorch(ExecutorchBackendConfig())

        first_plan = first.executorch_program.execution_plan[0]
        second_plan = second.executorch_program.execution_plan[0]
        self.assertEqual(_arena_size(first_plan), _arena_size(second_plan))
        self.assertEqual(len(_delegate_calls(second_plan)[0].args), 3)

    def test_two_buffers_keep_declaration_order_and_their_own_pools(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(
            SingleDelegateModule(), inputs, "TwoBufferScratchBackend"
        )

        (delegate_call,) = _delegate_calls(plan)
        first, second = delegate_call.args[-2], delegate_call.args[-1]
        self.assertEqual(_tensor_sizes(plan, first), [TWO_BUFFER_SIZES[0]])
        self.assertEqual(_tensor_sizes(plan, second), [TWO_BUFFER_SIZES[1]])
        self.assertEqual(plan.values[first].val.scalar_type, ScalarType.BYTE)
        self.assertEqual(
            plan.values[first].val.allocation_info.memory_id, POOLED_MEM_ID
        )
        self.assertEqual(plan.values[second].val.allocation_info.memory_id, 1)

    def test_scratch_survives_a_deepcopy_of_the_program(self):
        # EdgeProgramManager.transform() deep-copies, so a lost declaration
        # here would emit a program without the scratch argument.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("ScratchBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        plan = (
            edge.transform([])
            .to_executorch(ExecutorchBackendConfig())
            .executorch_program.execution_plan[0]
        )
        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(
            _tensor_sizes(plan, delegate_call.args[-1]), [12 * BYTES_PER_ELEMENT]
        )

    def test_scratch_survives_a_serde_round_trip(self):
        # Dropping the declaration here would emit a program without the
        # scratch argument the backend's blob still expects.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("ScratchBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        buffer = io.BytesIO()
        exir.save(edge.exported_program(), buffer)
        buffer.seek(0)
        reloaded = EdgeProgramManager(exir.load(buffer))

        plan = reloaded.to_executorch(
            ExecutorchBackendConfig()
        ).executorch_program.execution_plan[0]
        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(
            _tensor_sizes(plan, delegate_call.args[-1]), [12 * BYTES_PER_ELEMENT]
        )

    def test_unusable_requests_are_rejected_at_declaration(self):
        # A backend author should see these at preprocess(), not deep inside
        # to_executorch() or at load time on the device.
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DelegateScratchSpec(nbytes=0)
        with self.assertRaisesRegex(ValueError, "mem_id must be at least 1"):
            DelegateScratchSpec(nbytes=16, mem_id=0)

    def test_scratch_of_disjoint_delegate_calls_shares_memory(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4), torch.randn(8, 8))
        baseline = _execution_plan(TwoDelegateModule(), inputs, "NoScratchBackend")
        plan = _execution_plan(TwoDelegateModule(), inputs, "ScratchBackend")

        first, second = _delegate_calls(plan)
        small, large = 12 * BYTES_PER_ELEMENT, 64 * BYTES_PER_ELEMENT
        # The two lifetimes do not overlap, so the planner reuses the region
        # instead of stacking both buffers. Growth is bounded by the larger of
        # the two rather than by their sum.
        self.assertEqual(_offset(plan, first.args[-1]), _offset(plan, second.args[-1]))
        self.assertGreaterEqual(_arena_size(plan), large)
        self.assertLessEqual(_arena_size(plan), _arena_size(baseline) + large)
        self.assertLess(_arena_size(plan), _arena_size(baseline) + small + large)


if __name__ == "__main__":
    unittest.main()
