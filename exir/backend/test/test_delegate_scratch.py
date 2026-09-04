# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
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
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.emit import emit_program
from executorch.exir.error import InternalError
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.memory_planning import (
    get_node_tensor_specs,
    update_all_tensors_lifetime,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.delegate_scratch_pass import DelegateScratchSpecPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.print_program import print_program
from executorch.exir.program._program import _transform
from executorch.exir.schema import DelegateCall, DelegateScratch, DeviceType
from executorch.exir.tensor import ALIGNMENT, TensorSpec
from executorch.util.activation_memory_profiler import create_tensor_allocation_info
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase

# Scratch bytes each partition asks for, per element of its output.
BYTES_PER_ELEMENT = 128


@final
class DelegateScratchTestBackend(BackendDetails):
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
class NoScratchTestBackend(BackendDetails):
    """Declares nothing, to pin down that the graph is untouched in that case."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(processed_bytes=b"no-scratch-backend-blob")


# The second size is deliberately not a multiple of the planner's alignment, so
# that a buffer described by its reserved size rather than its declared size is
# visible. Rounding it off would silently delete that coverage.
TWO_BUFFER_SIZES = (2048, 500)
assert TWO_BUFFER_SIZES[1] % ALIGNMENT != 0


@final
class TwoBufferScratchTestBackend(BackendDetails):
    """Declares two buffers, to pin down ordering and independent sizing."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return PreprocessResult(
            processed_bytes=b"two-buffer-scratch-backend-blob",
            scratch_specs=[
                DelegateScratchSpec(nbytes=TWO_BUFFER_SIZES[0]),
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


class OnePartitionForAllAddsPartitioner(OneDelegatePerAddPartitioner):
    """Tags every add with the same tag, so they share one delegate call."""

    def _tag_graph(
        self, graph_module: torch.fx.GraphModule
    ) -> Dict[str, DelegationSpec]:
        tags = super()._tag_graph(graph_module)
        for node in graph_module.graph.nodes:
            if "delegation_tag" in node.meta:
                node.meta["delegation_tag"] = "tag0"
        return {"tag0": self.delegation_spec} if tags else {}


class SkipScratchPlanner(MemoryPlanningPass):
    """Stands in for a custom planner that does not know about scratch."""

    def run(self, graph_module, graph_signature=None):
        result = super().run(graph_module, graph_signature)
        for node in result.graph_module.graph.nodes:
            for spec in memory.delegate_scratch_specs(node):
                spec.mem_id = None
                spec.mem_offset = None
        return result


class RetracingPlanner(MemoryPlanningPass):
    """Stands in for a custom planner that retraces before planning.

    The previous design could not support this: the delegate call carried
    extra arguments its lowered module's signature did not have.
    """

    def run(self, graph_module, graph_signature=None):
        retraced = ExportPass()(graph_module).graph_module
        return super().run(retraced, graph_signature)


class SingleDelegateModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.sin(x + y)


class LiveAcrossDelegateModule(torch.nn.Module):
    """``keep`` is produced before the delegated add and consumed after it.

    Multiplied rather than added at the end so the combine is not itself
    delegated.
    """

    def forward(self, x, y):
        keep = torch.sin(x)
        return keep * torch.cos(x + y)


class TwoOutputDelegateModule(torch.nn.Module):
    """Both adds land in one partition, so the delegate call has two outputs."""

    def forward(self, x, y):
        return torch.sin(x + y), torch.cos(x + x)


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


def _scratch(delegate_call: "DelegateCall") -> List[DelegateScratch]:
    """The delegate's scratch buffers, described inline rather than as values."""
    return delegate_call.scratch or []


def _scratch_sizes(delegate_call: "DelegateCall") -> List[int]:
    return [buffer.size for buffer in _scratch(delegate_call)]


def _scratch_offset(buffer: DelegateScratch) -> int:
    return buffer.allocation.memory_offset


def _delegate_calls(plan) -> List[DelegateCall]:
    return [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, DelegateCall)
    ]


def _reserved(nbytes: int) -> int:
    """What the planner sets aside for a buffer of ``nbytes``."""
    return (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _arena_size(plan) -> int:
    return plan.non_const_buffer_sizes[1]


def _overlaps(lhs: TensorSpec, rhs: TensorSpec) -> bool:
    return (
        lhs.mem_offset < rhs.mem_offset + rhs.allocated_memory
        and rhs.mem_offset < lhs.mem_offset + lhs.allocated_memory
    )


class TestDelegateScratch(unittest.TestCase):
    def test_backend_that_declares_no_scratch_is_untouched(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(SingleDelegateModule(), inputs, "NoScratchTestBackend")

        (delegate_call,) = _delegate_calls(plan)
        # Two inputs and one output, and nothing else.
        self.assertEqual(len(delegate_call.args), 3)
        # Absent rather than an empty vector, so the program costs no bytes for
        # a feature it does not use.
        self.assertIsNone(delegate_call.scratch)

    def test_scratch_survives_serialization(self):
        # The buffers are described in their own schema field, so they have to
        # come back from a real .pte, not just from the in-memory program.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(SingleDelegateModule(), inputs, "DelegateScratchTestBackend")

        plan = deserialize_pte_binary(lowered.buffer).program.execution_plan[0]
        (delegate_call,) = _delegate_calls(plan)
        (scratch,) = _scratch(delegate_call)
        self.assertEqual(scratch.size, 12 * BYTES_PER_ELEMENT)
        self.assertEqual(scratch.allocation.memory_id, 1)

    def test_scratch_is_attributed_to_its_delegate_in_the_memory_profile(self):
        # Scratch is not a value, so without this the arena grows with nothing
        # in the timeline to explain it.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(SingleDelegateModule(), inputs, "DelegateScratchTestBackend")

        allocations = create_tensor_allocation_info(lowered.exported_program().graph)
        scratch_bytes = 12 * BYTES_PER_ELEMENT
        owners = {
            allocation.op_name
            for timeline in allocations
            if timeline is not None
            for allocation in timeline.allocations
            if allocation.size_bytes == scratch_bytes
        }
        self.assertEqual(owners, {executorch_call_delegate})

    def test_declared_scratch_is_planned_into_the_arena(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected_bytes = 12 * BYTES_PER_ELEMENT

        baseline = _execution_plan(
            SingleDelegateModule(), inputs, "NoScratchTestBackend"
        )
        plan = _execution_plan(
            SingleDelegateModule(), inputs, "DelegateScratchTestBackend"
        )

        (delegate_call,) = _delegate_calls(plan)
        # Scratch is not an argument at all, so no backend can misread it as an
        # input or an output, and it costs no value slot either.
        self.assertEqual(delegate_call.args, _delegate_calls(baseline)[0].args)
        self.assertEqual(len(plan.values), len(baseline.values))
        (scratch,) = _scratch(delegate_call)
        self.assertEqual(scratch.size, expected_bytes)
        self.assertGreaterEqual(_arena_size(plan), expected_bytes)
        self.assertLessEqual(
            _arena_size(plan), _arena_size(baseline) + _reserved(expected_bytes)
        )

    def test_scratch_inside_control_flow_is_materialized(self):
        inputs = (torch.tensor(True), torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(CondModule(), inputs, "DelegateScratchTestBackend")

        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(
            _scratch_sizes(delegate_call)[0],
            12 * BYTES_PER_ELEMENT,
        )

    def test_to_executorch_twice_is_stable_inside_control_flow(self):
        # to_executorch() writes its lowered graph back into the edge program,
        # so the second call finds the first call's specs already on the node
        # and has to replace them rather than accumulate.
        inputs = (torch.tensor(True), torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(CondModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        first = edge.to_executorch(ExecutorchBackendConfig())
        second = edge.to_executorch(ExecutorchBackendConfig())

        first_plan = first.executorch_program.execution_plan[0]
        second_plan = second.executorch_program.execution_plan[0]
        self.assertEqual(_arena_size(first_plan), _arena_size(second_plan))
        self.assertEqual(
            _scratch_sizes(_delegate_calls(second_plan)[0]),
            _scratch_sizes(_delegate_calls(first_plan)[0]),
        )

    def test_to_executorch_twice_is_stable(self):
        # The delegate call is in the root graph here rather than in a
        # submodule, which the pass walks separately.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        first = edge.to_executorch(ExecutorchBackendConfig()).executorch_program
        second = edge.to_executorch(ExecutorchBackendConfig()).executorch_program

        first_plan, second_plan = first.execution_plan[0], second.execution_plan[0]
        self.assertEqual(_arena_size(first_plan), _arena_size(second_plan))
        self.assertEqual(len(second_plan.values), len(first_plan.values))
        self.assertEqual(
            _scratch_sizes(_delegate_calls(second_plan)[0]),
            _scratch_sizes(_delegate_calls(first_plan)[0]),
        )

    def test_a_user_pass_that_retraces_is_supported(self):
        # config.passes runs on a graph that a previous to_executorch() may
        # have left scratch on. Metadata replays through a retrace, so this
        # neither loses the specs nor breaks the pass.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        config = ExecutorchBackendConfig(passes=[ExportPass()])
        edge.to_executorch(config)
        plan = edge.to_executorch(config).executorch_program.execution_plan[0]

        self.assertEqual(
            _scratch_sizes(_delegate_calls(plan)[0]), [12 * BYTES_PER_ELEMENT]
        )

    def test_specs_survive_a_pass_that_retraces_after_them(self):
        # This is what keeps the pass out of the pipeline's ordering
        # constraints. Scratch is metadata, so the delegate call's arguments
        # still match the signature its lowered module was partitioned with
        # and an interpreter-based pass can still replay the graph.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = _transform(
            edge.exported_program(),
            SpecPropPass(),
            DelegateScratchSpecPass(),
            ExportPass(),
        )

        (call,) = [
            node
            for node in program.graph.nodes
            if node.target is executorch_call_delegate
        ]
        self.assertEqual(
            [spec.nbytes() for spec in memory.delegate_scratch_specs(call)],
            [12 * BYTES_PER_ELEMENT],
        )

    def test_to_executorch_leaves_the_edge_program_retraceable(self):
        # to_executorch() writes its lowered graph back into the caller's edge
        # program. Whatever it leaves behind has to be something the edge
        # program can still be retraced with.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        edge.to_executorch(ExecutorchBackendConfig())

        retraced = ExportPass()(edge.exported_program().graph_module).graph_module
        (call,) = [
            node
            for node in retraced.graph.nodes
            if node.target is executorch_call_delegate
        ]
        # The lowered module, two inputs: what it was partitioned with.
        self.assertEqual(len(call.args), 3)

    def test_running_the_pass_twice_leaves_one_set_of_specs(self):
        # to_executorch() is not the only caller, so the pass has to be safe to
        # apply to a graph that already carries its own output.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = _transform(
            edge.exported_program(),
            SpecPropPass(),
            DelegateScratchSpecPass(),
            DelegateScratchSpecPass(),
        )

        (call,) = [
            node
            for node in program.graph.nodes
            if node.target is executorch_call_delegate
        ]
        self.assertEqual(len(memory.delegate_scratch_specs(call)), 1)

    def test_scratch_is_inserted_when_view_copies_are_kept(self):
        # pre_memory_planning_passes builds a different list in this mode, so
        # the pass has to appear in both branches.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        plan = edge.to_executorch(
            ExecutorchBackendConfig(remove_view_copy=False)
        ).executorch_program.execution_plan[0]

        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(_scratch_sizes(delegate_call), [12 * BYTES_PER_ELEMENT])

    def _lowered_single_delegate(self):
        exported = torch.export.export(
            SingleDelegateModule().eval(), (torch.randn(3, 4), torch.randn(3, 4))
        )
        edge = to_edge_transform_and_lower(
            exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        return to_backend("DelegateScratchTestBackend", edge.exported_program(), [])

    def test_lowered_module_emits_its_own_program_with_scratch(self):
        # program() and buffer() run their own small pipeline rather than
        # to_executorch(), so it has to materialize the scratch too.
        lowered = self._lowered_single_delegate()

        (delegate_call,) = _delegate_calls(lowered.program().execution_plan[0])
        self.assertEqual(_scratch_sizes(delegate_call), [12 * BYTES_PER_ELEMENT])
        self.assertTrue(lowered.buffer())

    def test_emitting_without_the_pass_is_an_error_not_a_silent_drop(self):
        # Any emit path that skips the pass must fail. The count comes from the
        # graph, not the declaration, so otherwise the program ships without
        # the buffers the backend is going to reach for at execute().
        exported = torch.export.export(
            SingleDelegateModule().eval(), (torch.randn(3, 4), torch.randn(3, 4))
        )
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = _transform(
            edge.exported_program(), SpecPropPass(), MemoryPlanningPass()
        )

        with self.assertRaisesRegex(InternalError, "DelegateScratchSpecPass"):
            emit_program(program)

    def test_two_buffers_keep_declaration_order(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(
            SingleDelegateModule(), inputs, "TwoBufferScratchTestBackend"
        )

        (delegate_call,) = _delegate_calls(plan)
        first, second = _scratch(delegate_call)
        self.assertEqual(first.size, TWO_BUFFER_SIZES[0])
        self.assertEqual(second.size, TWO_BUFFER_SIZES[1])
        self.assertNotEqual(_scratch_offset(first), _scratch_offset(second))

    def test_scratch_survives_a_deepcopy_of_the_program(self):
        # The declaration lives on the LoweredBackendModule, which has a
        # hand-written __deepcopy__, so it is not carried by default.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        copied = EdgeProgramManager(copy.deepcopy(edge.exported_program()))
        plan = copied.to_executorch(
            ExecutorchBackendConfig()
        ).executorch_program.execution_plan[0]
        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(
            _scratch_sizes(delegate_call)[0],
            12 * BYTES_PER_ELEMENT,
        )

    def test_scratch_survives_a_serde_round_trip(self):
        # Dropping the declaration here would emit a program without the
        # scratch the backend's blob still expects.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
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
            _scratch_sizes(delegate_call)[0],
            12 * BYTES_PER_ELEMENT,
        )

    def test_scratch_is_live_only_for_the_delegate_call(self):
        # Consumers of the call's outputs extend those outputs' lifetimes.
        # Scratch must not ride along, or it stays allocated across work that
        # could have reused the bytes.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(SingleDelegateModule(), inputs, "DelegateScratchTestBackend")

        nodes = list(lowered.exported_program().graph.nodes)
        (call,) = [node for node in nodes if node.target is executorch_call_delegate]
        index = nodes.index(call)

        # sin() consumes the delegate's output, so the output does outlive the
        # call. That is what the scratch must not inherit.
        (output_spec,) = [
            spec
            for spec in get_node_tensor_specs(call)
            if spec not in memory.delegate_scratch_specs(call)
        ]
        self.assertGreater(output_spec.lifetime[1], index)

        (scratch_spec,) = memory.delegate_scratch_specs(call)
        self.assertEqual(scratch_spec.lifetime, [index, index])

    def test_scratch_does_not_overlap_a_tensor_live_across_the_call(self):
        # The safety property the whole feature rests on. Liveness is taken
        # from the graph rather than from spec.lifetime, so this still holds a
        # planner to account when the lifetime bookkeeping is what is wrong.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(
            LiveAcrossDelegateModule(), inputs, "DelegateScratchTestBackend"
        )

        nodes = list(lowered.exported_program().graph.nodes)
        (call,) = [node for node in nodes if node.target is executorch_call_delegate]
        (scratch,) = memory.delegate_scratch_specs(call)

        live = [
            spec
            for node in nodes
            if nodes.index(node) < nodes.index(call)
            and any(nodes.index(user) > nodes.index(call) for user in node.users)
            for spec in get_node_tensor_specs(node)
            if spec.mem_id == scratch.mem_id and not spec.const
        ]
        self.assertTrue(live, "the model no longer keeps a tensor across the call")
        for spec in live:
            self.assertFalse(
                _overlaps(spec, scratch),
                f"scratch at {scratch.mem_offset}+{scratch.allocated_memory} overlaps "
                f"a tensor live across the call at {spec.mem_offset}+{spec.allocated_memory}",
            )

    def test_emitted_scratch_fits_inside_the_declared_arena(self):
        # What getMemPlannedPtr rejects at Method::init, checked without C++.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        plan = _execution_plan(
            SingleDelegateModule(), inputs, "TwoBufferScratchTestBackend"
        )

        (delegate_call,) = _delegate_calls(plan)
        for buffer in _scratch(delegate_call):
            pool = plan.non_const_buffer_sizes[buffer.allocation.memory_id]
            self.assertLessEqual(buffer.allocation.memory_offset + buffer.size, pool)

    def test_a_multi_output_delegate_keeps_scratch_after_its_outputs(self):
        # get_node_tensor_specs appends scratch, so anything reading it
        # positionally needs the output count to be what it looks like.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(TwoOutputDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[
                OnePartitionForAllAddsPartitioner("DelegateScratchTestBackend")
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        lowered = edge.to_executorch(ExecutorchBackendConfig())

        (call,) = [
            node
            for node in lowered.exported_program().graph.nodes
            if node.target is executorch_call_delegate
        ]
        scratch = memory.delegate_scratch_specs(call)
        self.assertEqual(len(scratch), 1)
        specs = get_node_tensor_specs(call)
        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[len(specs) - len(scratch) :], scratch)

    def test_a_scratch_spec_of_the_wrong_size_is_rejected(self):
        # The count alone is not enough: a spec some other pass rewrote would
        # otherwise hand the backend a span the blob writing into it overruns.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = _transform(
            edge.exported_program(),
            SpecPropPass(),
            DelegateScratchSpecPass(),
            MemoryPlanningPass(),
        )
        (call,) = [
            node
            for node in program.graph.nodes
            if node.target is executorch_call_delegate
        ]
        (spec,) = memory.delegate_scratch_specs(call)
        spec.shape = [spec.nbytes() // 2]

        with self.assertRaisesRegex(InternalError, "declared scratch buffers of"):
            emit_program(program)

    def test_the_key_means_nothing_on_a_node_that_is_not_a_delegate_call(self):
        # Any pass can write metadata. A stray key elsewhere would otherwise be
        # planned into the arena and never emitted.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("NoScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        baseline = _arena_size(
            copy.deepcopy(edge)
            .to_executorch(ExecutorchBackendConfig())
            .executorch_program.execution_plan[0]
        )

        stray = [
            node
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function"
            and node.target is not executorch_call_delegate
        ][0]
        stray.meta[memory.DELEGATE_SCRATCH_SPECS_META_KEY] = [
            TensorSpec(dtype=torch.uint8, shape=torch.Size([100000]))
        ]
        self.assertEqual(memory.delegate_scratch_specs(stray), [])

        plan = edge.to_executorch(
            ExecutorchBackendConfig()
        ).executorch_program.execution_plan[0]
        self.assertEqual(_arena_size(plan), baseline)

    def test_a_backend_that_stops_declaring_scratch_clears_the_old_specs(self):
        # to_executorch() writes its lowered graph back, so the specs from the
        # previous call are still on the node when the second one runs.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        edge.to_executorch(ExecutorchBackendConfig())

        program = edge.exported_program()
        for module in program.graph_module.children():
            if hasattr(module, "_scratch_specs"):
                module._scratch_specs = []
        plan = edge.to_executorch(
            ExecutorchBackendConfig()
        ).executorch_program.execution_plan[0]

        (delegate_call,) = _delegate_calls(plan)
        self.assertIsNone(delegate_call.scratch)

    def test_scratch_reached_from_two_calls_is_live_across_both(self):
        # fx copies node.meta shallowly, so a pass that clones a delegate call
        # leaves two nodes sharing one spec. Truncating to the later call would
        # free the bytes while the earlier one is still writing to them.
        inputs = (torch.randn(3, 4), torch.randn(3, 4), torch.randn(8, 8))
        exported = torch.export.export(TwoDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        program = _transform(
            edge.exported_program(), SpecPropPass(), DelegateScratchSpecPass()
        )
        nodes = list(program.graph.nodes)
        first, second = [
            node for node in nodes if node.target is executorch_call_delegate
        ]
        shared = first.meta[memory.DELEGATE_SCRATCH_SPECS_META_KEY]
        second.meta[memory.DELEGATE_SCRATCH_SPECS_META_KEY] = shared

        update_all_tensors_lifetime(program.graph_module, program.graph_signature)

        (spec,) = shared
        self.assertEqual(spec.lifetime, [nodes.index(first), nodes.index(second)])

    def test_a_memory_planning_pass_that_retraces_is_supported(self):
        # The restriction this design lifts, exercised through the config
        # rather than by calling the pass directly.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        plan = edge.to_executorch(
            ExecutorchBackendConfig(memory_planning_pass=RetracingPlanner())
        ).executorch_program.execution_plan[0]

        (delegate_call,) = _delegate_calls(plan)
        self.assertEqual(_scratch_sizes(delegate_call), [12 * BYTES_PER_ELEMENT])

    def test_unusable_requests_are_rejected_at_declaration(self):
        # A backend author should see these at preprocess(), not deep inside
        # to_executorch() or at load time on the device.
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DelegateScratchSpec(nbytes=0)

    def test_scratch_of_disjoint_delegate_calls_shares_memory(self):
        inputs = (torch.randn(3, 4), torch.randn(3, 4), torch.randn(8, 8))
        baseline = _execution_plan(TwoDelegateModule(), inputs, "NoScratchTestBackend")
        plan = _execution_plan(
            TwoDelegateModule(), inputs, "DelegateScratchTestBackend"
        )

        first, second = _delegate_calls(plan)
        # The runtime resolves scratch onto delegates_[delegate_index], so two
        # call sites must never share an entry.
        self.assertNotEqual(first.delegate_index, second.delegate_index)
        small, large = 12 * BYTES_PER_ELEMENT, 64 * BYTES_PER_ELEMENT
        # Each call is sized from its own partition, not from a shared number.
        self.assertEqual(_scratch_sizes(first), [small])
        self.assertEqual(_scratch_sizes(second), [large])
        # The two lifetimes do not overlap, so the planner reuses the region
        # instead of stacking both buffers. Growth is bounded by the larger of
        # the two rather than by their sum.
        self.assertEqual(
            _scratch_offset(_scratch(first)[0]),
            _scratch_offset(_scratch(second)[0]),
        )
        self.assertGreaterEqual(_arena_size(plan), large)
        self.assertLessEqual(_arena_size(plan), _arena_size(baseline) + large)
        self.assertLess(_arena_size(plan), _arena_size(baseline) + small + large)

    def test_emitted_allocation_matches_the_planner(self):
        # The descriptor is written from the spec but read by the runtime
        # against the arena, so the two have to be the same placement.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(SingleDelegateModule(), inputs, "TwoBufferScratchTestBackend")

        graph = lowered.exported_program().graph
        (call,) = [
            node for node in graph.nodes if node.target is executorch_call_delegate
        ]
        planned = [
            (spec.mem_id, spec.mem_offset)
            for spec in memory.delegate_scratch_specs(call)
        ]

        plan = lowered.executorch_program.execution_plan[0]
        (delegate_call,) = _delegate_calls(plan)
        emitted = [
            (buffer.allocation.memory_id, buffer.allocation.memory_offset)
            for buffer in _scratch(delegate_call)
        ]
        self.assertEqual(emitted, planned)

    def test_print_program_shows_the_scratch(self):
        # Scratch is not a value, so it is invisible in the instruction dump
        # unless print_program is taught about it.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        lowered = _lower(SingleDelegateModule(), inputs, "TwoBufferScratchTestBackend")

        out = io.StringIO()
        print_program(lowered.executorch_program, out=out)
        dump = out.getvalue()
        for size in TWO_BUFFER_SIZES:
            self.assertIn(f"[{size}]", dump)

    def test_a_planner_that_skips_scratch_is_rejected(self):
        # An integrator with a custom planner should hear about it here rather
        # than ship a program with an unresolvable allocation.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[OneDelegatePerAddPartitioner("DelegateScratchTestBackend")],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        with self.assertRaisesRegex(InternalError, "was not memory planned"):
            edge.to_executorch(
                ExecutorchBackendConfig(memory_planning_pass=SkipScratchPlanner())
            )

    def test_scratch_for_a_device_delegate_lands_in_the_device_pool(self):
        # The bytes belong to the accelerator, so planning them into the host
        # pool would hand the backend an address on the wrong device.
        # cuda:1 rather than cuda:0 so that a dropped device_index does not
        # coincide with the default.
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        exported = torch.export.export(SingleDelegateModule().eval(), inputs)
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[
                OneDelegatePerAddPartitioner(
                    "DelegateScratchTestBackend",
                    compile_specs=[CompileSpec("target_device", b"cuda:1")],
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        lowered = edge.to_executorch(ExecutorchBackendConfig())
        plan = lowered.executorch_program.execution_plan[0]

        (delegate_call,) = _delegate_calls(plan)
        (scratch,) = _scratch(delegate_call)
        device_buffer_idxs = {
            entry.buffer_idx
            for entry in (plan.non_const_buffer_device or [])
            if entry.device_type == DeviceType.CUDA and entry.device_index == 1
        }
        self.assertIn(scratch.allocation.memory_id, device_buffer_idxs)
        # The host pool is 1, so a descriptor that ignored the spec's mem_id
        # would land there.
        self.assertNotEqual(scratch.allocation.memory_id, 1)

        graph = lowered.exported_program().graph
        (call,) = [
            node for node in graph.nodes if node.target is executorch_call_delegate
        ]
        (spec,) = memory.delegate_scratch_specs(call)
        self.assertEqual(
            (scratch.allocation.memory_id, scratch.allocation.memory_offset),
            (spec.mem_id, spec.mem_offset),
        )


if __name__ == "__main__":
    unittest.main()
