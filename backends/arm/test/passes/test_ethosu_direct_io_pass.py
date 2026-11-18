# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
import unittest
from dataclasses import replace
from functools import partial
from typing import cast
from unittest.mock import Mock, patch

import torch
from executorch.backends.arm._passes import EthosUMemoryPlanning
from executorch.backends.arm._passes.ethosu_direct_io_pass import (
    _greedy_by_storage_group,
    EthosUPreferredAlgorithmSuite,
    logger,
)
from executorch.backends.arm.arm_vela import VelaIO, VelaIOLayout
from executorch.exir import memory, to_edge
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.backend.backend_details import DelegateScratchSpec
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.error import InternalError
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.memory_planning import (
    _extend_storage_base_lifetimes,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    SpecAllocResult,
    Verifier,
)
from executorch.exir.passes.delegate_scratch_pass import DelegateScratchSpecPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.schema import DelegateCall, Tensor
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
from torch.fx import Graph, GraphModule


def _vela_layout(
    scratch_size: int,
    inputs: list[tuple[int, int]],
    outputs: list[tuple[int, int]],
) -> VelaIOLayout:
    return VelaIOLayout(
        scratch_size=scratch_size,
        inputs=tuple(VelaIO(offset, size, 1) for offset, size in inputs),
        outputs=tuple(VelaIO(offset, size, 1) for offset, size in outputs),
    )


class EthosUMemoryPlanningTest(unittest.TestCase):
    def _graph_with_cpu_nodes(
        self,
        scratch_size: int = 5632,
        input_offset: int = 4864,
        output_offset: int = 0,
        input_size: int = 768,
        output_size: int = 4096,
        graph_output_size: int = 16384,
    ) -> tuple[GraphModule, torch.fx.Node]:
        root = torch.nn.Module()
        root.lowered_module_0 = LoweredBackendModule(
            None,  # type: ignore[arg-type]
            "EthosUBackend",
            b"compiled delegate",
            [],
            scratch_specs=[
                DelegateScratchSpec(nbytes=scratch_size, label="ethosu_scratch")
            ],
        )
        root.lowered_module_0.meta = {
            "_delegate_info_meta": {
                "vela_io": _vela_layout(
                    scratch_size,
                    [(input_offset, input_size)],
                    [(output_offset, output_size)],
                )
            }
        }
        graph = Graph()
        graph_input = graph.placeholder("input")
        graph_input.meta["spec"] = TensorSpec(torch.uint8, torch.Size([input_size * 4]))
        # Clone nodes model CPU work; tensor sizes are supplied for planning only.
        delegate_input = graph.call_function(torch.clone, (graph_input,))
        delegate_input.meta["spec"] = TensorSpec(torch.uint8, torch.Size([input_size]))
        lowered = graph.get_attr("lowered_module_0")
        delegate = graph.call_function(
            executorch_call_delegate, (lowered, delegate_input)
        )
        delegate.meta["spec"] = (TensorSpec(torch.uint8, torch.Size([output_size])),)
        result = graph.call_function(operator.getitem, (delegate, 0))
        result.meta["spec"] = delegate.meta["spec"][0]
        graph_result = graph.call_function(torch.clone, (result,))
        graph_result.meta["spec"] = TensorSpec(
            torch.uint8, torch.Size([graph_output_size])
        )
        output = graph.output(graph_result)
        output.meta["spec"] = graph_result.meta["spec"]
        module = GraphModule(root, graph)
        DelegateScratchSpecPass()(module)
        return module, delegate

    def test_reuses_dead_scratch_tail_without_increasing_peak(self) -> None:
        ordinary, _ = self._graph_with_cpu_nodes()
        shared, delegate = self._graph_with_cpu_nodes()
        MemoryPlanningPass().run(ordinary)
        with self.assertLogs(logger, level="INFO") as report:
            EthosUMemoryPlanning().run(shared)

        report_text = "\n".join(report.output)
        self.assertIn("Selected direct I/O (", report_text)
        self.assertIn("ordinary arena sizes=[0, 20480]", report_text)
        self.assertIn("2/2 attempted tensors placed", report_text)

        self.assertEqual(ordinary.meta["non_const_buffer_sizes"], [0, 20480])
        self.assertEqual(shared.meta["non_const_buffer_sizes"], [0, 20480])
        (scratch,) = memory.delegate_scratch_specs(delegate)
        output = shared.graph.output_node().meta["spec"]
        self.assertEqual(scratch.mem_offset, 0)
        self.assertEqual(output.mem_offset, 4096)
        self.assertEqual(scratch.lifetime, [3, 3])
        self.assertIs(
            cast(torch.fx.Node, delegate.args[1]).meta["spec"].storage_base, scratch
        )
        self.assertIs(delegate.meta["spec"][0].storage_base, scratch)

    def test_falls_back_when_growth_exceeds_percentage_limit(self) -> None:
        args = {
            "scratch_size": 160,
            "input_offset": 144,
            "output_offset": 80,
            "input_size": 16,
            "output_size": 16,
            "graph_output_size": 160,
        }
        ordinary, _ = self._graph_with_cpu_nodes(**args)
        shared, delegate = self._graph_with_cpu_nodes(**args)
        MemoryPlanningPass().run(ordinary)
        baseline_algo = Mock(wraps=MemoryPlanningAlgorithmSuite())
        with self.assertLogs(logger, level="INFO") as report:
            EthosUMemoryPlanning(baseline_algo=baseline_algo).run(shared)
        baseline_algo.assert_called_once()
        report_text = "\n".join(report.output)
        self.assertIn("Selected ordinary:", report_text)
        self.assertIn("0/2 attempted tensors placed", report_text)
        self.assertIn("declared delegate scratch remains planned", report_text)
        self.assertIn("rejected (arena 1 grows by", report_text)

        self.assertEqual(
            shared.meta["non_const_buffer_sizes"],
            ordinary.meta["non_const_buffer_sizes"],
        )
        self.assertIsNone(
            cast(torch.fx.Node, delegate.args[1]).meta["spec"].storage_base
        )
        self.assertIsNone(delegate.meta["spec"][0].storage_base)
        self.assertEqual(
            cast(torch.fx.Node, delegate.args[1]).meta["spec"].lifetime, [1, 3]
        )
        (scratch,) = memory.delegate_scratch(delegate)
        self.assertEqual(scratch.label, "ethosu_scratch")
        self.assertEqual(scratch.spec.nbytes(), args["scratch_size"])
        self.assertEqual(scratch.spec.mem_id, 1)
        self.assertIsNotNone(scratch.spec.mem_offset)
        Verifier(shared, True, True, True).verify_storage_reuse()

    def test_limits_direct_io_arena_growth(self) -> None:
        for (
            persistent_size,
            scale,
            percent,
            byte_limit,
            expected_size,
            aliases_retained,
        ) in (
            (8192, 1, 1.0, 1024, 8448, True),
            (8192, 1, 0.0, 1024, 8384, False),
            (8192, 1, 1.0, 32, 8384, False),
            (1048576, 64, 1.0, 1024, 1060864, False),
            (1048576, 64, 1.0, 8192, 1064960, True),
        ):
            with self.subTest(
                persistent_size=persistent_size,
                scale=scale,
                percent=percent,
                byte_limit=byte_limit,
            ):
                module, delegate = self._graph_with_cpu_nodes(
                    scratch_size=160 * scale,
                    input_offset=144 * scale,
                    output_offset=80 * scale,
                    input_size=16 * scale,
                    output_size=16 * scale,
                    graph_output_size=160 * scale,
                )
                with module.graph.inserting_before(next(iter(module.graph.nodes))):
                    persistent = module.graph.placeholder("persistent")
                    persistent.meta["spec"] = TensorSpec(
                        torch.uint8, torch.Size([persistent_size])
                    )
                output = module.graph.output_node()
                output.args = ((output.args[0], persistent),)
                output.meta["spec"] = (output.meta["spec"], persistent.meta["spec"])
                module.recompile()
                EthosUMemoryPlanning(
                    max_arena_growth_percent=percent,
                    max_arena_growth_bytes=byte_limit,
                ).run(module)

                self.assertEqual(
                    module.meta["non_const_buffer_sizes"], [0, expected_size]
                )
                (scratch,) = memory.delegate_scratch_specs(delegate)
                for spec in (
                    cast(torch.fx.Node, delegate.args[1]).meta["spec"],
                    delegate.meta["spec"][0],
                ):
                    self.assertIs(
                        spec.storage_base, scratch if aliases_retained else None
                    )
                self.assertIsNotNone(scratch.mem_offset)

    def test_rejects_invalid_arena_growth_limits(self) -> None:
        for percent, byte_limit in (
            (-1.0, 1024),
            (float("nan"), 1024),
            (float("inf"), 1024),
            (1.0, -1),
        ):
            with self.subTest(percent=percent, byte_limit=byte_limit):
                with self.assertRaises(ValueError):
                    EthosUMemoryPlanning(
                        max_arena_growth_percent=percent,
                        max_arena_growth_bytes=byte_limit,
                    )

    def test_requires_scratch_compatible_alignment(self) -> None:
        for alignment in (16, 32):
            with self.subTest(alignment=alignment):
                module, delegate = self._graph_with_cpu_nodes()
                EthosUMemoryPlanning(alignment=alignment).run(module)
                (scratch,) = memory.delegate_scratch_specs(delegate)
                self.assertEqual(cast(int, scratch.mem_offset) % alignment, 0)
                self.assertIs(delegate.meta["spec"][0].storage_base, scratch)
        for alignment in (0, -16, 8, 24):
            with self.subTest(alignment=alignment):
                with self.assertRaisesRegex(ValueError, "positive multiple of 16"):
                    EthosUMemoryPlanning(alignment=alignment)

    def test_reuses_input_storage_for_delegate_output(self) -> None:
        module, delegate = self._graph_with_cpu_nodes(
            scratch_size=160,
            input_offset=0,
            output_offset=0,
            input_size=16,
            output_size=16,
            graph_output_size=64,
        )
        EthosUMemoryPlanning().run(module)

        (scratch,) = memory.delegate_scratch_specs(delegate)
        input_spec = cast(torch.fx.Node, delegate.args[1]).meta["spec"]
        output_spec = delegate.meta["spec"][0]
        self.assertIs(input_spec.storage_base, scratch)
        self.assertIs(output_spec.storage_base, scratch)
        self.assertEqual(input_spec.mem_offset, scratch.mem_offset)
        self.assertEqual(output_spec.mem_offset, scratch.mem_offset)
        self.assertEqual(input_spec.lifetime, [1, 2])
        self.assertEqual(scratch.lifetime, [3, 3])
        Verifier(module, True, True, True).verify_storage_reuse()

    def test_aliases_follow_scratch_into_an_explicit_memory_pool(self) -> None:
        module, delegate = self._graph_with_cpu_nodes()
        (scratch,) = memory.delegate_scratch_specs(delegate)
        scratch.mem_id = 2  # type: ignore[assignment]
        EthosUMemoryPlanning().run(module)

        for spec in (
            cast(torch.fx.Node, delegate.args[1]).meta["spec"],
            delegate.meta["spec"][0],
        ):
            self.assertIs(spec.storage_base, scratch)
            self.assertEqual(spec.mem_id, 2)

    def test_preserves_existing_aliases_when_accepting_or_rejecting_io(self) -> None:
        for existing_scratch_alias in (False, True):
            for accept_io in (False, True):
                with self.subTest(
                    existing_scratch_alias=existing_scratch_alias, accept_io=accept_io
                ):
                    module, delegate = self._graph_with_cpu_nodes(
                        scratch_size=160,
                        input_offset=144,
                        output_offset=80,
                        input_size=16,
                        output_size=16,
                        graph_output_size=160,
                    )
                    input_node = cast(torch.fx.Node, delegate.args[1])
                    input_spec = input_node.meta["spec"]
                    (scratch,) = memory.delegate_scratch_specs(delegate)
                    base = (
                        scratch
                        if existing_scratch_alias
                        else cast(torch.fx.Node, input_node.args[0]).meta["spec"]
                    )
                    offset = 144 if existing_scratch_alias else 16
                    input_spec.storage_base = base
                    input_spec.storage_base_offset = offset

                    with self.assertLogs(logger, level="INFO") as report:
                        EthosUMemoryPlanning(
                            max_arena_growth_percent=1000.0 if accept_io else 0.0,
                            max_arena_growth_bytes=4096,
                        ).run(module)

                    expected_placed = int(existing_scratch_alias) + int(accept_io)
                    self.assertIn(
                        f"{expected_placed}/2 attempted tensors placed",
                        "\n".join(report.output),
                    )

                    self.assertIs(input_spec.storage_base, base)
                    self.assertEqual(input_spec.storage_base_offset, offset)
                    assert base.mem_offset is not None
                    self.assertEqual(input_spec.mem_offset, base.mem_offset + offset)
                    self.assertIs(
                        delegate.meta["spec"][0].storage_base,
                        scratch if accept_io else None,
                    )
                    Verifier(module, True, True, True).verify_storage_reuse()

    def test_plans_multiple_delegates(self) -> None:
        for connection in ("boundary", "shared_input", "direct"):
            with self.subTest(connection=connection):
                module, first, input_0, _ = self._graph_module()
                output = module.graph.output_node()
                first_result = cast(torch.fx.Node, output.args[0])
                second_input = input_0 if connection == "shared_input" else first_result
                lowered_module = LoweredBackendModule(
                    None,  # type: ignore[arg-type]
                    "EthosUBackend",
                    b"compiled delegate",
                    [],
                    scratch_specs=[DelegateScratchSpec(256, "ethosu_scratch")],
                )
                lowered_module.meta = {
                    "_delegate_info_meta": {
                        "vela_io": _vela_layout(
                            256,
                            [(128, second_input.meta["spec"].nbytes())],
                            [(192, 32)],
                        )
                    }
                }
                module.add_submodule("lowered_module_1", lowered_module)
                with module.graph.inserting_before(output):
                    if connection == "boundary":
                        second_input = module.graph.call_function(
                            torch.clone, (first_result,)
                        )
                        second_input.meta["spec"] = TensorSpec(
                            torch.uint8, torch.Size([32])
                        )
                    lowered_node = module.graph.get_attr("lowered_module_1")
                    second = module.graph.call_function(
                        executorch_call_delegate, (lowered_node, second_input)
                    )
                    second.meta["spec"] = (TensorSpec(torch.uint8, torch.Size([32])),)
                    result = module.graph.call_function(operator.getitem, (second, 0))
                    result.meta["spec"] = second.meta["spec"][0]
                output.args = (result,)
                output.meta["spec"] = result.meta["spec"]
                module.recompile()
                DelegateScratchSpecPass()(module)

                with self.assertLogs(logger, level="INFO") as report:
                    EthosUMemoryPlanning(
                        max_arena_growth_percent=1000.0, max_arena_growth_bytes=4096
                    ).run(module)

                attempted = {"boundary": 5, "shared_input": 3, "direct": 4}[connection]
                self.assertIn(
                    f"{attempted}/{attempted} attempted tensors placed",
                    "\n".join(report.output),
                )

                (first_scratch,) = memory.delegate_scratch_specs(first)
                (second_scratch,) = memory.delegate_scratch_specs(second)
                input_spec = second_input.meta["spec"]
                if connection == "shared_input":
                    self.assertIsNone(input_spec.storage_base)
                elif connection == "direct":
                    self.assertIs(input_spec.storage_base, first_scratch)
                    self.assertEqual(input_spec.storage_base_offset, 96)
                else:
                    self.assertIs(input_spec.storage_base, second_scratch)
                    self.assertEqual(input_spec.storage_base_offset, 128)
                self.assertIs(result.meta["spec"].storage_base, second_scratch)
                Verifier(module, True, True, True).verify_storage_reuse()

    def _graph_module(
        self,
    ) -> tuple[GraphModule, torch.fx.Node, torch.fx.Node, torch.fx.Node]:
        root = torch.nn.Module()
        root.lowered_module_0 = LoweredBackendModule(
            None,  # type: ignore[arg-type]
            "EthosUBackend",
            b"compiled delegate",
            [],
            scratch_specs=[DelegateScratchSpec(nbytes=160, label="ethosu_scratch")],
        )

        root.lowered_module_0.meta = {
            "_delegate_info_meta": {
                "vela_io": _vela_layout(160, [(16, 16), (48, 16)], [(96, 32)])
            }
        }
        graph = Graph()
        input_0 = graph.placeholder("input_0")
        input_0.meta["spec"] = TensorSpec.from_tensor(
            torch.empty(16, dtype=torch.uint8)
        )
        input_1 = graph.placeholder("input_1")
        input_1.meta["spec"] = TensorSpec.from_tensor(
            torch.empty(16, dtype=torch.uint8)
        )
        lowered = graph.get_attr("lowered_module_0")
        delegate = graph.call_function(
            executorch_call_delegate, (lowered, input_0, input_1)
        )
        output_spec = TensorSpec.from_tensor(torch.empty(32, dtype=torch.uint8))
        delegate.meta["spec"] = (output_spec,)
        result = graph.call_function(operator.getitem, (delegate, 0))
        result.meta["spec"] = output_spec
        output = graph.output(result)
        output.meta["spec"] = output_spec
        graph_module = GraphModule(root, graph)
        DelegateScratchSpecPass()(graph_module)
        return graph_module, delegate, input_0, input_1

    def test_plans_scratch_inputs_and_outputs_at_vela_offsets(self) -> None:
        graph_module, delegate, input_0, input_1 = self._graph_module()

        EthosUMemoryPlanning().run(graph_module)

        self.assertEqual(len(delegate.args), 3)
        (scratch_spec,) = memory.delegate_scratch_specs(delegate)
        output = next(
            user
            for user in delegate.users
            if user.op == "call_function" and user.target == operator.getitem
        )
        output_spec = cast(TensorSpec, output.meta["spec"])
        expected = ((input_0, 16), (input_1, 48), (output, 96))
        scratch_mem_offset = scratch_spec.mem_offset
        self.assertIsNotNone(scratch_mem_offset)
        assert scratch_mem_offset is not None
        for node, offset in expected:
            spec = cast(TensorSpec, node.meta["spec"])
            self.assertIs(spec.storage_base, scratch_spec)
            self.assertEqual(spec.storage_base_offset, offset)
            self.assertEqual(spec.mem_id, scratch_spec.mem_id)
            self.assertEqual(spec.mem_obj_id, scratch_spec.mem_obj_id)
            self.assertEqual(spec.mem_offset, scratch_mem_offset + offset)

        self.assertIs(output_spec.storage_base, scratch_spec)
        self.assertEqual(scratch_spec.lifetime, [3, 3])
        self.assertEqual(graph_module.meta["non_const_buffer_sizes"][1], 160)

    def test_uses_ordinary_planning_without_io_layout(self) -> None:
        metadata_cases: list[dict[str, dict[str, VelaIOLayout]] | None] = [
            None,
            {"_delegate_info_meta": {}},
        ]
        for metadata in metadata_cases:
            with self.subTest(metadata=metadata):
                graph_module, delegate, input_0, input_1 = self._graph_module()
                lowered = cast(
                    LoweredBackendModule, graph_module.get_submodule("lowered_module_0")
                )
                lowered.meta = metadata

                with self.assertLogs(logger, level="INFO") as report:
                    EthosUMemoryPlanning().run(graph_module)

                report_text = "\n".join(report.output)
                self.assertIn("no eligible direct I/O placements", report_text)
                self.assertIn("0/0 attempted tensors placed", report_text)

                for spec in (
                    input_0.meta["spec"],
                    input_1.meta["spec"],
                    delegate.meta["spec"][0],
                ):
                    self.assertIsNone(spec.storage_base)
                    self.assertIsNotNone(spec.mem_offset)
                (scratch_spec,) = memory.delegate_scratch_specs(delegate)
                self.assertIsNotNone(scratch_spec.mem_offset)

    def test_verifies_storage_reuse_without_debug_logging(self) -> None:
        graph_module, _, _, _ = self._graph_module()
        lowered = cast(
            LoweredBackendModule, graph_module.get_submodule("lowered_module_0")
        )
        lowered.meta = None

        def overlapping_allocations(alignment, specs, *args):
            return MemoryAlgoResult(
                {spec: SpecAllocResult(1, 0, 0) for spec in specs}, [0, 160]
            )

        with (
            patch.object(logging.getLogger(), "isEnabledFor", return_value=False),
            self.assertRaisesRegex(InternalError, "Unexpected storage overlap"),
        ):
            EthosUMemoryPlanning(
                baseline_algo=MemoryPlanningAlgorithmSuite([overlapping_allocations])
            ).run(graph_module)

    def test_does_not_alias_io_in_separate_vela_regions(self) -> None:
        module, delegate, input_0, input_1 = self._graph_module()
        lowered = cast(LoweredBackendModule, module.get_submodule("lowered_module_0"))
        metadata = cast(dict[str, dict[str, VelaIOLayout]], lowered.meta)[
            "_delegate_info_meta"
        ]
        layout = metadata["vela_io"]
        metadata["vela_io"] = replace(
            layout,
            inputs=tuple(replace(io, region=3) for io in layout.inputs),
            outputs=tuple(replace(io, region=4) for io in layout.outputs),
        )
        EthosUMemoryPlanning().run(module)

        for spec in (
            input_0.meta["spec"],
            input_1.meta["spec"],
            delegate.meta["spec"][0],
        ):
            self.assertIsNone(spec.storage_base)

    def test_does_not_alias_an_input_used_after_delegate(self) -> None:
        graph_module, delegate, input_0, _ = self._graph_module()
        output = graph_module.graph.output_node()
        result = cast(torch.fx.Node, output.args[0])
        output.args = ((result, input_0),)
        output.meta["spec"] = (result.meta["spec"], input_0.meta["spec"])
        graph_module.graph.lint()
        graph_module.recompile()

        EthosUMemoryPlanning().run(graph_module)

        self.assertIsNone(cast(TensorSpec, input_0.meta["spec"]).storage_base)
        self.assertNotIn("_shared_alloc_offset", input_0.meta)
        self.assertIsNotNone(cast(TensorSpec, delegate.meta["spec"][0]).storage_base)

    def test_respects_unplanned_graph_inputs_and_outputs(self) -> None:
        graph_module, delegate, input_0, input_1 = self._graph_module()

        EthosUMemoryPlanning(alloc_graph_input=False, alloc_graph_output=False).run(
            graph_module
        )

        for spec in (
            input_0.meta["spec"],
            input_1.meta["spec"],
            delegate.meta["spec"][0],
        ):
            self.assertIsNone(spec.storage_base)
            self.assertIsNone(spec.mem_offset)
        (scratch_spec,) = memory.delegate_scratch_specs(delegate)
        self.assertIsNotNone(scratch_spec.mem_offset)

    def test_does_not_alias_output_with_mismatched_vela_size(self) -> None:
        graph_module, delegate, _, _ = self._graph_module()
        output_spec = delegate.meta["spec"][0]
        output_spec.shape = [16]

        with self.assertLogs(logger, level="INFO") as report:
            EthosUMemoryPlanning().run(graph_module)

        self.assertIsNone(output_spec.storage_base)
        self.assertIn("2/3 attempted tensors placed", "\n".join(report.output))

    def _lowered_module(
        self, scratch_specs: list[DelegateScratchSpec]
    ) -> LoweredBackendModule:
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example = (torch.zeros(16, dtype=torch.uint8),) * 2
        edge = to_edge(torch.export.export(Add(), example))
        lowered = LoweredBackendModule(
            edge.exported_program(),
            "EthosUBackend",
            b"compiled delegate",
            [],
            scratch_specs=scratch_specs,
        )
        lowered.meta = {
            "_delegate_info_meta": {
                "vela_io": _vela_layout(160, [(16, 16), (48, 16)], [(96, 16)])
            }
        }
        return lowered

    def test_emits_scratch_separately_with_aliased_io(self) -> None:
        lowered = self._lowered_module(
            [DelegateScratchSpec(nbytes=160, label="ethosu_scratch")]
        )
        with self.assertNoLogs(logger, level="WARNING"):
            program = deserialize_pte_binary(
                lowered.buffer(memory_planning=EthosUMemoryPlanning())
            ).program

        (plan,) = program.execution_plan
        calls = [
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
            if isinstance(instruction.instr_args, DelegateCall)
        ]
        (call,) = calls
        self.assertEqual(len(call.args), 3)
        assert call.scratch is not None
        (scratch,) = call.scratch
        self.assertEqual(scratch.size, 160)
        for index, offset in zip(call.args, (16, 48, 96)):
            tensor = cast(Tensor, plan.values[index].val)
            allocation = tensor.allocation_info
            assert allocation is not None
            self.assertEqual(allocation.memory_id, scratch.allocation.memory_id)
            self.assertEqual(
                allocation.memory_offset,
                scratch.allocation.memory_offset + offset,
            )
        self.assertEqual(plan.non_const_buffer_sizes[1], 160)

    def test_serializes_without_declared_scratch(self) -> None:
        lowered = self._lowered_module([])
        with self.assertLogs(logger, level="WARNING") as report:
            program = deserialize_pte_binary(
                lowered.buffer(memory_planning=EthosUMemoryPlanning())
            ).program
        self.assertEqual(len(report.records), 1)
        self.assertIn("has no planned scratch", report.output[0])
        self.assertIn("plan_delegate_scratch=True", report.output[0])
        (plan,) = program.execution_plan
        (call,) = (
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
            if isinstance(instruction.instr_args, DelegateCall)
        )
        self.assertEqual(len(call.args), 3)
        self.assertIsNone(call.scratch)
        self.assertEqual(plan.non_const_buffer_sizes[1], 48)


class TestStorageGroupMemoryPlanning(unittest.TestCase):
    def _specs(self) -> tuple[TensorSpec, TensorSpec, TensorSpec]:
        scratch = TensorSpec(torch.uint8, torch.Size([160]))
        scratch.lifetime = [2, 2]  # type: ignore[list-item]
        output = TensorSpec(torch.uint8, torch.Size([16]))
        output.lifetime = [2, 5]  # type: ignore[list-item]
        output.storage_base = scratch
        other = TensorSpec(torch.uint8, torch.Size([144]))
        other.lifetime = [3, 5]  # type: ignore[list-item]
        return scratch, output, other

    def _graph(self, specs: tuple[TensorSpec, ...]) -> GraphModule:
        graph = Graph()
        for index, spec in enumerate(specs):
            graph.placeholder(f"tensor_{index}").meta["spec"] = spec
        graph.output(())
        return GraphModule({}, graph)

    def _plan(self, specs: tuple[TensorSpec, ...], graph: GraphModule) -> list[int]:
        return MemoryPlanningAlgorithmSuite(
            [
                partial(
                    _greedy_by_storage_group,
                    scratch_specs=set(),
                    prioritize_scratch=False,
                )
            ]
        )(16, set(specs), graph, cast(ExportGraphSignature, None), 0)

    def test_reuses_dead_base_bytes_while_protecting_live_alias(self) -> None:
        specs = self._specs()
        scratch, output, other = specs
        graph = self._graph(specs)
        sizes = self._plan(specs, graph)

        self.assertEqual(scratch.lifetime, [2, 2])
        self.assertEqual(sizes, [0, 160])
        self.assertEqual(scratch.mem_offset, 0)
        self.assertEqual(output.mem_offset, 0)
        self.assertEqual(other.mem_offset, 16)
        Verifier(graph, True, True, True).verify_storage_reuse()

    def test_preserves_ordinary_storage_base_lifetimes(self) -> None:
        specs = self._specs()
        scratch, _, _ = specs
        _extend_storage_base_lifetimes(set(specs))
        sizes = self._plan(specs, self._graph(specs))

        self.assertEqual(scratch.lifetime, [2, 5])
        self.assertEqual(sizes, [0, 304])

    def test_protects_full_scratch_during_delegate(self) -> None:
        specs = self._specs()
        scratch, _, other = specs
        other.lifetime = [2, 5]  # type: ignore[list-item]
        sizes = self._plan(specs, self._graph(specs))

        self.assertEqual(sizes, [0, 304])
        self.assertTrue(
            cast(int, scratch.mem_offset) + scratch.allocated_memory
            <= cast(int, other.mem_offset)
            or cast(int, other.mem_offset) + other.allocated_memory
            <= cast(int, scratch.mem_offset)
        )

    def test_honors_pool_prefix_padding_and_separate_memory_ids(self) -> None:
        specs = self._specs()
        scratch, output, other = specs
        scratch.mem_id = 2  # type: ignore[assignment]
        other.mem_id = 1  # type: ignore[assignment]
        graph = self._graph(specs)
        graph.input_mem_buffer_sizes = [0, 32, 48]  # type: ignore[assignment]
        sizes = MemoryPlanningAlgorithmSuite(
            [
                partial(
                    _greedy_by_storage_group,
                    scratch_specs=set(),
                    prioritize_scratch=False,
                )
            ]
        )(16, set(specs), graph, cast(ExportGraphSignature, None), 64)

        self.assertEqual(sizes, [0, 240, 272])
        self.assertEqual(other.mem_offset, 32)
        self.assertEqual(scratch.mem_offset, 48)
        self.assertEqual(output.mem_id, 2)
        self.assertEqual(output.mem_offset, 48)

    def test_supports_nested_aliases(self) -> None:
        specs = self._specs()
        scratch, output, _ = specs
        child = TensorSpec(torch.uint8, torch.Size([16]))
        child.lifetime = [3, 5]  # type: ignore[list-item]
        output.shape = [32]
        child.storage_base = output
        child.storage_base_offset = 16
        specs = (scratch, output, child)
        self._plan(specs, self._graph(specs))

        self.assertEqual(child.mem_offset, cast(int, scratch.mem_offset) + 16)

    def test_rejects_invalid_storage_groups(self) -> None:
        for name in ("alignment", "bounds", "pool", "device", "siblings"):
            with self.subTest(name=name):
                specs = self._specs()
                scratch, output, other = specs
                if name == "alignment":
                    output.storage_base_offset = 1
                elif name == "bounds":
                    output.storage_base_offset = 160
                elif name == "pool":
                    scratch.mem_id = 2  # type: ignore[assignment]
                    output.mem_id = 1  # type: ignore[assignment]
                elif name == "device":
                    output.device_index = 1
                else:
                    other.storage_base = scratch
                with self.assertRaises(InternalError):
                    self._plan(specs, self._graph(specs))

    def test_rejects_missing_storage_base(self) -> None:
        _, output, other = self._specs()
        specs = (output, other)
        with self.assertRaisesRegex(InternalError, "must be memory planned"):
            self._plan(specs, self._graph(specs))


class TestEthosUPreferredAlgorithmSuite(unittest.TestCase):
    def test_prefers_smallest_eligible_plan_over_smaller_baseline(self) -> None:
        baseline = MemoryAlgoResult({}, [0, 8192])
        larger = MemoryAlgoResult({}, [0, 8256])
        smaller = MemoryAlgoResult({}, [0, 8224])
        suite = EthosUPreferredAlgorithmSuite(
            preferred_algos={
                "larger": Mock(return_value=larger),
                "smaller": Mock(return_value=smaller),
            },
            baseline=MemoryPlanningAlgorithmSuite(),
            max_arena_growth_percent=1.0,
            max_arena_growth_bytes=1024,
        )
        with self.assertLogs(logger, level="INFO") as report:
            result = suite._select(
                16,
                set(),
                GraphModule({}, Graph()),
                cast(ExportGraphSignature, None),
                0,
                baseline=baseline,
                scratch_specs=set(),
            )

        self.assertIs(result, smaller)
        self.assertIn("Selected smaller:", "\n".join(report.output))

    def test_checks_each_arena_before_ranking_preferred_plans(self) -> None:
        baseline = MemoryAlgoResult({}, [0, 8192, 8192])
        # The smaller total exceeds arena 1's growth limit. The other ordering
        # is eligible even though its total size is larger.
        rejected = MemoryAlgoResult({}, [0, 8320, 4096])
        accepted = MemoryAlgoResult({}, [0, 8192, 8192])
        suite = EthosUPreferredAlgorithmSuite(
            preferred_algos={
                "rejected": Mock(return_value=rejected),
                "accepted": Mock(return_value=accepted),
            },
            baseline=MemoryPlanningAlgorithmSuite(),
            max_arena_growth_percent=1.0,
            max_arena_growth_bytes=1024,
        )
        result = suite._select(
            16,
            set(),
            GraphModule({}, Graph()),
            cast(ExportGraphSignature, None),
            0,
            baseline=baseline,
            scratch_specs=set(),
        )

        self.assertIs(result, accepted)

    def test_falls_back_when_preferred_placements_are_invalid_or_too_large(
        self,
    ) -> None:
        baseline = MemoryAlgoResult({}, [0, 8192])
        suite = EthosUPreferredAlgorithmSuite(
            preferred_algos={
                "invalid": Mock(side_effect=InternalError("overlapping live ranges")),
                "too large": Mock(return_value=MemoryAlgoResult({}, [0, 8320])),
            },
            baseline=MemoryPlanningAlgorithmSuite(),
            max_arena_growth_percent=1.0,
            max_arena_growth_bytes=1024,
        )
        with self.assertLogs(logger, level="INFO") as report:
            result = suite._select(
                16,
                set(),
                GraphModule({}, Graph()),
                cast(ExportGraphSignature, None),
                0,
                baseline=baseline,
                scratch_specs=set(),
            )

        self.assertIs(result, baseline)
        report_text = "\n".join(report.output)
        self.assertIn("invalid: rejected (invalid placement)", report_text)
        self.assertIn("arena 1 grows by 128 bytes; allowance=81 bytes", report_text)
        self.assertIn("Selected ordinary:", report_text)


if __name__ == "__main__":
    unittest.main()
