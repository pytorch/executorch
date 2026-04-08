# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from copy import deepcopy
from typing import List, Optional

# Import to register et_copy ops
import executorch.exir.passes._device_copy_ops_registry  # noqa: F401

import torch
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.backend.test.device_util import (
    CpuOnlyPartitioner,
    DeviceAwarePartitioner,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.passes.propagate_device_pass import (
    _get_target_device_from_compile_specs,
    _parse_device_spec_value,
    TARGET_DEVICE_COMPILE_SPEC_KEY,
)
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from torch.export import export


def _lower_model_to_executorch(
    model: torch.nn.Module,
    inputs: tuple,
    partitioner: Partitioner,
    et_config: Optional[ExecutorchBackendConfig] = None,
) -> List:
    """Lower model all the way through to_executorch for E2E tests."""
    if et_config is None:
        et_config = ExecutorchBackendConfig(emit_stacktrace=False)
    ep = export(model, inputs)
    ep_copied = deepcopy(ep)

    edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    lowered_1 = edge_1.to_backend(partitioner)
    et_1 = lowered_1.to_executorch(deepcopy(et_config))
    gm_1 = et_1.exported_program().graph_module

    edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
    et_2 = edge_2.to_executorch(deepcopy(et_config))
    gm_2 = et_2.exported_program().graph_module

    return [
        ("to_edge+to_backend", gm_1),
        ("to_edge_transform_and_lower", gm_2),
    ]


class TestPropagateDevicePass(unittest.TestCase):
    @staticmethod
    def _collect_tensor_specs(node: torch.fx.Node) -> List[TensorSpec]:
        """Return a flat list of TensorSpecs from a node's 'spec' metadata."""
        spec = node.meta.get("spec")
        if spec is None:
            return []
        if isinstance(spec, TensorSpec):
            return [spec]
        if isinstance(spec, (tuple, list)):
            return [s for s in spec if isinstance(s, TensorSpec)]
        return []

    @staticmethod
    def _is_delegate_getitem(node: torch.fx.Node) -> bool:
        """Return True if *node* is a getitem extracting from a delegate call."""
        if node.target != operator.getitem:
            return False
        source = node.args[0]
        return (
            isinstance(source, torch.fx.Node)
            and source.op == "call_function"
            and source.target == executorch_call_delegate
        )

    def _assert_specs_device(
        self,
        specs: List[TensorSpec],
        expected_device: DeviceType,
        msg: str,
        expected_index: int | None = None,
    ) -> None:
        """Assert every spec has the expected device (and optionally index)."""
        for s in specs:
            self.assertEqual(s.device, expected_device, msg)
            if expected_index is not None:
                self.assertEqual(s.device_index, expected_index)

    def _assert_buffer_device(
        self,
        spec: TensorSpec,
        program,
        expected_device: DeviceType,
        msg: str,
    ) -> None:
        """Assert the emitted program maps the spec's buffer to the expected device.

        The memory planner assigns each TensorSpec a ``mem_id`` (buffer index).
        When ``enable_non_cpu_memory_planning`` is True, non-CPU buffers get an
        entry in ``execution_plan[0].non_const_buffer_device``.  CPU buffers have
        no explicit entry (CPU is the default).
        """
        plan = program.execution_plan[0]
        mem_id = spec.mem_id
        self.assertIsNotNone(mem_id, f"{msg}: spec.mem_id should not be None")

        if expected_device == DeviceType.CPU:
            # CPU buffers have no explicit entry in non_const_buffer_device.
            if plan.non_const_buffer_device is not None:
                for entry in plan.non_const_buffer_device:
                    self.assertNotEqual(
                        entry.buffer_idx,
                        mem_id,
                        f"{msg}: buffer {mem_id} should be CPU but found "
                        f"in non_const_buffer_device as {entry.device_type.name}",
                    )
        else:
            self.assertIsNotNone(
                plan.non_const_buffer_device,
                f"{msg}: non_const_buffer_device should exist for non-CPU buffers",
            )
            matching = [
                e for e in plan.non_const_buffer_device if e.buffer_idx == mem_id
            ]
            self.assertEqual(
                len(matching),
                1,
                f"{msg}: expected exactly one entry for buffer {mem_id} "
                f"in non_const_buffer_device, got {len(matching)}",
            )
            self.assertEqual(
                matching[0].device_type,
                expected_device,
                f"{msg}: buffer {mem_id} device type mismatch",
            )

    @staticmethod
    def _collect_copy_nodes(gm):
        """Classify call_function nodes into H2D, D2H, delegate, and getitem lists."""
        h2d, d2h, delegate, getitem = [], [], [], []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.et_copy._h2d_copy.out:
                h2d.append(node)
            elif node.target == torch.ops.et_copy._d2h_copy.out:
                d2h.append(node)
            elif node.target == executorch_call_delegate:
                delegate.append(node)
            elif node.target == operator.getitem:
                getitem.append(node)
        return {"h2d": h2d, "d2h": d2h, "delegate": delegate, "getitem": getitem}

    @staticmethod
    def _collect_placeholders_by_device(gm):
        """Partition placeholder nodes by device type. Returns (cuda_list, cpu_list)."""
        cuda, cpu = [], []
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec) and spec.device == DeviceType.CUDA:
                cuda.append(node)
            elif isinstance(spec, TensorSpec):
                cpu.append(node)
        return cuda, cpu

    def _collect_delegate_getitems(self, gm):
        """Return list of getitem nodes extracting from delegate calls."""
        return [n for n in gm.graph.nodes if self._is_delegate_getitem(n)]

    def _assert_nodes_device(
        self, nodes, expected_device, pipeline, label, expected_index=None
    ):
        """Assert every node's TensorSpec has the expected device."""
        for node in nodes:
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec):
                self.assertEqual(
                    spec.device,
                    expected_device,
                    f"[{pipeline}] {label} '{node.name}' should have "
                    f"{expected_device.name} device spec",
                )
                if expected_index is not None:
                    self.assertEqual(spec.device_index, expected_index)

    def _assert_nodes_buffer_device(
        self, nodes, program, expected_device, pipeline, label
    ):
        """Assert each node's buffer is mapped to the expected device."""
        for node in nodes:
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec):
                self._assert_buffer_device(
                    spec,
                    program,
                    expected_device,
                    f"[{pipeline}] {label} '{node.name}' buffer",
                )

    # ---- Integration tests: copy nodes after to_executorch ----

    def test_h2d_d2h_nodes_inserted(self):
        """Verify H2D/D2H copy nodes are inserted and survive the full
        to_executorch pipeline with correct .out variant targets, exact
        counts, and proper graph ordering."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                nodes = self._collect_copy_nodes(gm)
                h2d_nodes = nodes["h2d"]
                d2h_nodes = nodes["d2h"]
                delegate_nodes = nodes["delegate"]
                getitem_nodes = nodes["getitem"]

                # Model has 2 inputs, 1 output → 2 H2D, 1 D2H
                self.assertEqual(
                    len(h2d_nodes),
                    2,
                    f"[{pipeline}] Expected 2 H2D copy nodes (one per "
                    f"delegate input), got {len(h2d_nodes)}",
                )
                self.assertEqual(
                    len(d2h_nodes),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node (one per "
                    f"delegate output), got {len(d2h_nodes)}",
                )
                self.assertEqual(len(delegate_nodes), 1)

                # Verify graph ordering:
                # placeholder → h2d_copy → delegate → getitem → d2h_copy → output
                all_nodes = list(gm.graph.nodes)
                delegate_idx = all_nodes.index(delegate_nodes[0])
                for h2d in h2d_nodes:
                    self.assertLess(
                        all_nodes.index(h2d),
                        delegate_idx,
                        f"[{pipeline}] H2D '{h2d.name}' must appear before "
                        f"delegate '{delegate_nodes[0].name}'",
                    )
                for d2h in d2h_nodes:
                    for gi in getitem_nodes:
                        if gi.args[0] == delegate_nodes[0]:
                            self.assertGreater(
                                all_nodes.index(d2h),
                                all_nodes.index(gi),
                                f"[{pipeline}] D2H '{d2h.name}' must appear "
                                f"after getitem '{gi.name}'",
                            )

    def test_e2e_copy_nodes_in_executorch_graph(self):
        """End-to-end: copy nodes survive the full to_executorch pipeline
        and have correct .out targets and device specs on TensorSpecs."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                nodes = self._collect_copy_nodes(gm)
                h2d_nodes = nodes["h2d"]
                d2h_nodes = nodes["d2h"]

                self.assertGreater(
                    len(h2d_nodes),
                    0,
                    f"[{pipeline}] H2D copy nodes must survive to_executorch",
                )
                self.assertGreater(
                    len(d2h_nodes),
                    0,
                    f"[{pipeline}] D2H copy nodes must survive to_executorch",
                )

                for h2d in h2d_nodes:
                    spec = h2d.meta.get("spec")
                    self.assertIsNotNone(
                        spec,
                        f"[{pipeline}] H2D node '{h2d.name}' missing spec",
                    )
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CUDA,
                            f"[{pipeline}] H2D output '{h2d.name}' should be "
                            f"on CUDA, got {spec.device.name}",
                        )
                        self.assertEqual(spec.device_index, 0)

                for d2h in d2h_nodes:
                    spec = d2h.meta.get("spec")
                    self.assertIsNotNone(
                        spec,
                        f"[{pipeline}] D2H node '{d2h.name}' missing spec",
                    )
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CPU,
                            f"[{pipeline}] D2H output '{d2h.name}' should be "
                            f"on CPU, got {spec.device.name}",
                        )

    def test_no_copy_nodes_without_device(self):
        """When the partitioner has no target_device CompileSpec, no H2D/D2H
        copy nodes should be inserted in the final graph."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, CpuOnlyPartitioner()
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    self.assertNotEqual(
                        node.target,
                        torch.ops.et_copy._h2d_copy.out,
                        f"[{pipeline}] Unexpected H2D copy node '{node.name}' "
                        f"when no target_device is set",
                    )
                    self.assertNotEqual(
                        node.target,
                        torch.ops.et_copy._d2h_copy.out,
                        f"[{pipeline}] Unexpected D2H copy node '{node.name}' "
                        f"when no target_device is set",
                    )

        # ---- Integration tests: device consistency after to_executorch ----

    def test_device_consistency_cuda_1(self):
        """Verify device tags are correct with cuda:1 after to_executorch()
        to verify device_index propagation through the full pipeline."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, DeviceAwarePartitioner("cuda:1")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    specs = self._collect_tensor_specs(node)
                    if not specs:
                        continue

                    label = f"[{pipeline}] '{node.name}'"
                    if node.target == executorch_call_delegate:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate should be CUDA",
                            expected_index=1,
                        )
                    elif self._is_delegate_getitem(node):
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate getitem should be CUDA",
                            expected_index=1,
                        )

    def test_no_device_spec_remains_cpu(self):
        """When partitioner has no target_device, all specs remain CPU
        through the full to_executorch pipeline."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, CpuOnlyPartitioner()
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    specs = self._collect_tensor_specs(node)
                    for s in specs:
                        self.assertEqual(
                            s.device,
                            DeviceType.CPU,
                            f"[{pipeline}] All specs should be CPU when no "
                            f"target_device, but node '{node.name}' is {s.device.name}",
                        )

    def test_device_consistency_after_to_executorch(self):
        """Verify device tags are correct in the final graph after
        to_executorch(), not just after PropagateDevicePass alone.
        Copy nodes should bridge CPU ↔ device at delegate boundaries."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    specs = self._collect_tensor_specs(node)
                    if not specs:
                        continue

                    label = f"[{pipeline}] '{node.name}'"
                    if node.target == torch.ops.et_copy._h2d_copy.out:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} H2D output should be CUDA",
                            expected_index=0,
                        )
                    elif node.target == torch.ops.et_copy._d2h_copy.out:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CPU,
                            f"{label} D2H output should be CPU",
                        )
                    elif node.target == executorch_call_delegate:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate should be CUDA",
                            expected_index=0,
                        )
                    elif self._is_delegate_getitem(node):
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate getitem should be CUDA",
                            expected_index=0,
                        )

    # --- Unit tests for helper functions ---

    def test_parse_device_spec_value(self):
        dt, idx = _parse_device_spec_value(b"cuda:0")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"cuda:1")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 1)

        dt, idx = _parse_device_spec_value(b"cpu")
        self.assertEqual(dt, DeviceType.CPU)
        self.assertEqual(idx, 0)

    def test_parse_device_spec_value_unknown_raises(self):
        with self.assertRaises(ValueError):
            _parse_device_spec_value(b"tpu:0")

    def test_parse_device_spec_value_case_insensitive(self):
        dt, idx = _parse_device_spec_value(b"CUDA:0")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"Cuda:2")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 2)

    def test_get_target_device_from_compile_specs(self):
        class MockLoweredModule:
            __slots__ = ["compile_specs"]

            def __init__(self, specs):
                self.compile_specs = specs

        module_with_cuda = MockLoweredModule(
            [
                CompileSpec("max_value", bytes([4])),
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0"),
            ]
        )
        result = _get_target_device_from_compile_specs(module_with_cuda)
        self.assertIsNotNone(result)
        dt, idx = result
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        module_without_device = MockLoweredModule(
            [
                CompileSpec("max_value", bytes([4])),
            ]
        )
        result = _get_target_device_from_compile_specs(module_without_device)
        self.assertIsNone(result)

    # ---- End-to-end tests: verify device info survives to_executorch ----

    def _get_executorch_program(self, model, inputs, partitioner, et_config=None):
        """Run the full pipeline and return (emitted_program, graph_module) pairs
        for both export pipelines."""
        if et_config is None:
            et_config = ExecutorchBackendConfig(emit_stacktrace=False)

        ep = export(model, inputs)
        ep_copied = deepcopy(ep)

        # Pipeline 1: to_edge → to_backend → to_executorch
        edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        lowered_1 = edge_1.to_backend(partitioner)
        et_1 = lowered_1.to_executorch(deepcopy(et_config))
        program_1 = et_1._emitter_output.program
        gm_1 = et_1.exported_program().graph_module

        # Pipeline 2: to_edge_transform_and_lower → to_executorch
        edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
        et_2 = edge_2.to_executorch(deepcopy(et_config))
        program_2 = et_2._emitter_output.program
        gm_2 = et_2.exported_program().graph_module

        return [
            ("to_edge+to_backend", program_1, gm_1),
            ("to_edge_transform_and_lower", program_2, gm_2),
        ]

    def test_e2e_device_on_specs_after_to_executorch(self):
        """
        After the full to_executorch pipeline, delegate output TensorSpecs
        should still have device == CUDA on the graph module nodes.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, _program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                found_delegate = False
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        found_delegate = True
                        specs = node.meta.get("spec")
                        self.assertIsNotNone(specs)
                        if isinstance(specs, TensorSpec):
                            self.assertEqual(
                                specs.device,
                                DeviceType.CUDA,
                                f"[{pipeline}] spec.device should be CUDA after to_executorch",
                            )
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    self.assertEqual(
                                        s.device,
                                        DeviceType.CUDA,
                                        f"[{pipeline}] spec.device should be CUDA after to_executorch",
                                    )

                self.assertTrue(found_delegate)

    def test_e2e_non_delegated_tensor_specs_remain_cpu(self):
        """
        After to_executorch, non-delegated node specs should still be CPU.
        Getitem nodes extracting from a delegate call are considered delegated.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.add(a, b)
                d = torch.sin(c)
                return d

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, _program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    # Skip delegate call nodes
                    if node.target == executorch_call_delegate:
                        continue
                    # Skip getitem nodes that extract from a delegate call
                    if node.target == operator.getitem:
                        source = node.args[0]
                        if (
                            isinstance(source, torch.fx.Node)
                            and source.op == "call_function"
                            and source.target == executorch_call_delegate
                        ):
                            continue

    # ---- Skip-copy optimization tests ----

    def test_skip_h2d_for_method_inputs(self):
        """When skip_h2d_for_method_inputs=True, placeholder inputs feeding
        directly into a device delegate should NOT get _h2d_copy nodes."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            skip_h2d_for_method_inputs=True,
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = self._collect_copy_nodes(gm)
                self.assertEqual(
                    len(nodes["h2d"]),
                    0,
                    f"[{pipeline}] Expected no H2D copy nodes when "
                    f"skip_h2d_for_method_inputs=True, got {len(nodes['h2d'])}",
                )
                self.assertEqual(
                    len(nodes["d2h"]),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node for the single "
                    f"output, got {len(nodes['d2h'])}",
                )

                # Placeholder inputs should be tagged as CUDA since H2D was
                # skipped and the pass sets their spec to the target device.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(len(cpu_ph), 0)
                self._assert_nodes_device(
                    cuda_ph,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                    expected_index=0,
                )

                # Verify buffer device mapping: CUDA placeholders should
                # have their memory planned on a CUDA buffer.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                )

    def test_skip_d2h_for_method_outputs(self):
        """When skip_d2h_for_method_outputs=True, delegate outputs that feed
        directly to the graph output should NOT get _d2h_copy nodes."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            skip_d2h_for_method_outputs=True,
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = self._collect_copy_nodes(gm)
                self.assertEqual(
                    len(nodes["d2h"]),
                    0,
                    f"[{pipeline}] Expected no D2H copy nodes when "
                    f"skip_d2h_for_method_outputs=True, got {len(nodes['d2h'])}",
                )
                self.assertEqual(
                    len(nodes["h2d"]),
                    2,
                    f"[{pipeline}] Expected 2 H2D copy nodes for the two "
                    f"inputs, got {len(nodes['h2d'])}",
                )

                # Delegate getitem nodes feeding to output should stay on
                # CUDA since D2H was skipped.
                getitems = self._collect_delegate_getitems(gm)
                self._assert_nodes_device(
                    getitems,
                    DeviceType.CUDA,
                    pipeline,
                    "Delegate getitem",
                )

                # Verify buffer device mapping: CUDA getitem outputs should
                # have their memory planned on a CUDA buffer.
                self._assert_nodes_buffer_device(
                    getitems,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Getitem",
                )

    def test_skip_both_h2d_and_d2h(self):
        """When both skip flags are True, neither H2D nor D2H copy nodes
        should be inserted for a direct input->delegate->output flow."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = self._collect_copy_nodes(gm)
                self.assertEqual(
                    len(nodes["h2d"]),
                    0,
                    f"[{pipeline}] Expected no H2D copy nodes when "
                    f"skip_h2d_for_method_inputs=True, got {len(nodes['h2d'])}",
                )
                self.assertEqual(
                    len(nodes["d2h"]),
                    0,
                    f"[{pipeline}] Expected no D2H copy nodes when "
                    f"skip_d2h_for_method_outputs=True, got {len(nodes['d2h'])}",
                )

                # Placeholder inputs should be tagged as CUDA since H2D
                # was skipped.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(len(cpu_ph), 0)
                self._assert_nodes_device(
                    cuda_ph,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                    expected_index=0,
                )

                # Delegate getitem outputs should stay on CUDA since D2H
                # was skipped.
                getitems = self._collect_delegate_getitems(gm)
                self._assert_nodes_device(
                    getitems,
                    DeviceType.CUDA,
                    pipeline,
                    "Delegate getitem",
                )

                # Verify buffer device mapping: both input and output
                # buffers should be on CUDA.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                )
                self._assert_nodes_buffer_device(
                    getitems,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Getitem",
                )

    def test_skip_h2d_partial_with_intermediate_input(self):
        """When skip_h2d_for_method_inputs=True, only placeholder inputs
        skip H2D copies. An intermediate (non-placeholder) input feeding
        into the delegate should still get an _h2d_copy node."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.sin(a)
                return torch.add(c, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            skip_h2d_for_method_inputs=True,
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                # sin(a) is intermediate (not a placeholder), so it still
                # gets an H2D copy. Placeholder b is skipped.
                nodes = self._collect_copy_nodes(gm)
                self.assertEqual(
                    len(nodes["h2d"]),
                    1,
                    f"[{pipeline}] Expected 1 H2D copy node for the "
                    f"intermediate input, got {len(nodes['h2d'])}",
                )
                self.assertEqual(
                    len(nodes["d2h"]),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node for the single "
                    f"output, got {len(nodes['d2h'])}",
                )

                # Exactly 1 placeholder should be on CUDA (b, which feeds
                # directly into the delegate and skips H2D). The other
                # placeholder (a) feeds through sin() so it stays CPU.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(
                    len(cuda_ph),
                    1,
                    f"[{pipeline}] Expected exactly 1 placeholder with CUDA "
                    f"device spec, got {len(cuda_ph)}",
                )

                # Verify buffer device mapping: the CUDA placeholder's
                # buffer should be on CUDA, the CPU placeholder's buffer
                # should be on CPU.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "CUDA placeholder",
                )
                self._assert_nodes_buffer_device(
                    cpu_ph,
                    program,
                    DeviceType.CPU,
                    pipeline,
                    "CPU placeholder",
                )

    def test_tensorspec_repr_includes_device(self):
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        repr_str = repr(spec)
        self.assertIn("device=", repr_str)
        self.assertIn("CPU", repr_str)


if __name__ == "__main__":
    unittest.main()
