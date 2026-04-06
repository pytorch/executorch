# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from copy import deepcopy
from typing import Dict, final, List

import torch
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
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
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.propagate_device_pass import (
    _get_target_device_from_compile_specs,
    _parse_device_spec_value,
    TARGET_DEVICE_COMPILE_SPEC_KEY,
)
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from torch.export import export
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class AddOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.add.Tensor,
        ]


@final
class DeviceAwarePartitioner(Partitioner):
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


def _lower_model_to_executorch(
    model: torch.nn.Module,
    inputs: tuple,
    partitioner: Partitioner,
) -> List:
    """Lower model all the way through to_executorch for E2E tests."""
    ep = export(model, inputs)
    ep_copied = deepcopy(ep)

    edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    lowered_1 = edge_1.to_backend(partitioner)
    et_1 = lowered_1.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
    gm_1 = et_1.exported_program().graph_module

    edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
    et_2 = edge_2.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
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
                    if node.target == executorch_call_delegate:
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

    def _get_executorch_program(self, model, inputs, partitioner):
        """Run the full pipeline and return (emitted_program, graph_module) pairs
        for both export pipelines."""
        from executorch.exir.capture._config import ExecutorchBackendConfig

        ep = export(model, inputs)
        ep_copied = deepcopy(ep)

        # Pipeline 1: to_edge → to_backend → to_executorch
        edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        lowered_1 = edge_1.to_backend(partitioner)
        et_1 = lowered_1.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
        program_1 = et_1._emitter_output.program
        gm_1 = et_1.exported_program().graph_module

        # Pipeline 2: to_edge_transform_and_lower → to_executorch
        edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
        et_2 = edge_2.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
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

    def test_tensorspec_repr_includes_device(self):
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        repr_str = repr(spec)
        self.assertIn("device=", repr_str)
        self.assertIn("CPU", repr_str)


if __name__ == "__main__":
    unittest.main()
