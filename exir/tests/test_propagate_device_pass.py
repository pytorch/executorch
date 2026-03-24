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


class MatmulOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            exir_ops.edge.aten.mm.default,
        ]


@final
class DeviceAwarePartitioner(Partitioner):
    """
    A test partitioner that tags add/mm ops for delegation and includes
    a target_device CompileSpec to indicate the delegate's target device.
    """

    def __init__(self, target_device: str = "cuda:0") -> None:
        super().__init__()
        self.op_support = any_chain(AddOperatorSupport(), MatmulOperatorSupport())
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
    """
    A test partitioner that does NOT include target_device in CompileSpecs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.op_support = any_chain(AddOperatorSupport(), MatmulOperatorSupport())
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


def _lower_model_with_partitioner(
    model: torch.nn.Module,
    inputs: tuple,
    partitioner: Partitioner,
) -> List[torch.fx.GraphModule]:
    """Lower a model through both export pipelines, run to_executorch, and
    return the graph modules.

    Returns a list of (pipeline_name, graph_module) pairs for both:
      1. to_edge → to_backend → to_executorch
      2. to_edge_transform_and_lower → to_executorch
    """
    ep = export(model, inputs)
    ep_copied = deepcopy(ep)

    # Pipeline 1: to_edge → to_backend → to_executorch
    edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    lowered_1 = edge_1.to_backend(partitioner)
    et_1 = lowered_1.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
    gm_1 = et_1.exported_program().graph_module

    # Pipeline 2: to_edge_transform_and_lower → to_executorch
    edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
    et_2 = edge_2.to_executorch(ExecutorchBackendConfig(emit_stacktrace=False))
    gm_2 = et_2.exported_program().graph_module

    return [
        ("to_edge+to_backend", gm_1),
        ("to_edge_transform_and_lower", gm_2),
    ]


class TestPropagateDevicePass(unittest.TestCase):
    def test_delegate_output_specs_get_device(self):
        """
        Delegate output TensorSpecs should have device == CUDA when
        partitioner includes target_device CompileSpec.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
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
                            self.assertEqual(specs.device, DeviceType.CUDA)
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    self.assertEqual(s.device, DeviceType.CUDA)

                self.assertTrue(
                    found_delegate,
                    f"[{pipeline}] Should have at least one delegate call node",
                )

    def test_delegate_input_specs_get_device(self):
        """
        Delegate input TensorSpecs should also have device == CUDA when
        partitioner includes target_device CompileSpec.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        # args[0] is get_attr for the lowered module; args[1:]
                        # are the actual tensor inputs to the delegate.
                        for arg in node.args[1:]:
                            if isinstance(arg, torch.fx.Node):
                                spec = arg.meta.get("spec")
                                if isinstance(spec, TensorSpec):
                                    self.assertEqual(
                                        spec.device,
                                        DeviceType.CUDA,
                                        f"[{pipeline}] Delegate input {arg.name} "
                                        f"should have device CUDA",
                                    )

    def test_non_delegated_nodes_remain_cpu(self):
        """
        Non-delegated nodes should retain device == CPU.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.add(a, b)
                d = torch.sin(c)
                return d

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
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

                    spec = node.meta.get("spec")
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CPU,
                            f"[{pipeline}] Non-delegated node {node.name} should remain on CPU",
                        )

    def test_no_device_spec_remains_cpu(self):
        """
        When a partitioner does NOT include target_device in CompileSpecs,
        delegate output specs should remain on CPU (default).
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, CpuOnlyPartitioner()
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        specs = node.meta.get("spec")
                        if isinstance(specs, TensorSpec):
                            self.assertEqual(specs.device, DeviceType.CPU)
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    self.assertEqual(s.device, DeviceType.CPU)

    def test_getitem_inherits_device_from_delegate(self):
        """
        Getitem nodes that extract outputs from a delegate call should
        inherit the delegate's device.
        """

        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                return z

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                found_getitem = False
                for node in gm.graph.nodes:
                    if node.op == "call_function" and node.target == operator.getitem:
                        source_node = node.args[0]
                        if (
                            isinstance(source_node, torch.fx.Node)
                            and source_node.op == "call_function"
                            and source_node.target == executorch_call_delegate
                        ):
                            found_getitem = True
                            spec = node.meta.get("spec")
                            if isinstance(spec, TensorSpec):
                                self.assertEqual(spec.device, DeviceType.CUDA)

                self.assertTrue(
                    found_getitem,
                    f"[{pipeline}] Should have at least one getitem from delegate node",
                )

    def test_multiple_delegates_same_device(self):
        """
        Multiple delegates targeting the same device should all get the
        correct device annotation.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                x = torch.add(a, b)
                y = torch.sin(x)
                z = torch.add(y, c)
                return z

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                delegate_devices = []
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        specs = node.meta.get("spec")
                        if isinstance(specs, TensorSpec):
                            delegate_devices.append(specs.device)
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    delegate_devices.append(s.device)
                                    break

                for device in delegate_devices:
                    self.assertEqual(device, DeviceType.CUDA)

    def test_tensorspec_from_tensor_captures_device(self):
        """
        TensorSpec.from_tensor should default to CPU device.
        """
        cpu_tensor = torch.randn(3, 4)
        spec = TensorSpec.from_tensor(cpu_tensor)
        self.assertEqual(spec.device, DeviceType.CPU)

    def test_delegate_output_specs_get_device_index(self):
        """
        Delegate output TensorSpecs should have the correct device_index
        when partitioner specifies e.g., cuda:1.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, DeviceAwarePartitioner("cuda:1")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        specs = node.meta.get("spec")
                        self.assertIsNotNone(specs)
                        if isinstance(specs, TensorSpec):
                            self.assertEqual(specs.device, DeviceType.CUDA)
                            self.assertEqual(specs.device_index, 1)
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    self.assertEqual(s.device, DeviceType.CUDA)
                                    self.assertEqual(s.device_index, 1)

    def test_delegate_input_specs_get_device_index(self):
        """
        Delegate input TensorSpecs should also carry the correct device_index.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_with_partitioner(
            model, inputs, DeviceAwarePartitioner("cuda:1")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        for arg in node.args[1:]:
                            if isinstance(arg, torch.fx.Node):
                                spec = arg.meta.get("spec")
                                if isinstance(spec, TensorSpec):
                                    self.assertEqual(spec.device, DeviceType.CUDA)
                                    self.assertEqual(spec.device_index, 1)

    def test_cpu_specs_have_device_index_zero(self):
        """
        Non-delegated CPU specs should have device_index == 0.
        """
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        self.assertEqual(spec.device, DeviceType.CPU)
        self.assertEqual(spec.device_index, 0)

    def test_tensorspec_default_device_is_cpu(self):
        """
        A TensorSpec created directly should default to CPU.
        """
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        self.assertEqual(spec.device, DeviceType.CPU)

    def test_tensorspec_repr_includes_device(self):
        """
        TensorSpec.__repr__ should include the device field.
        """
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        repr_str = repr(spec)
        self.assertIn("device=", repr_str)
        self.assertIn("CPU", repr_str)

    def test_parse_device_spec_value(self):
        """
        _parse_device_spec_value should correctly parse device strings.
        """
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
        """
        _parse_device_spec_value should raise ValueError for unknown device types.
        """
        with self.assertRaises(ValueError):
            _parse_device_spec_value(b"tpu:0")

        with self.assertRaises(ValueError):
            _parse_device_spec_value(b"unknown")

    def test_parse_device_spec_value_case_insensitive(self):
        """
        _parse_device_spec_value should match device names case-insensitively.
        """
        dt, idx = _parse_device_spec_value(b"CUDA:0")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"CPU")
        self.assertEqual(dt, DeviceType.CPU)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"Cuda:2")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 2)

    def test_get_target_device_from_compile_specs(self):
        """
        _get_target_device_from_compile_specs should correctly extract device info.
        """

        class MockLoweredModule:
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

        for pipeline, program, gm in self._get_executorch_program(
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

                    spec = node.meta.get("spec")
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CPU,
                            f"[{pipeline}] Non-delegated node {node.name} should be CPU after to_executorch",
                        )


if __name__ == "__main__":
    unittest.main()
