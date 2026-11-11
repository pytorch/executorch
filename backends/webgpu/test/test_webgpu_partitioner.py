# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Tuple

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)
from executorch.backends.webgpu.partitioner import WebGPUPartitioner
from executorch.backends.webgpu.partitioner.partitioner import (
    _WEBGPU_SKIP_FUSION_PATTERNS,
)
from executorch.backends.webgpu.test.ops.test_elu_silu import (
    EluModule,
    SiluCondModule,
    SiluModule,
    SiluMulModule,
)
from executorch.backends.webgpu.test.tester import Partition, ToEdgeTransformAndLower
from executorch.exir import (
    ExecutorchProgramManager,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.backend.utils import get_non_lowered_nodes
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.lowered_backend_module import get_lowered_submodules
from torch.export import ExportedProgram


class AddModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class ConvReluModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


class ArangeModule(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(0, 8, 1, dtype=self.dtype).to(x.dtype)


class DynamicArangeModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(0, x.shape[0], 1, dtype=torch.float32)


class TestWebGPUPartitioner(unittest.TestCase):
    def _export_add(self) -> ExportedProgram:
        return torch.export.export(
            AddModule(),
            (torch.ones(2, 3), torch.full((2, 3), 2.0)),
        )

    def _partition(self, partitioner: Partitioner) -> PartitionResult:
        return partitioner.partition(to_edge(self._export_add()).exported_program())

    def _lower(self, partitioner: Partitioner) -> ExecutorchProgramManager:
        return to_edge_transform_and_lower(
            self._export_add(), partitioner=[partitioner]
        ).to_executorch()

    def _delegate_ids(self, program: ExecutorchProgramManager) -> List[str]:
        return [
            delegate.id
            for plan in program.executorch_program.execution_plan
            for delegate in plan.delegates
        ]

    def _assert_same_partitioning(
        self,
        webgpu_partitioner: WebGPUPartitioner,
        vulkan_partitioner: VulkanPartitioner,
    ) -> PartitionResult:
        webgpu_result = self._partition(webgpu_partitioner)
        vulkan_result = self._partition(vulkan_partitioner)
        self.assertEqual(webgpu_result.partition_tags, vulkan_result.partition_tags)
        return webgpu_result

    def test_lowers_identically_with_vulkan_backend_id(self) -> None:
        webgpu_program = self._lower(WebGPUPartitioner())
        vulkan_program = self._lower(
            VulkanPartitioner(
                compile_options={"skip_fusion_patterns": _WEBGPU_SKIP_FUSION_PATTERNS}
            )
        )

        self.assertEqual(webgpu_program.buffer, vulkan_program.buffer)
        self.assertEqual(self._delegate_ids(webgpu_program), ["VulkanBackend"])

    def test_fuses_conv_activation_into_conv_with_clamp(self) -> None:
        # The WebGPU runtime implements conv_with_clamp, not conv2d.with_epilogue.
        edge = to_edge_transform_and_lower(
            torch.export.export(ConvReluModule(), (torch.randn(1, 4, 8, 8),)),
            partitioner=[WebGPUPartitioner()],
        )
        ops = [
            op.name
            for _, module in edge.exported_program().graph_module.named_children()
            for op in flatbuffer_to_vk_graph(
                extract_vk_flatbuffer(module.processed_bytes)
            ).chain
        ]
        self.assertIn("et_vk.conv_with_clamp.default", ops)
        self.assertNotIn("et_vk.conv2d.with_epilogue", ops)

    def test_forwards_compile_options(self) -> None:
        compile_options = {"skip_bool_tensors": True}
        expected_options = compile_options.copy()

        result = self._assert_same_partitioning(
            WebGPUPartitioner(compile_options=compile_options),
            VulkanPartitioner(
                compile_options={
                    **compile_options,
                    "skip_fusion_patterns": _WEBGPU_SKIP_FUSION_PATTERNS,
                }
            ),
        )

        self.assertGreater(len(result.partition_tags), 0)
        self.assertEqual(compile_options, expected_options)
        for spec in result.partition_tags.values():
            self.assertEqual(spec.backend_id, "VulkanBackend")
            self.assertIn(
                "skip_bool_tensors", [item.key for item in spec.compile_specs]
            )

    def test_forwards_operator_blocklist(self) -> None:
        operator_blocklist = [exir_ops.edge.aten.add.Tensor]
        expected_blocklist = operator_blocklist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(operator_blocklist=operator_blocklist),
            VulkanPartitioner(operator_blocklist=operator_blocklist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(operator_blocklist, expected_blocklist)

    def test_forwards_operator_allowlist(self) -> None:
        operator_allowlist = [exir_ops.edge.aten.mul.Tensor]
        expected_allowlist = operator_allowlist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(operator_allowlist=operator_allowlist),
            VulkanPartitioner(operator_allowlist=operator_allowlist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(operator_allowlist, expected_allowlist)

    def test_forwards_nn_module_blocklist(self) -> None:
        nn_module_blocklist = ["AddModule"]
        expected_blocklist = nn_module_blocklist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(nn_module_blocklist=nn_module_blocklist),
            VulkanPartitioner(nn_module_blocklist=nn_module_blocklist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(nn_module_blocklist, expected_blocklist)

    def test_forwards_nn_module_allowlist(self) -> None:
        nn_module_allowlist = ["DoesNotMatch"]
        expected_allowlist = nn_module_allowlist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(nn_module_allowlist=nn_module_allowlist),
            VulkanPartitioner(nn_module_allowlist=nn_module_allowlist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(nn_module_allowlist, expected_allowlist)

    def test_forwards_ops_to_not_decompose(self) -> None:
        exported_program = self._export_add()
        webgpu_ops, webgpu_filter = WebGPUPartitioner().ops_to_not_decompose(
            exported_program
        )
        vulkan_ops, vulkan_filter = VulkanPartitioner().ops_to_not_decompose(
            exported_program
        )

        self.assertEqual(webgpu_ops, vulkan_ops + [torch.ops.aten.silu.default])
        self.assertIsNotNone(webgpu_filter)
        self.assertIsNotNone(vulkan_filter)
        assert webgpu_filter is not None
        assert vulkan_filter is not None
        node = next(
            node
            for node in exported_program.graph_module.graph.nodes
            if node.op == "call_function"
        )
        self.assertEqual(webgpu_filter(node), vulkan_filter(node))

    def test_webgpu_tester_defaults_to_webgpu_partitioner(self) -> None:
        self.assertIsInstance(Partition().partitioner, WebGPUPartitioner)
        self.assertTrue(
            all(
                isinstance(partitioner, WebGPUPartitioner)
                for partitioner in ToEdgeTransformAndLower().partitioners
            )
        )

    def _arange_is_delegated(
        self,
        partitioner: Partitioner,
        module: torch.nn.Module,
        example_input: torch.Tensor,
        dynamic_shapes: object = None,
    ) -> bool:
        exported_program = torch.export.export(
            module, (example_input,), dynamic_shapes=dynamic_shapes
        )
        result = partitioner.partition(to_edge(exported_program).exported_program())
        (node,) = [
            node
            for node in result.tagged_exported_program.graph.nodes
            if node.target == exir_ops.edge.aten.arange.start_step
        ]
        return "delegation_tag" in node.meta

    def test_arange_supported_forms_are_delegated(self) -> None:
        for dtype in (torch.float32, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.assertTrue(
                    self._arange_is_delegated(
                        WebGPUPartitioner(), ArangeModule(dtype), torch.zeros(8)
                    )
                )

    def test_arange_unsupported_forms_stay_on_cpu(self) -> None:
        # Vulkan implements fp16 and SymInt-bounded arange; the WebGPU kernel does
        # not, so WebGPU must leave them on CPU rather than fail at delegate init.
        cases = {
            "fp16": (
                ArangeModule(torch.float16),
                torch.zeros(8, dtype=torch.float16),
                None,
            ),
            "dynamic_end": (
                DynamicArangeModule(),
                torch.zeros(8),
                {"x": {0: torch.export.Dim("n", min=2, max=64)}},
            ),
        }
        for name, (module, example_input, dynamic_shapes) in cases.items():
            with self.subTest(case=name):
                self.assertFalse(
                    self._arange_is_delegated(
                        WebGPUPartitioner(), module, example_input, dynamic_shapes
                    )
                )
                self.assertTrue(
                    self._arange_is_delegated(
                        VulkanPartitioner(), module, example_input, dynamic_shapes
                    )
                )

    def _lower_split(
        self,
        partitioner: Partitioner,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        dynamic_shapes: object = None,
    ) -> Tuple[List[torch.fx.Node], List[object]]:
        """Returns the non-lowered nodes and the targets inside the delegates."""
        exported_program = torch.export.export(
            module, inputs, dynamic_shapes=dynamic_shapes
        )
        edge = to_edge_transform_and_lower(exported_program, partitioner=[partitioner])
        graph_module = edge.exported_program().graph_module
        delegated_targets = [
            node.target
            for _, lowered, _ in get_lowered_submodules(graph_module)
            for node in lowered.original_module.graph.nodes
            if node.op == "call_function"
        ]
        return get_non_lowered_nodes(graph_module.graph), delegated_targets

    def _targets(self, nodes: List[torch.fx.Node]) -> List[object]:
        return [node.target for node in nodes]

    def test_layer_norm_over_last_dim_only(self) -> None:
        layer_norm = exir_ops.edge.aten.native_layer_norm.default
        cases = {
            "last_dim": ([16], (2, 8, 16), True),
            "unit_leading_dim": ([1, 16], (2, 1, 16), True),
            "multi_dim": ([4, 3, 8], (1, 4, 3, 8), False),
        }
        for name, (normalized_shape, shape, delegated) in cases.items():
            with self.subTest(case=name):
                non_lowered, in_delegate = self._lower_split(
                    WebGPUPartitioner(),
                    torch.nn.LayerNorm(normalized_shape),
                    (torch.randn(*shape),),
                )
                self.assertEqual(layer_norm in in_delegate, delegated)
                self.assertEqual(
                    layer_norm in self._targets(non_lowered), not delegated
                )

    def test_elu_and_silu_are_delegated(self) -> None:
        cases = {
            "elu_alpha": (EluModule(alpha=0.5), exir_ops.edge.aten.elu.default),
            "elu_scaled": (
                EluModule(alpha=1.5, scale=2.0, input_scale=0.5),
                exir_ops.edge.aten.elu.default,
            ),
            "elu_param_input": (
                EluModule(param_shape=(4, 16)),
                exir_ops.edge.aten.elu.default,
            ),
            "silu": (SiluModule(), exir_ops.edge.aten.silu.default),
            "silu_param_input": (
                SiluModule(param_shape=(4, 16)),
                exir_ops.edge.aten.silu.default,
            ),
        }
        x = torch.linspace(-4.0, 4.0, 64).reshape(4, 16)
        for name, (module, target) in cases.items():
            with self.subTest(case=name):
                non_lowered, delegated = self._lower_split(
                    WebGPUPartitioner(), module, (x,)
                )
                self.assertEqual(non_lowered, [])
                self.assertIn(target, delegated)

    def test_dynamic_silu_is_delegated(self) -> None:
        non_lowered, delegated = self._lower_split(
            WebGPUPartitioner({"require_dynamic_shapes": True}),
            SiluModule(),
            (torch.randn(1, 8, 16),),
            {"x": {1: torch.export.Dim("s", min=1, max=8)}},
        )
        self.assertEqual(non_lowered, [])
        self.assertIn(exir_ops.edge.aten.silu.default, delegated)

    def test_silu_decomposes_where_it_would_not_be_delegated(self) -> None:
        # Kept silu that stays on CPU fails lowering, so each of these must
        # decompose to x * sigmoid(x) instead.
        x = torch.randn(4, 16)
        allowlist_without_silu = WebGPUPartitioner(
            operator_allowlist=[
                exir_ops.edge.aten.sigmoid.default,
                exir_ops.edge.aten.mul.Tensor,
            ]
        )
        cases = {
            # The SwiGLU fusion matches x * sigmoid(x) * up.
            "feeds_mul": (WebGPUPartitioner(), SiluMulModule(), (x, x)),
            "not_allowlisted": (allowlist_without_silu, SiluModule(), (x,)),
            "rank5": (WebGPUPartitioner(), SiluModule(), (torch.randn(1, 2, 3, 4, 5),)),
        }
        for name, (partitioner, module, inputs) in cases.items():
            with self.subTest(case=name):
                non_lowered, delegated = self._lower_split(partitioner, module, inputs)
                self.assertNotIn(exir_ops.edge.aten.silu.default, delegated)
                self.assertNotIn(
                    exir_ops.edge.aten.silu.default, self._targets(non_lowered)
                )

    def test_silu_in_cond_branch_stays_decomposed(self) -> None:
        # Control-flow submodules are not partitioned, so a silu in a torch.cond
        # branch must decompose (a kept one would make lowering raise).
        exported_program = torch.export.export(SiluCondModule(), (torch.randn(4, 16),))
        edge = to_edge_transform_and_lower(
            exported_program, partitioner=[WebGPUPartitioner()]
        )
        targets = [
            node.target
            for module in edge.exported_program().graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
        ]
        self.assertNotIn(exir_ops.edge.aten.silu.default, targets)
        self.assertIn(exir_ops.edge.aten.sigmoid.default, targets)


if __name__ == "__main__":
    unittest.main()
