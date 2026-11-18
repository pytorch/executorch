# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import (
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)
from executorch.exir.tests.test_memory_format_ops_pass_utils import (
    SimpleCloneChannelsLastModule,
    SimpleToCopyChannelsLastModule,
)
from torch.export import export
from torch.fx import GraphModule
from torch.testing import FileCheck
from torch.testing._internal.common_utils import TestCase


class CloneChannelsLastReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.clone(memory_format=torch.channels_last))


class ToChannelsLastReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.to(memory_format=torch.channels_last))


class ContiguousModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()


class TestRemoveCloneOpsTransform(TestCase):
    # Clone ops can appear as either aten.clone or _clone_dim_order depending on the _skip_dim_order flag.
    # _skip_dim_order=True tests aten.clone
    # _skip_dim_order=False tests _clone_dim_order
    CLONE_OP_CASES = [
        (True, "executorch_exir_dialects_edge__ops_aten_clone_default"),
        (
            False,
            "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default",
        ),
    ]

    def test_dq_clone_q_linear(self):
        """
        Test RemoveCloneOpsTransform on a graph with d/q -> clone -> q -> linear pattern

        Before: Should contain all nodes
        After: Should only have the final dequantize and linear operations
        """

        # Create a graph module directly with the pattern: quant -> clone -> dequant -> fp linear
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                # This will be replaced with our custom graph
                return self.linear(x)

        # Create a module instance
        module = TestModule()

        # Create a new graph with our desired pattern
        graph = torch.fx.Graph()

        # Add placeholders
        input_node = graph.placeholder("x")

        # Create nodes for our pattern: quant -> clone -> dequant -> fp linear
        qparams = (0.1, 0, -128, 127, torch.int8)

        # Dequantize node
        dequant_node = graph.create_node(
            "call_function",
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            args=(input_node, *qparams),
            kwargs={},
        )

        # Clone node.
        # Use Edge op as this is an executorch pass
        clone_node = graph.create_node(
            "call_function",
            exir_ops.edge.aten.clone.default,
            args=(dequant_node,),
            kwargs={},
        )

        # Quantize node
        quant_node = graph.create_node(
            "call_function",
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=(clone_node, *qparams),
            kwargs={},
        )

        linear_input = graph.call_function(
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            args=(quant_node, *qparams),
        )
        weight = graph.create_node("get_attr", "linear.weight")
        bias = graph.create_node("get_attr", "linear.bias")
        linear_node = graph.create_node(
            "call_function",
            torch.nn.functional.linear,
            args=(linear_input, weight, bias),
            kwargs={},
        )

        # Output
        graph.output(linear_node)

        # Create a GraphModule with our custom graph
        gm = GraphModule(module, graph)

        # Verify we have the expected nodes before transformation using FileCheck
        FileCheck().check(
            "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
        ).check(
            "executorch_exir_dialects_edge__ops_aten_clone_default",
        ).check(
            "torch.ops.quantized_decomposed.quantize_per_tensor.default",
        ).check(
            "torch._C._nn.linear",
        ).run(
            gm.code
        )

        # Apply the transform
        inputs = torch.randint(-128, 128, (2, 10), dtype=torch.int8)
        expected = gm(inputs)
        transformed_gm = RemoveCloneOpsTransform()(gm).graph_module

        # Verify the dq -> clone -> q pattern is removed and linear op is still present using FileCheck
        FileCheck().check_not(
            "executorch_exir_dialects_edge__ops_aten_clone_default"
        ).check_count(
            "quantized_decomposed.dequantize_per_tensor.default", 1, exactly=True
        ).check_not(
            "quantized_decomposed.quantize_per_tensor.default"
        ).check_count(
            "torch._C._nn.linear",
            1,
            exactly=True,
        ).run(
            transformed_gm.code
        )
        torch.testing.assert_close(transformed_gm(inputs), expected, rtol=0, atol=0)

    def _count(self, epm, ops) -> int:
        return sum(
            n.target in ops for n in epm.exported_program().graph_module.graph.nodes
        )

    def _to_edge(self, model, x, skip_dim_order):
        exported = export(model.eval(), (x,), strict=True)
        return to_edge(
            exported,
            compile_config=EdgeCompileConfig(_skip_dim_order=skip_dim_order),
        )

    def test_non_contiguous_clone_removed(self):
        """Verify an aten.clone into a non-contiguous layout is deleted."""

        clone_op_str = "executorch_exir_dialects_edge__ops_aten_clone_default"
        x = torch.randn(3, 4, 5, 6).to(memory_format=torch.contiguous_format)
        before_epm = self._to_edge(CloneChannelsLastReluModule(), x, True)
        FileCheck().check_count(clone_op_str, 1, exactly=True).run(
            before_epm.exported_program().graph_module.code
        )

        updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

        FileCheck().check_not(clone_op_str).run(
            updated_epm.exported_program().graph_module.code
        )
        expected = before_epm.exported_program().module()(x)
        actual = updated_epm.exported_program().module()(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(is_contiguous_dim_order(actual))

    def test_non_contiguous_to_copy_removed(self):
        """Verify an aten._to_copy that only changes the layout is deleted."""

        to_copy_ops = {exir_ops.edge.aten._to_copy.default}
        x = torch.randn(3, 4, 5, 6)
        before_epm = self._to_edge(ToChannelsLastReluModule(), x, True)
        self.assertEqual(self._count(before_epm, to_copy_ops), 1)

        updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

        self.assertEqual(self._count(updated_epm, to_copy_ops), 0)
        expected = before_epm.exported_program().module()(x)
        actual = updated_epm.exported_program().module()(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dim_order_layout_copies_kept(self):
        """Verify dim-order layout copies are kept: the runtime honours their layout."""

        dim_order_ops = {
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        }
        for model in (CloneChannelsLastReluModule(), ToChannelsLastReluModule()):
            x = torch.randn(3, 4, 5, 6)
            before_epm = self._to_edge(model, x, False)
            self.assertEqual(self._count(before_epm, dim_order_ops), 1)

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            self.assertEqual(self._count(updated_epm, dim_order_ops), 1)
            actual = updated_epm.exported_program().module()(x)
            self.assertTrue(is_channel_last_dim_order(actual))

    def test_retrace_updates_dim_order(self):
        """Verify nodes after a deleted layout copy get a contiguous dim order."""

        x = torch.randn(3, 4, 5, 6)
        before_epm = self._to_edge(ToChannelsLastReluModule(), x, True)
        (relu,) = [
            n
            for n in before_epm.exported_program().graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.relu.default
        ]
        self.assertTrue(is_channel_last_dim_order(relu.meta["val"]))

        updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

        (relu,) = [
            n
            for n in updated_epm.exported_program().graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.relu.default
        ]
        self.assertTrue(is_contiguous_dim_order(relu.meta["val"]))

    def test_dtype_converting_copy_kept(self):
        """Verify a copy that also converts the dtype is not deleted."""

        to_copy_ops = {
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        }
        for skip_dim_order, _ in self.CLONE_OP_CASES:
            x = torch.randn(3, 4, 5, 6)
            before_epm = self._to_edge(
                SimpleToCopyChannelsLastModule(), x, skip_dim_order
            )

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            self.assertEqual(self._count(updated_epm, to_copy_ops), 1)
            self.assertEqual(
                updated_epm.exported_program().module()(x).dtype, torch.double
            )

    def test_layout_changing_output_copy_raises(self):
        """Verify an aten.clone that gives a graph output a new layout is an error."""

        for preserve_input_output_copies in (False, True):
            x = torch.randn(3, 4, 5, 6)
            before_epm = self._to_edge(SimpleCloneChannelsLastModule(), x, True)

            transform = RemoveCloneOpsTransform(
                preserve_input_output_copies=preserve_input_output_copies
            )
            with self.assertRaisesRegex(RuntimeError, "graph output"):
                transform(before_epm.exported_program().graph_module)

    def test_dim_order_output_copy_kept(self):
        """Verify a dim-order copy that gives a graph output a new layout is kept."""

        clone_op_str = (
            "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
        )
        x = torch.randn(3, 4, 5, 6)
        before_epm = self._to_edge(SimpleCloneChannelsLastModule(), x, False)

        updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

        FileCheck().check_count(clone_op_str, 1, exactly=True).run(
            updated_epm.exported_program().graph_module.code
        )
        actual = updated_epm.exported_program().module()(x)
        self.assertTrue(is_channel_last_dim_order(actual))

    def test_contiguous_clone_of_channels_last_input_kept(self):
        """Verify a contiguous clone of a channels-last input is kept."""

        for skip_dim_order, clone_op_str in self.CLONE_OP_CASES:
            x = torch.randn(3, 4, 5, 6).to(memory_format=torch.channels_last)
            before_epm = self._to_edge(ContiguousModule(), x, skip_dim_order)
            FileCheck().check_count(clone_op_str, 1, exactly=True).run(
                before_epm.exported_program().graph_module.code
            )

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            FileCheck().check_count(clone_op_str, 1, exactly=True).run(
                updated_epm.exported_program().graph_module.code
            )
            actual = updated_epm.exported_program().module()(x)
            self.assertTrue(is_contiguous_dim_order(actual))

    def test_clone_memory_format_emits_without_dim_order(self):
        """Verify a channels_last aten.clone serializes once the transform has run."""

        model = CloneChannelsLastReluModule()
        x = torch.randn(3, 4, 5, 6)

        exported = export(model.eval(), (x,), strict=True)
        edge = to_edge(
            exported, compile_config=EdgeCompileConfig(_skip_dim_order=True)
        ).transform([RemoveCloneOpsTransform()])

        self.assertGreater(len(edge.to_executorch().buffer), 0)

    def test_clone_identity_removed(self):
        """Verify identity clone ops are removed by RemoveCloneOpsTransform."""

        for skip_dim_order, clone_op_str in self.CLONE_OP_CASES:
            model = SimpleCloneChannelsLastModule()
            x = torch.randn(3, 4, 5, 6).to(memory_format=torch.channels_last)

            exported = export(model.eval(), (x,), strict=True)
            before_epm = to_edge(
                exported,
                compile_config=EdgeCompileConfig(_skip_dim_order=skip_dim_order),
            )

            FileCheck().check_count(clone_op_str, 1, exactly=True).run(
                before_epm.exported_program().graph_module.code
            )

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            FileCheck().check_not(clone_op_str).run(
                updated_epm.exported_program().graph_module.code
            )

            expected = before_epm.exported_program().module()(x)
            actual = updated_epm.exported_program().module()(x)
            assert torch.allclose(actual, expected)
            assert is_channel_last_dim_order(actual)


if __name__ == "__main__":
    unittest.main()
