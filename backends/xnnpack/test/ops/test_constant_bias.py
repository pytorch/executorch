# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for folding constant cat and split before decomposition."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from executorch.backends.xnnpack.operators.node_visitor import (
    _tensor_bytes,
    NodeVisitor,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNGraph
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class ProjectionBiases(torch.nn.Module):
    def __init__(self, split, projection="linear", runtime_bias=False):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(3)]
        )
        bias_shape = (1, 8) if projection == "addmm" else (8,)
        self.biases = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    (torch.arange(8, dtype=torch.float32) + 10 * index).reshape(
                        bias_shape
                    )
                )
                for index in range(3)
            ]
        )
        self.split = split
        self.projection = projection
        self.runtime_bias = runtime_bias

    def forward(self, value, bias):
        parts = [bias if self.runtime_bias else self.biases[0], *self.biases[1:]]
        packed = torch.cat(parts, dim=0)
        if self.split == "chunk":
            split = packed.chunk(3, dim=0)
        elif self.split == "split":
            split = packed.split(1 if self.projection == "addmm" else 8, dim=0)
        else:
            split = packed.split(
                [1, 1, 1] if self.projection == "addmm" else [8, 8, 8],
                dim=0,
            )
        if self.projection == "addmm":
            return tuple(
                torch.addmm(item, value, weight)
                for weight, item in zip(self.weights, split)
            )
        return tuple(
            F.linear(value, weight, item) for weight, item in zip(self.weights, split)
        )


class TestConstantProjectionBiases(unittest.TestCase):
    def test_split_forms_delegate_and_match_compiled_runtime(self):
        torch.manual_seed(0)
        for split in ("chunk", "split", "split_with_sizes"):
            for projection in ("linear", "addmm"):
                with self.subTest(split=split, projection=projection):
                    model = ProjectionBiases(split, projection).eval()
                    inputs = (torch.randn(2, 8), torch.randn(8))
                    program = torch.export.export(model, inputs)
                    edge = to_edge_transform_and_lower(
                        program, partitioner=[XnnpackPartitioner()]
                    )
                    info = get_delegation_info(edge.exported_program().graph_module)
                    op = (
                        "aten_linear_default"
                        if projection == "linear"
                        else "aten_addmm_default"
                    )
                    self.assertEqual(info.delegation_by_operator[op].non_delegated, 0)
                    self.assertEqual(info.num_delegated_nodes, 3)
                    runtime = _load_for_executorch_from_buffer(
                        edge.to_executorch().buffer
                    )
                    for actual, expected in zip(
                        runtime.forward(inputs), model(*inputs)
                    ):
                        torch.testing.assert_close(
                            actual, expected, atol=1e-5, rtol=1e-5
                        )

    def test_inherited_partitioner_folds(self):
        model = ProjectionBiases("chunk").eval()
        inputs = (torch.randn(2, 8), torch.randn(8))
        edge = to_edge_transform_and_lower(
            torch.export.export(model, inputs),
            partitioner=[XnnpackFloatingPointPartitioner()],
        )
        info = get_delegation_info(edge.exported_program().graph_module)
        self.assertEqual(
            info.delegation_by_operator["aten_linear_default"].non_delegated, 0
        )

    def test_serialized_bytes_use_tensor_extent_and_offset(self):
        backing = torch.arange(24, dtype=torch.float32)
        for view in (backing[4:12], backing[:8]):
            with self.subTest(offset=view.storage_offset()):
                self.assertEqual(_tensor_bytes(view), view.clone().numpy().tobytes())
                self.assertEqual(len(_tensor_bytes(view)), 8 * view.element_size())

    def test_quantization_scales_use_tensor_extent_and_offset(self):
        backing = torch.arange(16, dtype=torch.float32) / 10
        for offset, external_tag in ((0, None), (4, "scales.bin")):
            with self.subTest(offset=offset):
                scale = backing[offset : offset + 8]
                store = NamedDataStore()
                graph = XNNGraph("", [], [], 0, [], [], [])
                visitor = NodeVisitor(None, {}, store)
                params = SimpleNamespace(
                    per_channel=True,
                    per_channel_group=False,
                    is_dynamic=False,
                    scale=scale,
                    axis=0,
                )
                visitor.get_quant_params(params, graph, external_tag)
                expected = scale.clone().numpy().tobytes()
                self.assertEqual(store.buffers, [expected])
                self.assertEqual(graph.constant_data[0].size, len(expected))
                if external_tag is not None:
                    self.assertIn(external_tag, store.external_data)

    def test_parameter_views_match_compiled_runtime(self):
        class ViewBias(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                backing = torch.arange(16, dtype=torch.float32)
                self.bias = torch.nn.Parameter(backing[offset : offset + 8])
                self.weight = torch.nn.Parameter(torch.randn(8, 8))

            def forward(self, value):
                return F.linear(value, self.weight, self.bias)

        for offset in (0, 4):
            with self.subTest(offset=offset):
                model = ViewBias(offset).eval()
                inputs = (torch.randn(2, 8),)
                program = torch.export.export(model, inputs)
                edge = to_edge_transform_and_lower(
                    program, partitioner=[XnnpackPartitioner()]
                )
                runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
                torch.testing.assert_close(
                    runtime.forward(inputs)[0], model(*inputs), atol=1e-5, rtol=1e-5
                )

    def test_dynamic_activation_shapes(self):
        model = ProjectionBiases("split_with_sizes").eval()
        example = (torch.randn(2, 8), torch.randn(8))
        program = torch.export.export(
            model,
            example,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)}, None),
        )
        edge = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()])
        info = get_delegation_info(edge.exported_program().graph_module)
        self.assertEqual(
            info.delegation_by_operator["aten_linear_default"].non_delegated, 0
        )
        runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
        for batch in (2, 3):
            inputs = (torch.randn(batch, 8), torch.randn(8))
            for actual, expected in zip(runtime.forward(inputs), model(*inputs)):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_runtime_bias_and_other_arithmetic_stay_dynamic(self):
        model = ProjectionBiases("chunk", runtime_bias=True).eval()
        example = (torch.randn(2, 8), torch.randn(8))
        program = torch.export.export(model, example)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertTrue(
            any(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            )
        )
        alternate = (example[0], example[1] + 2)
        self.assertFalse(
            torch.equal(
                transformed.module()(*example)[0], transformed.module()(*alternate)[0]
            )
        )

        class OtherArithmetic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                return value + self.bias.sin()

        other = torch.export.export(OtherArithmetic(), (torch.randn(8),))
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(other)
        self.assertTrue(
            any(
                node.target == torch.ops.aten.sin.default
                for node in transformed.graph.nodes
            )
        )

        class QuantizedConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                quantized = torch.ops.aten.quantize_per_tensor.default(
                    self.bias, 0.1, 0, torch.quint8
                )
                return value + torch.ops.aten.dequantize.self(quantized)

        quantized = torch.export.export(QuantizedConstant(), (torch.randn(8),))
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(quantized)
        self.assertTrue(
            any(
                node.target == torch.ops.aten.quantize_per_tensor.default
                for node in transformed.graph.nodes
            )
        )

    def test_fresh_activation_mutation_does_not_block_constant_bias(self):
        class FreshTemporary(ProjectionBiases):
            def forward(self, value, bias):
                temporary = value * 2
                temporary.add_(1)
                return super().forward(temporary, bias)

        model = FreshTemporary("chunk").eval()
        inputs = (torch.randn(2, 8), torch.randn(8))
        edge = to_edge_transform_and_lower(
            torch.export.export(model, inputs),
            partitioner=[XnnpackPartitioner()],
        )
        info = get_delegation_info(edge.exported_program().graph_module)
        self.assertEqual(
            info.delegation_by_operator["aten_linear_default"].non_delegated, 0
        )
        runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
        for actual, expected in zip(runtime.forward(inputs), model(*inputs)):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_unsafe_tuple_consumer_skips_folding(self):
        class MixedTupleConsumers(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8, 8))
                self.unsafe_biases = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.randn(8)) for _ in range(3)]
                )
                self.safe_biases = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.randn(8)) for _ in range(3)]
                )

            def forward(self, value):
                unsafe = torch.cat(tuple(self.unsafe_biases)).chunk(3)
                safe = torch.cat(tuple(self.safe_biases)).chunk(3)
                projected_unsafe = F.linear(value, self.weight, unsafe[0])
                projected_safe = F.linear(value, self.weight, safe[0])
                unsupported = torch.ops.aten._foreach_add.Scalar(unsafe, 1.0)
                return projected_unsafe, projected_safe, unsupported[1]

        program = torch.export.export(MixedTupleConsumers(), (torch.randn(2, 8),))
        chunks = [
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.chunk.default
        ]
        unsafe_chunk = chunks[0]
        foreach = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten._foreach_add.Scalar
        )
        foreach.args = (unsafe_chunk, 1.0)
        program.graph_module.recompile()

        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(transformed.graph), before)

    def test_mutated_cat_split_is_not_folded(self):
        class MutatedPacking(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(8))

            def forward(self, value):
                packed = torch.cat((self.bias, self.bias, self.bias))
                packed.add_(1)
                return value + packed.chunk(3)[0]

        program = torch.export.export(MutatedPacking(), (torch.zeros(8),))
        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(transformed.graph), before)

    def test_cat_used_as_out_argument_is_not_folded(self):
        class MutatedOut(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.ones(8))

            def forward(self, value):
                packed = torch.cat((self.bias, self.bias))
                temporary = value * 2
                torch.ops.aten.add.out(temporary, 1, out=packed)
                return packed.chunk(2)[0]

        program = torch.export.export(MutatedOut(), (torch.zeros(16),))
        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(transformed.graph), before)
        edge = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()])
        torch.testing.assert_close(
            edge.exported_program().module()(torch.ones(16)), torch.full((8,), 3.0)
        )

    def test_higher_order_control_flow_skips_folding(self):
        class Conditional(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(8))

            def forward(self, predicate, value):
                packed = torch.cat((self.bias, self.bias)).chunk(2)[0]
                selected = torch.cond(
                    predicate, lambda item: item + 1, lambda item: item - 1, [value]
                )
                return selected + packed

        inputs = (torch.tensor(True), torch.randn(8))
        program = torch.export.export(Conditional(), inputs)
        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(transformed.graph), before)

    def test_mutable_state_is_not_frozen(self):
        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.zeros(8))
                self.register_buffer("other", torch.ones(8))

            def forward(self, value):
                self.bias.add_(1)
                packed = torch.cat((self.bias, self.other, self.other)).chunk(3)
                return value + packed[0]

        model = Stateful()
        program = torch.export.export(model, (torch.zeros(8),))
        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(transformed.graph), before)
        edge = to_edge_transform_and_lower(
            transformed, partitioner=[XnnpackPartitioner()]
        )
        runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
        first = runtime.forward((torch.zeros(8),))[0]
        second = runtime.forward((torch.zeros(8),))[0]
        torch.testing.assert_close(first, torch.ones(8))
        torch.testing.assert_close(second, torch.full((8,), 2.0))

    def test_constant_output_does_not_introduce_edge_op(self):
        class ConstantOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                packed = torch.cat((self.bias, self.bias))
                return value + 1, packed

        program = torch.export.export(ConstantOutput().eval(), (torch.randn(8),))
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        edge_ops = [
            node
            for node in transformed.graph.nodes
            if isinstance(node.target, EdgeOpOverload)
        ]
        self.assertEqual(
            edge_ops,
            [],
            f"Pre-decomposition transform introduced Edge ops: {edge_ops}",
        )


if __name__ == "__main__":
    unittest.main()
