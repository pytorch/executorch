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
    def test_shared_weight_norm_preserves_consumers_and_reference(self):
        class SharedWeight(torch.nn.Module):
            def __init__(self, mixed):
                super().__init__()
                self.conv = torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(4, 4, 3), dim=2
                )
                self.mixed = mixed

            def forward(self, value):
                weight = self.conv.weight
                first = F.conv1d(value, weight)
                second = (
                    weight + value.mean() if self.mixed else F.conv1d(value * 2, weight)
                )
                return first, second

        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                model = SharedWeight(mixed).eval()
                inputs = (torch.randn(1, 4, 12),)
                program = torch.export.export(model, inputs)
                graph = str(program.graph)
                signature = str(program.graph_signature)
                state = {
                    key: value.clone() for key, value in program.state_dict.items()
                }
                edge = to_edge_transform_and_lower(
                    program, partitioner=[XnnpackPartitioner()]
                )
                info = get_delegation_info(edge.exported_program().graph_module)
                self.assertEqual(
                    info.delegation_by_operator[
                        "aten_convolution_default"
                    ].non_delegated,
                    0,
                )
                runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
                for actual, expected in zip(runtime.forward(inputs), model(*inputs)):
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                self.assertEqual(str(program.graph), graph)
                self.assertEqual(str(program.graph_signature), signature)
                self.assertEqual(program.state_dict.keys(), state.keys())
                for key, value in state.items():
                    torch.testing.assert_close(program.state_dict[key], value)
                for actual, expected in zip(program.module()(*inputs), model(*inputs)):
                    torch.testing.assert_close(actual, expected)

    def test_external_constants_skip_early_folding(self):
        model = ProjectionBiases("chunk").eval()
        inputs = (torch.randn(2, 8), torch.randn(8))
        program = torch.export.export(model, inputs)
        for node in program.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in program.graph_signature.inputs_to_parameters
            ):
                node.meta["custom"] = {"delegate_constant_tag": "weights.ptd"}
        graph = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertIs(transformed, program)
        self.assertEqual(str(program.graph), graph)

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
                    pte_buffer = edge.to_executorch().buffer
                    runtime = _load_for_executorch_from_buffer(pte_buffer)
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
                pte_buffer = edge.to_executorch().buffer
                runtime = _load_for_executorch_from_buffer(pte_buffer)
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
        pte_buffer = edge.to_executorch().buffer
        runtime = _load_for_executorch_from_buffer(pte_buffer)
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
        pte_buffer = edge.to_executorch().buffer
        runtime = _load_for_executorch_from_buffer(pte_buffer)
        for actual, expected in zip(runtime.forward(inputs), model(*inputs)):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_tuple_consumer_preserves_independent_folding(self):
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
        self.assertEqual(str(program.graph), before)
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.chunk.default
                for node in transformed.graph.nodes
            ),
            1,
        )
        linears = [
            node
            for node in transformed.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        self.assertTrue(all(node.args[2].op == "placeholder" for node in linears))
        inputs = (torch.randn(2, 8),)
        for actual, expected in zip(
            transformed.module()(*inputs), program.module()(*inputs)
        ):
            torch.testing.assert_close(actual, expected)

    def test_unrelated_runtime_tuple_does_not_block_constant_chain(self):
        class RuntimeTuple(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                runtime = value.chunk(2)
                shifted = torch.ops.aten._foreach_add.Scalar(runtime, 1.0)
                constant = torch.cat((self.bias, self.bias)).chunk(2)[0]
                return runtime[0] + constant, shifted[1]

        inputs = (torch.randn(16),)
        program = torch.export.export(RuntimeTuple(), inputs)
        chunk = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.chunk.default
        )
        foreach = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten._foreach_add.Scalar
        )
        foreach.args = (chunk, 1.0)
        program.graph_module.recompile()
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertFalse(
            any(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.aten.chunk.default
                for node in transformed.graph.nodes
            )
        )
        for value in (inputs[0], inputs[0] + 2):
            for actual, expected in zip(
                transformed.module()(value), program.module()(value)
            ):
                torch.testing.assert_close(actual, expected)

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

    def test_unrelated_control_flow_does_not_block_folding(self):
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

        model = Conditional()
        inputs = (torch.tensor(True), torch.randn(8))
        program = torch.export.export(model, inputs)
        before = str(program.graph)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertEqual(str(program.graph), before)
        self.assertFalse(
            any(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.higher_order.cond
                for node in transformed.graph.nodes
            )
        )
        for predicate in (True, False):
            inputs = (torch.tensor(predicate), torch.randn(8))
            torch.testing.assert_close(transformed.module()(*inputs), model(*inputs))

    def test_higher_order_mutation_excludes_only_affected_chain(self):
        class RegionMutation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(8))
                self.safe = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                affected = torch.cat((self.bias, self.bias))
                with torch.no_grad():
                    affected.add_(value)
                safe = torch.cat((self.safe, self.safe))
                return affected + safe

        model = RegionMutation().eval()
        program = torch.export.export(model, (torch.zeros(16),))
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            ),
            1,
        )
        self.assertTrue(
            any(
                isinstance(node.target, torch._ops.HigherOrderOperator)
                for node in transformed.graph.nodes
            )
        )
        for value in (torch.zeros(16), torch.ones(16)):
            torch.testing.assert_close(transformed.module()(value), model(value))

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
        pte_buffer = edge.to_executorch().buffer
        runtime = _load_for_executorch_from_buffer(pte_buffer)
        first = runtime.forward((torch.zeros(8),))[0]
        second = runtime.forward((torch.zeros(8),))[0]
        torch.testing.assert_close(first, torch.ones(8))
        torch.testing.assert_close(second, torch.full((8,), 2.0))

    def test_unrelated_state_mutation_keeps_bias_folding(self):
        class Stateful(ProjectionBiases):
            def __init__(self):
                super().__init__("chunk")
                self.register_buffer("state", torch.zeros(8))

            def forward(self, value, bias):
                self.state.add_(1)
                return *super().forward(value, bias), self.state * 2

        for functionalized in (False, True):
            with self.subTest(functionalized=functionalized):
                model = Stateful().eval()
                inputs = (torch.randn(2, 8), torch.randn(8))
                program = torch.export.export(model, inputs)
                if functionalized:
                    program = program.run_decompositions({})
                before = str(program.graph)
                transformed = XnnpackPartitioner().transform_for_pre_decomposition(
                    program
                )
                self.assertEqual(str(program.graph), before)
                self.assertFalse(
                    any(
                        node.target == torch.ops.aten.cat.default
                        for node in transformed.graph.nodes
                    )
                )
                edge = to_edge_transform_and_lower(
                    transformed, partitioner=[XnnpackPartitioner()]
                )
                info = get_delegation_info(edge.exported_program().graph_module)
                self.assertEqual(
                    info.delegation_by_operator["aten_linear_default"].non_delegated, 0
                )
                runtime = _load_for_executorch_from_buffer(edge.to_executorch().buffer)
                for _ in range(2):
                    for actual, expected in zip(
                        runtime.forward(inputs), model(*inputs)
                    ):
                        torch.testing.assert_close(
                            actual, expected, atol=1e-5, rtol=1e-5
                        )

    def test_user_input_mutation_does_not_block_constant_chain(self):
        class InputMutation(ProjectionBiases):
            def forward(self, value, bias):
                value.add_(1)
                return super().forward(value, bias)

        model = InputMutation("chunk").eval()
        inputs = (torch.randn(2, 8), torch.randn(8))
        program = torch.export.export(model, inputs).run_decompositions({})
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertFalse(
            any(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            )
        )
        value = inputs[0].clone()
        for actual, expected in zip(
            transformed.module()(value, inputs[1]), model(inputs[0].clone(), inputs[1])
        ):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(value, inputs[0] + 1)

    def test_mutated_alias_keeps_only_affected_chain_dynamic(self):
        class AliasedState(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8))
                self.register_buffer("alias", self.state[:])
                self.register_buffer("safe", torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                self.state.add_(1)
                affected = torch.cat((self.alias, self.alias)).chunk(2)[0]
                safe = torch.cat((self.safe, self.safe)).chunk(2)[0]
                return value + affected + safe

        model = AliasedState()
        inputs = (torch.zeros(8),)
        program = torch.export.export(model, inputs)
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertEqual(
            sum(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            ),
            1,
        )
        module = transformed.module()
        for step in (1, 2):
            torch.testing.assert_close(
                module(*inputs), torch.arange(8, dtype=torch.float32) + step
            )

    def test_external_constants_exclude_only_dependent_chains(self):
        class MixedStorage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.external = torch.nn.Parameter(torch.ones(8))
                self.internal = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                external = torch.cat((self.external, self.external)).chunk(2)[0]
                internal = torch.cat((self.internal, self.internal)).chunk(2)[0]
                return value + external + internal

        model = MixedStorage().eval()
        inputs = (torch.randn(8),)
        program = torch.export.export(model, inputs)
        external = next(
            node for node in program.graph.nodes if node.name == "p_external"
        )
        external.meta["custom"] = {"delegate_constant_tag": "weights.ptd"}
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        cats = [
            node
            for node in transformed.graph.nodes
            if node.target == torch.ops.aten.cat.default
        ]
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0].args[0][0].name, "p_external")
        torch.testing.assert_close(transformed.module()(*inputs), model(*inputs))

    def test_external_linear_weight_keeps_storage_while_bias_folds(self):
        class ExternalWeight(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8, 8))
                self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

            def forward(self, value):
                bias = torch.cat((self.bias, self.bias)).chunk(2)[0]
                return F.linear(value, self.weight, bias)

        model = ExternalWeight().eval()
        inputs = (torch.randn(2, 8),)
        program = torch.export.export(model, inputs)
        weight = next(node for node in program.graph.nodes if node.name == "p_weight")
        weight.meta["custom"] = {"delegate_constant_tag": "weights.ptd"}
        transformed = XnnpackPartitioner().transform_for_pre_decomposition(program)
        self.assertFalse(
            any(
                node.target == torch.ops.aten.cat.default
                for node in transformed.graph.nodes
            )
        )
        edge = to_edge_transform_and_lower(
            transformed, partitioner=[XnnpackPartitioner()]
        )
        executable = edge.to_executorch()
        runtime = _load_for_executorch_from_buffer(
            executable.buffer, bytes(executable._tensor_data["weights.ptd"])
        )
        torch.testing.assert_close(
            runtime.forward(inputs)[0], model(*inputs), atol=1e-5, rtol=1e-5
        )

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
