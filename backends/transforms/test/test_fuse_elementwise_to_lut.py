# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.fuse_elementwise_to_lut import (
    FuseElementwiseToLUTPass,
)
from executorch.backends.transforms.utils import (
    _validate_graph_signature,
    create_constant_placeholder,
    get_param_tensor,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)


@torch.library.custom_op("test_elementwise_lut::lookup_table", mutates_args=())
def _lookup_table(input: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table[input.to(torch.int64) + 128]


@_lookup_table.register_fake
def _lookup_table_fake(input, table):
    return torch.empty_like(input)


def _fusion_pass(program, table_op=None, **kwargs):
    if table_op is None:
        table_op = exir_ops.edge.test_elementwise_lut.lookup_table.default
    return FuseElementwiseToLUTPass(
        program,
        table_op,
        {
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.minimum.default,
            exir_ops.edge.aten.maximum.default,
        },
        **kwargs,
    )


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _TwoInputs(torch.nn.Module):
    def forward(self, x, y):
        return x, y


def _program(two_inputs=False):
    model = _TwoInputs() if two_inputs else _Identity()
    inputs = (torch.arange(-128, 128, dtype=torch.int8),)
    if two_inputs:
        inputs += (torch.arange(-128, 128, dtype=torch.int8),)
    program = to_edge(export(model, inputs)).exported_program()
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    return program, placeholders


def _table(program, target, input_node, values, name):
    graph = program.graph
    first = next(node for node in graph.nodes if node.op == "placeholder")
    with graph.inserting_before(first):
        constant = create_constant_placeholder(
            program,
            graph,
            name,
            InputKind.BUFFER,
            values,
            persistent_buffer=True,
        )
    output = graph.output_node()
    with graph.inserting_before(output):
        node = graph.call_function(target, (input_node, constant))
    node.meta["val"] = input_node.meta["val"]
    return node


def _constant(program, values, name):
    first = next(node for node in program.graph.nodes if node.op == "placeholder")
    with program.graph.inserting_before(first):
        return create_constant_placeholder(
            program,
            program.graph,
            name,
            InputKind.BUFFER,
            values,
            persistent_buffer=True,
        )


def _set_output(program, node):
    program.graph.output_node().args = ((node,),)
    program._graph_signature = ExportGraphSignature(
        program.graph_signature.input_specs,
        [OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(node.name), None)],
    )


class FuseElementwiseToLUTTests(unittest.TestCase):
    def test_linear_chain_with_configured_table_target(self):
        domain = torch.arange(-128, 128, dtype=torch.int8)
        for target in (
            torch.ops.test_elementwise_lut.lookup_table.default,
            exir_ops.edge.test_elementwise_lut.lookup_table.default,
        ):
            with self.subTest(target=target):
                program, (source,) = _program()
                first_values = torch.bitwise_xor(
                    domain, torch.tensor(0x55, dtype=torch.int8)
                )
                second_values = torch.flip(domain, (0,))
                first = _table(program, target, source, first_values, "first_lut")
                with program.graph.inserting_before(program.graph.output_node()):
                    clamp = program.graph.call_function(
                        exir_ops.edge.aten.clamp.default, (first, -20, 30)
                    )
                clamp.meta["val"] = source.meta["val"]
                last = _table(program, target, clamp, second_values, "last_lut")
                _set_output(program, last)

                result = _fusion_pass(program, target).call(program.graph_module)

                self.assertTrue(result.modified)
                tables = [
                    node for node in program.graph.nodes if node.op == "call_function"
                ]
                self.assertEqual(len(tables), 1)
                self.assertIs(tables[0].target, target)
                self.assertIs(tables[0].args[0], source)
                expected = second_values[
                    torch.clamp(first_values, -20, 30).to(torch.int64) + 128
                ]
                torch.testing.assert_close(
                    get_param_tensor(program, tables[0].args[1]), expected
                )
                _validate_graph_signature(program)
                self.assertFalse(
                    _fusion_pass(program, target).call(program.graph_module).modified
                )

    def test_diamond_with_minimum(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        left_values = -domain
        right_values = torch.clamp(domain, -70, 70)
        left = _table(program, target, source, left_values, "left_lut")
        right = _table(program, target, source, right_values, "right_lut")
        with program.graph.inserting_before(program.graph.output_node()):
            minimum = program.graph.call_function(
                exir_ops.edge.aten.minimum.default, (left, right)
            )
        minimum.meta["val"] = source.meta["val"]
        _set_output(program, minimum)

        result = _fusion_pass(program).call(program.graph_module)

        self.assertTrue(result.modified)
        nodes = [node for node in program.graph.nodes if node.op == "call_function"]
        self.assertEqual(len(nodes), 1)
        torch.testing.assert_close(
            get_param_tensor(program, nodes[0].args[1]),
            torch.minimum(left_values, right_values),
        )
        _validate_graph_signature(program)

    def test_single_input_suffix_after_multiple_input_prefix(self):
        program, (left, right) = _program(two_inputs=True)
        with program.graph.inserting_before(program.graph.output_node()):
            minimum = program.graph.call_function(
                exir_ops.edge.aten.minimum.default, (left, right)
            )
            first = program.graph.call_function(
                exir_ops.edge.aten.clamp.default, (minimum, -70, 70)
            )
            last = program.graph.call_function(
                exir_ops.edge.aten.clamp.default, (first, -20, 30)
            )
        for node in (minimum, first, last):
            node.meta["val"] = left.meta["val"]
        _set_output(program, last)

        self.assertTrue(_fusion_pass(program).call(program.graph_module).modified)

        nodes = [node for node in program.graph.nodes if node.op == "call_function"]
        self.assertEqual(nodes, [minimum, last])
        self.assertIs(last.args[0], minimum)
        self.assertIs(
            last.target, exir_ops.edge.test_elementwise_lut.lookup_table.default
        )
        domain = torch.arange(-128, 128, dtype=torch.int8)
        torch.testing.assert_close(
            get_param_tensor(program, last.args[1]), torch.clamp(domain, -20, 30)
        )
        _validate_graph_signature(program)

    def test_scalar_constant_is_fused_and_retired(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        table = _table(program, target, source, domain, "scalar_table")
        constant = _constant(program, torch.tensor(7, dtype=torch.int8), "limit")
        with program.graph.inserting_before(program.graph.output_node()):
            minimum = program.graph.call_function(
                exir_ops.edge.aten.minimum.default, (table, constant)
            )
        minimum.meta["val"] = source.meta["val"]
        _set_output(program, minimum)

        result = _fusion_pass(program).call(program.graph_module)

        self.assertTrue(result.modified)
        self.assertNotIn("limit", program.state_dict)
        tables = [node for node in program.graph.nodes if node.op == "call_function"]
        torch.testing.assert_close(
            get_param_tensor(program, tables[0].args[1]),
            torch.minimum(domain, torch.tensor(7, dtype=torch.int8)),
        )
        _validate_graph_signature(program)

    def test_non_scalar_constant_does_not_fuse(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        table = _table(program, target, source, domain, "ranked_table")
        constant = _constant(
            program, torch.tensor([7, 8], dtype=torch.int8), "ranked_limit"
        )
        with program.graph.inserting_before(program.graph.output_node()):
            minimum = program.graph.call_function(
                exir_ops.edge.aten.minimum.default, (table, constant)
            )
        minimum.meta["val"] = source.meta["val"]
        _set_output(program, minimum)

        self.assertFalse(_fusion_pass(program).call(program.graph_module).modified)

    def test_mutated_buffer_is_not_treated_as_a_constant(self):
        class Mutable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("value", torch.zeros(256, dtype=torch.int8))

            def forward(self, x):
                self.value.add_(1)
                return x

        program = to_edge(
            export(Mutable(), (torch.arange(-128, 128, dtype=torch.int8),))
        ).exported_program()
        buffer = next(
            node
            for node in program.graph.nodes
            if node.op == "placeholder"
            and node.name in program.graph_signature.inputs_to_buffers
        )
        self.assertIsNone(_fusion_pass(program)._constant(buffer))

    def test_side_user_stops_cluster_at_boundary(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        first = _table(program, target, source, domain, "side_first")
        second = _table(program, target, first, torch.flip(domain, (0,)), "side_second")
        third = _table(program, target, second, domain, "side_third")
        program.graph.output_node().args = ((first, third),)

        result = _fusion_pass(program).call(program.graph_module)

        self.assertTrue(result.modified)
        tables = [node for node in program.graph.nodes if node.op == "call_function"]
        self.assertEqual(len(tables), 2)
        self.assertIs(tables[-1].args[0], first)
        torch.testing.assert_close(
            get_param_tensor(program, tables[-1].args[1]), torch.flip(domain, (0,))
        )

    def test_two_variable_sources_are_not_fused(self):
        program, (x, y) = _program(two_inputs=True)
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        left = _table(program, target, x, domain, "two_left")
        right = _table(program, target, y, domain, "two_right")
        with program.graph.inserting_before(program.graph.output_node()):
            minimum = program.graph.call_function(
                exir_ops.edge.aten.minimum.default, (left, right)
            )
        minimum.meta["val"] = x.meta["val"]
        _set_output(program, minimum)

        result = _fusion_pass(program).call(program.graph_module)

        self.assertFalse(result.modified)
        self.assertEqual(
            len([node for node in program.graph.nodes if node.op == "call_function"]),
            3,
        )

    def test_configured_pointwise_ops_fuse_without_an_existing_table(self):
        class NegateAndClamp(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(torch.neg(x), min=-20, max=30)

        domain = torch.arange(-128, 128, dtype=torch.int8)
        model = NegateAndClamp()
        program = to_edge(export(model, (domain,))).exported_program()
        transform = FuseElementwiseToLUTPass(
            program,
            exir_ops.edge.test_elementwise_lut.lookup_table.default,
            {exir_ops.edge.aten.neg.default, exir_ops.edge.aten.clamp.default},
        )

        self.assertTrue(transform.call(program.graph_module).modified)
        program.validate()
        torch.testing.assert_close(program.module()(domain), model(domain))
        self.assertEqual(
            sum(node.op == "call_function" for node in program.graph.nodes), 1
        )

    def test_one_element_tensor_arithmetic(self):
        class ScalarArithmetic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("constant", torch.tensor([[7]], dtype=torch.int8))

            def forward(self, x):
                return (x + self.constant) * self.constant - self.constant

        domain = torch.arange(-128, 128, dtype=torch.int8).reshape(1, 256)
        model = ScalarArithmetic()
        program = to_edge(export(model, (domain,))).exported_program()
        transform = FuseElementwiseToLUTPass(
            program,
            exir_ops.edge.test_elementwise_lut.lookup_table.default,
            {
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.sub.Tensor,
            },
        )

        self.assertTrue(transform.call(program.graph_module).modified)
        program.validate()
        torch.testing.assert_close(program.module()(domain), model(domain))
        self.assertNotIn("constant", program.state_dict)

    def test_unconfigured_operator_is_a_boundary(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        first = _table(program, target, source, domain, "boundary_first")
        with program.graph.inserting_before(program.graph.output_node()):
            bitwise_not = program.graph.call_function(
                exir_ops.edge.aten.bitwise_not.default, (first,)
            )
        bitwise_not.meta["val"] = source.meta["val"]
        second = _table(program, target, bitwise_not, domain, "boundary_second")
        _set_output(program, second)

        self.assertFalse(_fusion_pass(program).call(program.graph_module).modified)
        torch.testing.assert_close(program.module()(domain), torch.bitwise_not(domain))
        self.assertEqual(len(program.state_dict), 2)

    def test_region_filter_can_reject_fusion(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        first = _table(program, target, source, domain, "filter_first")
        second = _table(program, target, first, domain, "filter_second")
        _set_output(program, second)

        self.assertFalse(
            _fusion_pass(program, region_filter=lambda region: False)
            .call(program.graph_module)
            .modified
        )
        self.assertEqual(len(program.state_dict), 2)

    def test_invalid_operator_signature_is_not_silently_skipped(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        with program.graph.inserting_before(program.graph.output_node()):
            malformed = program.graph.call_function(target, (source,))
        malformed.meta["val"] = source.meta["val"]
        _set_output(program, malformed)

        with self.assertRaises(ValueError):
            _fusion_pass(program).call(program.graph_module)

    def test_kwargs_are_evaluated_in_the_lookup_table(self):
        program, (source,) = _program()
        target = exir_ops.edge.test_elementwise_lut.lookup_table.default
        domain = torch.arange(-128, 128, dtype=torch.int8)
        first = _table(program, target, source, domain, "kwargs_first")
        with program.graph.inserting_before(program.graph.output_node()):
            clamp = program.graph.call_function(
                exir_ops.edge.aten.clamp.default, (first,), {"min": -20, "max": 30}
            )
        clamp.meta["val"] = source.meta["val"]
        _set_output(program, clamp)

        self.assertTrue(_fusion_pass(program).call(program.graph_module).modified)
        torch.testing.assert_close(
            program.module()(domain), torch.clamp(domain, -20, 30)
        )


if __name__ == "__main__":
    unittest.main()
