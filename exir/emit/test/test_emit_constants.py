# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.emit._emitter import _Emitter, _EmitterState, _ProgramState
from executorch.exir.schema import Bool, Double, EValue, Int, IntList
from executorch.exir.tensor import TensorSpec
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch._higher_order_ops import cond, map as torch_map
from torch.export import export


class TestEmitConstants(unittest.TestCase):
    def make_emitter(self, state=None):
        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        graph.output(node)
        module = torch.fx.GraphModule({}, graph)
        module.meta["non_const_buffer_sizes"] = [0, 0]
        if state is None:
            state = _EmitterState([], [], [], {}, False, False)
        emitter = _Emitter(module, state, _ProgramState())
        emitter.node = node
        return emitter

    def test_scalar_types_and_signed_int64(self):
        emitter = self.make_emitter()
        for value in (-(2**63), -128, 0, 1, 2**63 - 1):
            with self.subTest(value=value):
                first = emitter._emit_argument(value, None, immutable=True)
                second = emitter._emit_argument(value, None, immutable=True)
                self.assertEqual(first.id, second.id)
                self.assertEqual(emitter.emitter_state.values[first.id].val, Int(value))

        integer = emitter._emit_argument(1, None, immutable=True)
        boolean = emitter._emit_argument(True, None, immutable=True)
        double = emitter._emit_argument(1.0, None, immutable=True)
        self.assertEqual(len({integer.id, boolean.id, double.id}), 3)
        self.assertEqual(emitter.emitter_state.values[boolean.id].val, Bool(True))
        self.assertEqual(emitter.emitter_state.values[double.id].val, Double(1.0))

    def test_literal_lists_and_element_references(self):
        emitter = self.make_emitter()
        list_type = torch.ListType.ofInts()
        values = emitter.emitter_state.values
        scalar = emitter._emit_argument(1, None, immutable=True)
        pair = emitter._emit_argument([1, 1], list_type, immutable=True)
        repeated = emitter._emit_argument((1, 1), list_type, immutable=True)
        self.assertEqual(pair.id, repeated.id)
        self.assertEqual(values[pair.id].val, IntList([scalar.id, scalar.id]))
        for items in ([0, 1], [1, 0], [1], []):
            with self.subTest(items=items):
                value = emitter._emit_argument(items, list_type, immutable=True)
                optional = emitter._emit_argument(
                    items, torch.OptionalType(list_type), immutable=True
                )
                self.assertEqual(value.id, optional.id)
                self.assertNotEqual(value.id, pair.id)
                self.assertEqual(
                    [values[index].val.int_val for index in values[value.id].val.items],
                    items,
                )
        self.assertEqual(sum(isinstance(value.val, Int) for value in values), 2)
        self.assertEqual(sum(isinstance(value.val, IntList) for value in values), 5)

    def test_serialized_signed_int64(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.clamp(x, min=-(2**63), max=2**63 - 1),
                    torch.clamp(x.flip(0), min=-(2**63), max=2**63 - 1),
                )

        model = Model()
        inputs = (torch.tensor([-(2**63), 0, 2**63 - 1]),)
        program = to_edge(export(model, inputs, strict=True)).to_executorch()
        plan = deserialize_pte_binary(program.buffer).program.execution_plan[0]
        for limit in (-(2**63), 2**63 - 1):
            self.assertEqual(sum(value.val == Int(limit) for value in plan.values), 1)
        runtime = _load_for_executorch_from_buffer(program.buffer)
        for actual, expected in zip(runtime.forward(inputs), model(*inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dynamic_lists_preserve_mutable_elements(self):
        emitter = self.make_emitter()
        values = emitter.emitter_state.values
        symbol = emitter._emit_evalue(EValue(Int(1)))
        literal = emitter._emit_argument(1, None, immutable=True)
        self.assertNotEqual(symbol.id, literal.id)
        self.assertEqual(
            emitter._emit_argument(symbol, None, immutable=True).id, symbol.id
        )
        lists = [
            emitter._emit_argument([symbol, 1], torch.ListType.ofInts(), immutable=True)
            for _ in range(2)
        ]
        self.assertNotEqual(lists[0].id, lists[1].id)
        for value in lists:
            self.assertEqual(values[value.id].val.items, [symbol.id, literal.id])
        values[symbol.id] = EValue(Int(7))
        self.assertEqual(values[literal.id].val, Int(1))
        for value in lists:
            self.assertEqual(
                [values[index].val.int_val for index in values[value.id].val.items],
                [7, 1],
            )

    def test_independent_dynamic_batches(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.ones((x.shape[0], 1)), torch.zeros((y.shape[0], 1))

        model = Model()
        inputs = (torch.zeros(2, 3), torch.zeros(2, 3))
        program = to_edge(
            export(
                model,
                inputs,
                dynamic_shapes={
                    "x": {0: torch.export.Dim("batch_x", min=1, max=5)},
                    "y": {0: torch.export.Dim("batch_y", min=1, max=5)},
                },
                strict=True,
            )
        ).to_executorch()
        plan = deserialize_pte_binary(program.buffer).program.execution_plan[0]
        shape_ids = [
            instruction.instr_args.args[0]
            for instruction in plan.chains[0].instructions
            if plan.operators[instruction.instr_args.op_index].name == "aten::full"
        ]
        self.assertEqual(len(shape_ids), 2)
        self.assertNotEqual(shape_ids[0], shape_ids[1])
        shapes = [plan.values[index].val.items for index in shape_ids]
        self.assertEqual([len(shape) for shape in shapes], [2, 2])
        self.assertNotEqual(shapes[0][0], shapes[1][0])
        self.assertEqual(shapes[0][1], shapes[1][1])
        self.assertNotIn(shapes[0][1], (shapes[0][0], shapes[1][0]))
        self.assertEqual(plan.values[shapes[0][1]].val, Int(1))

        runtime = _load_for_executorch_from_buffer(program.buffer)
        for batch_x, batch_y in ((2, 2), (5, 2), (5, 3), (1, 3), (1, 1), (2, 2)):
            with self.subTest(batch_x=batch_x, batch_y=batch_y):
                inputs = (torch.zeros(batch_x, 3), torch.zeros(batch_y, 3))
                outputs = runtime.forward(inputs)
                self.assertEqual(len(outputs), 2)
                for actual, expected in zip(outputs, model(*inputs)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_opaque_arguments_do_not_enter_pool(self):
        emitter = self.make_emitter()
        values = emitter.emitter_state.values
        for arg, arg_type in ((1, None), ([1, 1], torch.ListType.ofInts())):
            with self.subTest(arg=arg):
                first = emitter._emit_argument(arg, arg_type)
                second = emitter._emit_argument(arg, arg_type)
                pooled = emitter._emit_argument(arg, arg_type, immutable=True)
                self.assertEqual(len({first.id, second.id, pooled.id}), 3)
                if isinstance(arg, list):
                    self.assertTrue(
                        set(values[first.id].val.items).isdisjoint(
                            values[pooled.id].val.items
                        )
                    )

    def test_operator_alias_and_mutation_boundaries(self):
        emitter = self.make_emitter()
        emitter.node.meta["spec"] = TensorSpec.from_tensor(torch.ones(2))
        tensor = emitter._emit_spec(emitter.node.meta["spec"])
        with torch.library._scoped_library("emit_constant_test", "FRAGMENT") as library:
            for name, scalar, items in (
                ("read", "int", "int[]"),
                ("alias", "int(a)", "int[](b)"),
                ("mutate", "int(a!)", "int[](b!)"),
            ):
                library.define(
                    f"{name}.out(Tensor x, {scalar} scalar, {items} items, "
                    "*, Tensor(c!) out) -> Tensor(c!)"
                )
            for op, should_pool in (
                (torch.ops.emit_constant_test.read.out, True),
                (torch.ops.emit_constant_test.alias.out, False),
                (torch.ops.emit_constant_test.mutate.out, False),
            ):
                with self.subTest(op=op):
                    for _ in range(2):
                        emitter._emit_operator(op, (tensor, 1, [1, 1]), {"out": tensor})
                    first, second = [
                        instruction.instr_args.args
                        for instruction in emitter.chain.instructions[-2:]
                    ]
                    for index in (1, 2):
                        self.assertEqual(first[index] == second[index], should_pool)
                    if not should_pool:
                        items = emitter.emitter_state.values[first[2]].val.items
                        self.assertNotIn(first[1], items)
                        self.assertTrue(
                            set(items).isdisjoint(
                                emitter.emitter_state.values[second[2]].val.items
                            )
                        )

    def test_input_output_and_nested_containers(self):
        class Model(torch.nn.Module):
            def forward(self, x, number, flag):
                return {"tensors": [x + 1, x - 1], "constants": (1, [number, flag])}

        inputs = (torch.ones(2), 1, True)
        program = to_edge(export(Model(), inputs, strict=True)).to_executorch()
        plan = deserialize_pte_binary(program.buffer).program.execution_plan[0]
        input_int = plan.inputs[1]
        self.assertEqual(plan.values[input_int].val, Int(1))
        self.assertEqual(plan.values[plan.inputs[2]].val, Bool(True))
        literal_ids = [
            index
            for instruction in plan.chains[0].instructions
            for index in instruction.instr_args.args
            if plan.values[index].val == Int(1)
        ]
        self.assertGreaterEqual(len(literal_ids), 2)
        self.assertEqual(len(set(literal_ids)), 1)
        self.assertNotIn(input_int, literal_ids)
        self.assertTrue(set(plan.outputs).isdisjoint(literal_ids))
        runtime = _load_for_executorch_from_buffer(program.buffer)
        result = runtime.forward(inputs)
        torch.testing.assert_close(result[0], inputs[0] + 1, rtol=0, atol=0)
        torch.testing.assert_close(result[1], inputs[0] - 1, rtol=0, atol=0)
        self.assertEqual(result[2:], [1, 1, True])

    def test_pool_scope(self):
        emitter = self.make_emitter()
        first = emitter._emit_argument(7, None, immutable=True)
        subgraph = self.make_emitter(emitter.emitter_state)
        self.assertEqual(subgraph._emit_argument(7, None, immutable=True).id, first.id)
        other_method = self.make_emitter()
        other_method._emit_evalue(EValue(Int(0)))
        other = other_method._emit_argument(7, None, immutable=True)
        self.assertNotEqual(other.id, first.id)
        self.assertEqual(other_method.emitter_state.values[other.id].val, Int(7))

    def test_repeated_control_flow_execution(self):
        class Model(torch.nn.Module):
            def forward(self, pred, xs):
                def body(x):
                    return cond(pred, lambda x: x + 1, lambda x: x - 1, (x,))

                return torch_map(body, xs)

        model = Model()
        xs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        program = to_edge(
            export(model, (torch.tensor(True), xs), strict=True)
        ).to_executorch()
        runtime = _load_for_executorch_from_buffer(program.buffer)
        for pred in (True, False, True):
            inputs = (torch.tensor(pred), xs)
            torch.testing.assert_close(
                runtime.forward(inputs)[0], model(*inputs), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
