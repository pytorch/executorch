# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.passes.const_prop_pass import ConstPropPass
from executorch.exir.schema import Tensor, TensorList

from executorch.exir.verification.interpreter import Interpreter
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
from torch._export.verifier import SpecViolationError
from torch.export import export


class WrapperModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TestVerification(unittest.TestCase):
    def test_constant_buffer(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.ones(2) + x + torch.ones(2)

        # Generate program
        program = (
            to_edge(export(WrapperModule(f), (torch.randn(2),)))
            .transform(
                [
                    ConstPropPass(),
                ]
            )
            .to_executorch()
            ._emitter_output.program
        )

        test = Interpreter(program)
        for val_idx in range(len(test.execution_plan.values)):
            val = test.execution_plan.values[val_idx].val
            if not (
                isinstance(val, Tensor) and val.data_buffer_idx == 0
            ) and not isinstance(val, TensorList):
                test.load_value(val_idx)
        vlist = test.get_value_list()
        for e in vlist:
            if isinstance(e, torch.Tensor):
                self.assertTrue(torch.allclose(e, torch.ones(2)))

        # asserting only 2 constant Tensors exist in value list
        self.assertEqual(len([e for e in vlist if isinstance(e, torch.Tensor)]), 2)

    def test_operator_list(self) -> None:
        class Op1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(2, 2)
                self.b = 2 * torch.ones(2, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for _ in range(10):
                    z = self.a * x  # mul
                    y = z - self.b  # sub
                return y

        class Op2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(2, 2)
                self.b = 2 * torch.ones(2, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for _ in range(10):
                    z = self.a % x  # remainder
                    y = z / self.b  # div
                    z = z + z  # add
                return y + z

        # Generate a program with Op1's operations (mul, sub)
        model1 = Op1()
        inputs = (torch.ones(2, 2),)
        program = (
            to_edge(export(model1, inputs)).to_executorch()._emitter_output.program
        )

        # Initialize and test Interpreter -- assert that the operators are same as above
        test = Interpreter(program)
        self.assertEqual(
            set(test.get_operators_list()),
            {torch.ops.aten.mul.out, torch.ops.aten.sub.out},
        )

        # Generate a program with Op2's operations (remainder, div, add_, add)
        model2 = Op2()
        inputs = (torch.ones(2, 2),)
        program = (
            to_edge(export(model2, inputs)).to_executorch()._emitter_output.program
        )

        # Initialize and test Interpreter -- assert that the operators are same as above
        test = Interpreter(program)
        self.assertEqual(
            set(test.get_operators_list()),
            {
                torch.ops.aten.remainder.Tensor_out,
                torch.ops.aten.div.out,
                torch.ops.aten.add.out,
            },
        )

    def test_verification(self) -> None:
        class Op2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(2, 2)
                self.b = 2 * torch.ones(2, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for _ in range(10):
                    z = self.a % x  # remainder
                    y = z / self.b  # div
                    z = z + z  # add
                return y + z

        # Generate a program with Op2's operations (remainder, div, add)
        model2 = Op2()
        inputs = torch.ones(2, 2)
        exec_prog = to_edge(export(model2, (inputs,))).to_executorch()

        exported_prog = exec_prog.exported_program()
        res = exported_prog.module()(inputs)[0]  # noqa
        # Verifiers are run internally in to_edge, export, and to_executorch.
        # If we make it this far then no errors were thrown in verification


class TestEdgeVerification(unittest.TestCase):
    def test_edge_happy(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.randn(1, 3, 100, 100))

            def forward(self, x):
                b = self.a + x
                return torch._to_cpu([b, x])

        m = TestModel()
        egm = (
            to_edge(
                export(
                    m,
                    (torch.randn(1, 3, 100, 100).to(dtype=torch.int),),
                )
            )
            .exported_program()
            .graph_module
        )
        verifier = EXIREdgeDialectVerifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    def test_edge_happy_with_optional_tensor_input(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, bias):
                # weight and bias here are optional tensor inputs.
                return torch.group_norm(x, 4, weight, bias)

        m = TestModel()
        egm = (
            to_edge(
                export(
                    m,
                    (torch.rand(16, 8, 32, 32), torch.rand(8), torch.rand(8)),
                )
            )
            .exported_program()
            .graph_module
        )
        verifier = EXIREdgeDialectVerifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    def test_edge_happy_with_empty_tensorlist_input(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch._to_cpu(x)

        m = TestModel()
        egm = (
            to_edge(
                export(
                    m,
                    ([],),
                )
            )
            .exported_program()
            .graph_module
        )
        verifier = EXIREdgeDialectVerifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    def test_edge_sad(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.randn(1, 3, 100, 100))

            def forward(self, x):
                b = self.a + x
                return torch._to_cpu([b, x])

        m = TestModel()
        egm = export(
            m,
            (torch.randn(1, 3, 100, 100).to(dtype=torch.int),),
        ).graph_module
        verifier = EXIREdgeDialectVerifier()
        with self.assertRaises(SpecViolationError):
            verifier(egm)

    def test_edge_happy_with_edge_ops(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        m = TestModel()
        egm = (
            to_edge(
                export(
                    m,
                    (torch.randn(1, 3, 100, 100).to(dtype=torch.int),),
                )
            )
            .exported_program()
            .graph_module
        )
        verifier = EXIREdgeDialectVerifier()
        verifier(egm)
        self.assertTrue(verifier.is_valid(egm))

    def test_edge_sad_with_edge_ops(self) -> None:
        # log_softmax only takes float or double Tensor
        m = torch.nn.LogSoftmax(dim=1)
        with self.assertRaises(SpecViolationError):
            _ = (
                to_edge(
                    export(
                        m,
                        (torch.randn(1, 3, 100, 100).to(dtype=torch.bfloat16),),
                    )
                )
                .exported_program()
                .graph_module
            )
