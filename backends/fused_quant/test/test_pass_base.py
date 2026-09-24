# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import cast

import torch
from executorch.backends.fused_quant.pass_base import IterativePassGroup
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from parameterized import parameterized
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import TensorArgument


class _RemoveOutputPermutes(ExportedProgramPassBase):
    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for permute in graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.permute_copy.default,
        ):
            inp = permute.args[0]
            assert isinstance(inp, fx.Node)
            permute.replace_all_uses_with(inp)
            graph.erase_node(permute)
            modified = True
        return ExportedProgramPassResult(exported_program, modified)

    def ensures(self, exported_program: ExportedProgram) -> None:
        exported_program.validate()


class _NoOp(ExportedProgramPassBase):
    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        return ExportedProgramPassResult(exported_program, False)


class ExportedProgramPassBaseTest(unittest.TestCase):
    def test_syncs_output_specs_before_postcondition(self) -> None:
        builder = ProgramBuilder()
        first = builder.placeholder("first", torch.randn(2, 3))
        second = builder.placeholder("second", torch.randn(4, 5))
        first_permute = builder.call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (first, [1, 0]),
        )
        second_permute = builder.call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (second, [1, 0]),
        )
        builder.output([first_permute, second_permute, first_permute])
        exported_program = builder.get_edge_program().exported_program()

        result = _RemoveOutputPermutes()(exported_program)

        self.assertTrue(result.modified)
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            (first.node.name, second.node.name, first.node.name),
        )
        result.exported_program.validate()

    def test_signature_only_fix_is_reported_as_modified(self) -> None:
        builder = ProgramBuilder()
        inp = builder.placeholder("inp", torch.randn(2, 3))
        permute = builder.call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (inp, [1, 0]),
        )
        builder.output([permute])
        exported_program = builder.get_edge_program().exported_program()
        exported_program.graph_signature.output_specs[0].arg = TensorArgument(
            inp.node.name
        )

        result = _NoOp()(exported_program)

        self.assertTrue(result.modified)
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            (permute.node.name,),
        )
        result.exported_program.validate()


class _NoopEPPass(ExportedProgramPassBase):
    """EP pass that marks modified without changing the graph."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        return ExportedProgramPassResult(exported_program, True)


class _FuseFirstMulPair(ExportedProgramPassBase):
    """Fuses the first pair of consecutive muls it finds into one.

    Looks for mul(mul(a, b), c) and replaces it with mul(a, b * c),
    folding the constant operand.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mul.Tensor
        ):
            inp = node.args[0]
            if not isinstance(inp, torch.fx.Node):
                continue
            if inp.target != exir_ops.edge.aten.mul.Tensor:
                continue
            if len(inp.users) != 1:
                continue
            node.replace_input_with(inp, inp.args[0])
            node.args = (
                node.args[0],
                cast(float, inp.args[1]) * cast(float, node.args[1]),
            )
            graph.erase_node(inp)
            exported_program.graph_module.recompile()
            return ExportedProgramPassResult(exported_program, True)
        return ExportedProgramPassResult(exported_program, False)


def _build_mul_chain(n: int) -> ExportedProgram:
    """Build a graph: x * 2 * 3 * 4 * ... * (n+1)."""
    builder = ProgramBuilder()
    x = builder.placeholder("x", torch.randn(2, 4))
    cur = x
    for i in range(n):
        cur = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(cur, float(i + 2)),
        )
    builder.output([cur])
    return builder.get_program()


class IterativePassGroupTest(unittest.TestCase):
    @parameterized.expand([(1,), (2,), (3,)])
    def test_steps_reduce_mul_count(self, steps: int) -> None:
        initial_muls = 4
        ep = _build_mul_chain(initial_muls)

        mul_count_before = len(
            ep.graph_module.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.mul.Tensor
            )
        )
        self.assertEqual(mul_count_before, initial_muls)

        group = IterativePassGroup([_FuseFirstMulPair()], steps=steps)
        result = group.call(ep)
        self.assertTrue(result.modified)

        mul_count_after = len(
            result.exported_program.graph_module.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.mul.Tensor
            )
        )
        self.assertEqual(
            mul_count_before - mul_count_after,
            steps,
            f"Expected {steps} muls to be fused, but went from {mul_count_before} to {mul_count_after}",
        )

    def test_nested_error_message(self) -> None:
        """A failing pass inside a nested IterativePassGroup should produce
        a chained exception that names the exact pass that failed."""

        class _FailingPass(ExportedProgramPassBase):
            def call(
                self, exported_program: ExportedProgram
            ) -> ExportedProgramPassResult:
                raise RuntimeError("intentional failure")

        ep = _build_mul_chain(2)

        inner_group = IterativePassGroup([_NoopEPPass(), _FailingPass()], steps=1)
        outer_group = IterativePassGroup([_NoopEPPass(), inner_group], steps=1)

        with self.assertRaises(Exception) as ctx:
            outer_group.call(ep)

        # Walk the exception chain to collect all messages
        messages: list[str] = []
        exc: BaseException | None = ctx.exception
        while exc is not None:
            messages.append(str(exc))
            exc = exc.__cause__

        full_trace = "\n".join(messages)
        self.assertIn(
            "_FailingPass",
            full_trace,
            f"Exception chain should name _FailingPass:\n{full_trace}",
        )
        self.assertIn(
            "IterativePassGroup",
            full_trace,
            f"Exception chain should name IterativePassGroup:\n{full_trace}",
        )
        self.assertIn(
            "intentional failure",
            full_trace,
            f"Exception chain should include the original error:\n{full_trace}",
        )
