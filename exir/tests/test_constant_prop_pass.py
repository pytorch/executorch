# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import hashlib
import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.passes.constant_prop_pass import _expression, constant_prop_pass
from torch.export import export


class TestConstantPropPass(unittest.TestCase):
    def test_constant_prop_pass_folds_through_a_scalar(self) -> None:
        """
        A Python scalar in the middle of a constant chain, the float of
        aten.item, is not a tensor to lift, but its consumers fold with it:
        `b * w.max().item()` comes out as one constant, and neither the op
        nor the parameters stay in the graph.
        """

        class ScaleByMaxItem(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = torch.nn.Parameter(torch.randn(4))
                self.w = torch.nn.Parameter(torch.randn(4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.b * self.w.max().item()

        x = torch.randn(4)
        ep = to_edge(export(ScaleByMaxItem(), (x,), strict=True)).exported_program()
        expected = ep.module()(x)
        new_ep = constant_prop_pass(ep)

        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertEqual(targets, [exir_ops.edge.aten.add.Tensor])
        self.assertEqual(list(new_ep.constants), ["_prop_tensor_constant0"])
        self.assertEqual(list(new_ep.graph_signature.inputs_to_parameters), [])
        self.assertTrue(torch.equal(new_ep.module()(x), expected))

    def test_constant_prop_pass_keeps_a_scalar_op_used_outside_the_fold(self) -> None:
        """
        A scalar cannot become a placeholder. When a consumer outside the
        fold takes it, the op stays in the graph with the parameter it reads,
        while the consumers inside the fold are still folded.
        """

        class ScaleByItem(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4))
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                scale = self.scale.item()
                return x * scale + self.weight * scale

        x = torch.ones(4)
        new_ep = constant_prop_pass(export(ScaleByItem(), (x,), strict=True))

        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertEqual(
            targets,
            [
                torch.ops.aten.item.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.add.Tensor,
            ],
        )
        self.assertEqual(list(new_ep.constants), ["_prop_tensor_constant0"])
        self.assertEqual(
            list(new_ep.graph_signature.inputs_to_parameters.values()), ["scale"]
        )
        self.assertTrue(torch.equal(new_ep.module()(x), x * 2 + 2))

    def test_constant_prop_pass_fold_buffers_false(self) -> None:
        """
        A buffer this program only reads can be written by another method of
        the same program, which the pass cannot see. With fold_buffers=False
        only parameters and lifted constants seed the fold.
        """

        class ParamAndBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4))
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.weight * 2 + self.state.sum()

        x = torch.zeros(4)
        new_ep = constant_prop_pass(
            export(ParamAndBuffer(), (x,), strict=True), fold_buffers=False
        )

        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertNotIn(torch.ops.aten.mul.Tensor, targets)
        self.assertIn(torch.ops.aten.sum.default, targets)
        self.assertEqual(list(new_ep.graph_signature.inputs_to_parameters), [])
        self.assertEqual(
            list(new_ep.graph_signature.inputs_to_buffers.values()), ["state"]
        )
        self.assertEqual(list(new_ep.constants), ["_prop_tensor_constant0"])
        self.assertTrue(torch.equal(new_ep.module()(x), x + 2))

    def test_constant_prop_pass_registers_fold_like_its_source(self) -> None:
        """
        With register_like_source a folded value takes the kind and the
        custom meta of the placeholder it is computed from, and a name made
        of that placeholder and the expression. Before decomposition aten.t
        returns a view of the parameter, which keeps requires_grad: the
        registered value has to be a detached leaf.
        """

        class MatmulT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(3, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.w.t() + x @ (self.w * 2).t()

        x = torch.randn(2, 4)
        ep = export(MatmulT(), (x,), strict=True)
        expected = ep.module()(x)
        ep.graph.find_nodes(op="placeholder", target="p_w")[0].meta["custom"] = {
            "delegate_constant_tag": "w.ptd"
        }

        new_ep = constant_prop_pass(ep, register_like_source=True)
        new_ep._validate()

        names = list(new_ep.graph_signature.inputs_to_parameters.values())
        self.assertEqual(len(names), 2)
        for name in names:
            self.assertRegex(name, r"^w_prop_[0-9a-f]{64}$")
        self.assertNotEqual(names[0], names[1])
        self.assertNotIn("w", new_ep.state_dict)
        self.assertEqual(len(new_ep.constants), 0)
        for name in names:
            folded = new_ep.state_dict[name]
            self.assertIsInstance(folded, torch.nn.Parameter)
            self.assertFalse(folded.requires_grad)
            self.assertTrue(folded.is_leaf)
        for node in new_ep.graph.find_nodes(op="placeholder"):
            if node.name != "x":
                self.assertEqual(
                    node.meta["custom"], {"delegate_constant_tag": "w.ptd"}
                )
        self.assertTrue(torch.allclose(new_ep.module()(x), expected))

    def test_constant_prop_pass_registers_buffer_fold_as_buffer(self) -> None:
        class ScaledBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor([1.0, 2.0]))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * (self.scale * 2)

        x = torch.ones(2)
        new_ep = constant_prop_pass(
            export(ScaledBuffer(), (x,), strict=True), register_like_source=True
        )
        new_ep._validate()

        buffers = list(new_ep.graph_signature.inputs_to_buffers.values())
        self.assertEqual(len(buffers), 1)
        self.assertRegex(buffers[0], r"^scale_prop_[0-9a-f]{64}$")
        self.assertEqual(len(new_ep.constants), 0)
        self.assertTrue(torch.equal(new_ep.module()(x), torch.tensor([2.0, 4.0])))

    def test_constant_prop_pass_names_folds_after_their_expression(self) -> None:
        """
        With register_like_source the name of a folded value depends on the
        source and on the expression, not on the method: two methods folding
        the same expression over the same parameter produce the same name,
        and two expressions over one parameter produce different names. The
        external constant map is keyed by the name and shared by the methods.
        """

        class Transpose(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.w.t()

        class ScaledTranspose(Transpose):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ (self.w * 2).t()

        x = torch.ones(2, 4)
        names = {}
        for method, module in (
            ("a", Transpose()),
            ("b", Transpose()),
            ("c", ScaledTranspose()),
        ):
            ep = constant_prop_pass(
                export(module, (x,), strict=True), register_like_source=True
            )
            (names[method],) = ep.graph_signature.inputs_to_parameters.values()
        self.assertEqual(names["a"], names["b"])
        self.assertNotEqual(names["a"], names["c"])
        for name in names.values():
            self.assertRegex(name, r"^w_prop_[0-9a-f]{64}$")

    def test_constant_prop_pass_describes_each_producer_once(self) -> None:
        """
        The expression that names a folded value describes every producer
        once, so it grows with the number of producers, not with the number
        of paths through them: a chain of squarings, where each level uses
        its input twice, stays short. It is built without recursion, so a
        deep chain does not reach the recursion limit. Each target appears
        with its namespace, and the name of the folded value carries the
        whole sha256 digest of the expression.
        """

        class Squaring(torch.nn.Module):
            def __init__(self, depth: int) -> None:
                super().__init__()
                self.depth = depth
                self.w = torch.nn.Parameter(torch.full((4, 4), 0.5))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = self.w
                for _ in range(self.depth):
                    w = w * w
                return x @ w

        class Chain(torch.nn.Module):
            def __init__(self, depth: int) -> None:
                super().__init__()
                self.depth = depth
                self.w = torch.nn.Parameter(torch.zeros(4, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = self.w
                for _ in range(self.depth):
                    w = w + 1.0
                return x @ w

        def weight_expression(ep):
            (matmul,) = ep.graph.find_nodes(
                op="call_function", target=torch.ops.aten.matmul.default
            )
            return _expression(ep, matmul.args[1])

        x = torch.randn(2, 4)
        expressions = [
            weight_expression(export(Squaring(16), (x,), strict=True)) for _ in range(2)
        ]
        self.assertEqual(expressions[0], expressions[1])
        self.assertLess(len(expressions[0]), 100 * 16)
        self.assertLess(
            len(weight_expression(export(Chain(600), (x,), strict=True))), 100 * 600
        )

        for module, target in (
            (Squaring(16), "aten.mul.Tensor"),
            (Chain(600), "aten.add.Tensor"),
        ):
            ep = export(module, (x,), strict=True)
            expected = ep.module()(x)
            expression = weight_expression(ep)
            self.assertIn(f"={target}(", expression)
            new_ep = constant_prop_pass(ep, register_like_source=True)
            (name,) = new_ep.graph_signature.inputs_to_parameters.values()
            self.assertRegex(name, r"^w_prop_[0-9a-f]{64}$")
            self.assertEqual(
                name, "w_prop_" + hashlib.sha256(expression.encode()).hexdigest()
            )
            self.assertTrue(torch.equal(new_ep.module()(x), expected))

    def test_constant_prop_pass_registers_folds_in_graph_order(self) -> None:
        """
        The placeholders of the folded values and their entries in the
        program agree, in both registration modes: a caller that binds the
        graph module by position, the state dict values then the constants
        then the user inputs, gets the right tensor in every slot.
        """

        class ThreeFolds(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(3, 4))
                self.bias = torch.nn.Parameter(torch.randn(3))
                self.register_buffer("scale", torch.tensor(2.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x @ self.w.t() + x @ (self.w * 2).t() + x @ (self.w + 1).t()
                return (y + self.bias) * self.scale

        x = torch.randn(2, 4)
        for register_like_source in (False, True):
            ep = to_edge(export(ThreeFolds(), (x,), strict=True)).exported_program()
            expected = ep.module()(x)
            new_ep = constant_prop_pass(ep, register_like_source=register_like_source)
            new_ep._validate()
            # Three folds, bias, scale and x; w and the lifted scalars are
            # folded away.
            self.assertEqual(
                sum(1 for n in new_ep.graph.nodes if n.op == "placeholder"), 6
            )
            actual = new_ep.graph_module(
                *new_ep.state_dict.values(), *new_ep.constants.values(), x
            )
            self.assertTrue(torch.allclose(actual[0], expected, atol=1e-6))

    def test_constant_prop_pass_nodes_to_fold(self) -> None:
        """
        An allowlist restricts the fold to the given nodes; a node outside it
        stays an op, and so do the nodes computed from it.
        """

        class TwoFolds(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * (self.w * 2) + (self.w + 1).sum()

        x = torch.ones(4)
        ep = export(TwoFolds(), (x,), strict=True)
        mul = ep.graph.find_nodes(op="call_function", target=torch.ops.aten.mul.Tensor)
        new_ep = constant_prop_pass(ep, nodes_to_fold={mul[0]})

        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertEqual(targets.count(torch.ops.aten.mul.Tensor), 1)
        self.assertIn(torch.ops.aten.add.Tensor, targets)
        self.assertIn(torch.ops.aten.sum.default, targets)
        self.assertEqual(list(new_ep.constants), ["_prop_tensor_constant0"])
        self.assertEqual(
            list(new_ep.graph_signature.inputs_to_parameters.values()), ["w"]
        )
        self.assertTrue(torch.equal(new_ep.module()(x), x * 2 + 8))

    def test_constant_prop_pass_keeps_sources_tagged_for_different_files(self) -> None:
        """
        A folded value is one tensor with one custom meta. In a merged
        adapter weight whose base is tagged for one external file and whose
        adapter factors for another, the sum is not folded: the base keeps
        its data and its tag, and only the adapter product, whose sources
        agree, folds into a value tagged like them. With one tag on all
        three the whole weight folds and carries that tag.
        """

        class AdaptedLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = torch.nn.Parameter(torch.randn(3, 4))
                self.lora_a = torch.nn.Parameter(torch.randn(2, 4))
                self.lora_b = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ (self.base + self.lora_b @ self.lora_a).t()

        def tagged_program(model, base_file, lora_file):
            ep = export(model, (x,), strict=True)
            for node in ep.graph.find_nodes(op="placeholder"):
                if node.name != "x":
                    file = lora_file if "lora" in node.name else base_file
                    node.meta["custom"] = {"delegate_constant_tag": file}
            return ep

        def parameter_tags(ep):
            placeholders = {n.name: n for n in ep.graph.find_nodes(op="placeholder")}
            return {
                fqn: placeholders[name].meta["custom"]["delegate_constant_tag"]
                for name, fqn in ep.graph_signature.inputs_to_parameters.items()
            }

        x = torch.randn(2, 4)
        model = AdaptedLinear()
        expected = model(x)

        new_ep = constant_prop_pass(
            tagged_program(model, "foundation.ptd", "lora.ptd"),
            register_like_source=True,
        )
        new_ep._validate()
        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.add.Tensor, targets)
        tags = parameter_tags(new_ep)
        self.assertEqual(tags.pop("base"), "foundation.ptd")
        ((product, tag),) = tags.items()
        self.assertRegex(product, r"^lora_[ab]_prop_[0-9a-f]{64}$")
        self.assertEqual(tag, "lora.ptd")
        self.assertTrue(
            torch.allclose(new_ep.state_dict[product], model.lora_b @ model.lora_a)
        )
        self.assertTrue(torch.allclose(new_ep.module()(x), expected))

        new_ep = constant_prop_pass(
            tagged_program(model, "model.ptd", "model.ptd"), register_like_source=True
        )
        targets = [n.target for n in new_ep.graph.nodes if n.op == "call_function"]
        self.assertEqual(targets, [torch.ops.aten.matmul.default])
        ((weight, tag),) = parameter_tags(new_ep).items()
        self.assertRegex(weight, r"^base_prop_[0-9a-f]{64}$")
        self.assertEqual(tag, "model.ptd")
        self.assertTrue(torch.allclose(new_ep.module()(x), expected))

    def test_constant_prop_pass_clones_a_constant_output_in_the_dialect_of_the_graph(
        self,
    ) -> None:
        """
        A folded value that is also an output is returned through a clone,
        in the dialect of the graph: before to_edge, as in a pre-decomposition
        hook, the clone is the ATen op and the program still decomposes; an
        edge program gets the edge clone.
        """

        class TransposeAndReturn(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(3, 4))

            def forward(self, x: torch.Tensor):
                w = self.w.t()
                return x @ w, w

        def returned_clone(ep):
            (output,) = ep.graph.find_nodes(op="output")
            clone = output.args[0][1]
            self.assertEqual(clone.args[0].op, "placeholder")
            return clone.target

        x = torch.randn(2, 4)
        model = TransposeAndReturn()
        expected = model(x)

        new_ep = constant_prop_pass(
            export(model, (x,), strict=True), register_like_source=True
        )
        self.assertEqual(returned_clone(new_ep), torch.ops.aten.clone.default)
        self.assertFalse(
            any(isinstance(n.target, EdgeOpOverload) for n in new_ep.graph.nodes)
        )
        decomposed = new_ep.run_decompositions({})
        for actual, want in zip(decomposed.module()(x), expected):
            self.assertTrue(torch.allclose(actual, want))

        edge = to_edge(export(model, (x,), strict=True)).exported_program()
        self.assertEqual(
            returned_clone(constant_prop_pass(edge)), exir_ops.edge.aten.clone.default
        )

    def test_constant_prop_pass_fold_buffers_false_leaves_a_buffer_output_alone(
        self,
    ) -> None:
        """
        With fold_buffers=False a buffer is not a constant to the pass, also
        where it is an output: a read-only buffer the method returns stays
        the output as it is. By default it is returned through a clone.
        """

        class ReturnState(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(4))
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x: torch.Tensor):
                return x + self.w * 2, self.state

        def returned(ep):
            (output,) = ep.graph.find_nodes(op="output")
            return output.args[0][1]

        x = torch.ones(4)
        new_ep = constant_prop_pass(
            export(ReturnState(), (x,), strict=True), fold_buffers=False
        )
        self.assertEqual(returned(new_ep).op, "placeholder")
        self.assertEqual(
            new_ep.graph_signature.inputs_to_buffers[returned(new_ep).name], "state"
        )
        self.assertTrue(torch.equal(new_ep.module()(x)[1], torch.zeros(4)))

        new_ep = constant_prop_pass(export(ReturnState(), (x,), strict=True))
        clone = returned(new_ep)
        self.assertEqual(clone.target, torch.ops.aten.clone.default)
        self.assertEqual(
            new_ep.graph_signature.inputs_to_buffers[clone.args[0].name], "state"
        )
