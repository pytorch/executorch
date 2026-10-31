# mypy: disable-error-code=import-not-found

import itertools
import unittest
from unittest import mock

import torch
from executorch.backends.openvino._passes import (
    fold_constant_cumsum_pass,
    FoldConstantCumsumPass,
)
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from openvino.frontend.pytorch.torchdynamo.op_support import (  # type: ignore[import-untyped]
    OperatorSupport,
)


class TestIntegerReductions(BaseOpenvinoOpTest):
    """PyTorch reduces bool and small integer tensors to int64."""

    def test_cumsum(self):
        class CumSum(torch.nn.Module):
            def forward(self, x):
                return torch.cumsum(x, 1)

        for x in (
            torch.rand(2, 7) > 0.5,
            torch.randint(0, 5, (2, 7), dtype=torch.int32),
        ):
            with self.subTest(dtype=x.dtype):
                self.execute_layer_test(CumSum(), (x,))

    def test_sum(self):
        class Sum(torch.nn.Module):
            def forward(self, x):
                return x.sum(1)

        # More than 255 true values in a row, which 8 bits cannot count.
        mask = torch.arange(800).reshape(2, 400) % 4 != 0
        for x in (mask, torch.randint(0, 5, (2, 7), dtype=torch.int32)):
            with self.subTest(dtype=x.dtype):
                self.execute_layer_test(Sum(), (x,))

    def test_prod(self):
        class Prod(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return x.prod() if self.dim is None else x.prod(self.dim)

        support = OperatorSupport([])._support_dict
        x = torch.randint(1, 4, (2, 5), dtype=torch.int32)
        for dim, op in ((1, "prod.dim_int"), (None, "prod.default")):
            with self.subTest(op=op):
                if f"torch.ops.aten.{op}" not in support:
                    self.skipTest(f"this OpenVINO release does not delegate {op}")
                self.execute_layer_test(Prod(dim), (x,))

    def test_attention_masks_with_more_valid_tokens(self):
        class AttentionMasks(torch.nn.Module):
            """Masks and positions from padding, as pi0 builds them."""

            def __init__(self, dtype, from_input):
                super().__init__()
                self.dtype = dtype
                self.from_input = from_input

            def forward(self, pad_masks):
                if self.from_input:
                    # Constant too, but only once OpenVINO folds the shape.
                    att_masks = torch.ones_like(pad_masks[0])
                else:
                    att_masks = torch.tensor([0] * 765 + [1] * 51, dtype=torch.bool)
                cumsum = torch.cumsum(att_masks[None, :], dim=1, dtype=self.dtype)
                att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
                pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
                return att_2d & pad_2d, torch.cumsum(pad_masks, dim=1) - 1

        def pad_masks(valid):
            return (torch.arange(816) < valid)[None, :]

        for dtype, from_input in itertools.product(
            (torch.int64, torch.int32), (False, True)
        ):
            with self.subTest(dtype=dtype, from_input=from_input):
                # One loaded program, run again with more valid tokens.
                self.execute_layer_test(
                    AttentionMasks(dtype, from_input),
                    (pad_masks(300),),
                    more_inputs=((pad_masks(700),),),
                )


class TestFoldConstantCumsumPass(unittest.TestCase):
    def fold(self, module, inputs):
        """Edge program module of module, with constant cumsums folded."""
        edge = to_edge(torch.export.export(module, inputs))
        folded = edge.exported_program().module()
        FoldConstantCumsumPass()(folded)
        return folded

    def test_buffer_written_by_the_graph_is_not_folded(self):
        class State(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("s", torch.arange(3))

            def forward(self, x):
                self.s.add_(1)
                return torch.cumsum(self.s, 0) + x

        x = torch.zeros(3, dtype=torch.int64)
        folded = self.fold(State(), (x,))
        eager = State()
        for _ in range(2):
            torch.testing.assert_close(folded(x), eager(x))

    def test_folded_name_does_not_replace_a_buffer(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "_folded_cumsum_aten_cumsum_default", torch.ones(3)
                )

            def forward(self, x):
                counts = torch.cumsum(torch.ones(3, dtype=torch.int64), 0)
                return counts + x + self._folded_cumsum_aten_cumsum_default

        x = torch.zeros(3)
        folded = self.fold(Model(), (x,))
        # The folded cumsum would have taken this name.
        self.assertTrue(hasattr(folded, "_folded_cumsum_aten_cumsum_default_"))
        torch.testing.assert_close(folded(x), Model()(x))

    def test_a_cumsum_that_cannot_be_evaluated_leaves_the_others_folded(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                ones = torch.ones(3, dtype=torch.int64)
                return torch.cumsum(ones, 0) + torch.cumsum(ones * 2, 0) + x

        evaluate = fold_constant_cumsum_pass._evaluate

        def fail_on_the_doubled_one(graph_module, targets):
            if any("mul" in str(n.args[0].target) for n in targets):
                raise RuntimeError("not on the CPU")
            return evaluate(graph_module, targets)

        x = torch.zeros(3, dtype=torch.int64)
        with mock.patch.object(
            fold_constant_cumsum_pass, "_evaluate", side_effect=fail_on_the_doubled_one
        ), self.assertLogs(level="WARNING") as logs:
            folded = self.fold(Model(), (x,))
        targets = [n.target for n in folded.graph.nodes]
        self.assertEqual(targets.count(exir_ops.edge.aten.cumsum.default), 1)
        self.assertIn("wrong results", logs.output[0])
        torch.testing.assert_close(folded(x), Model()(x))

    def test_float_cumsum_is_not_folded(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cumsum(torch.ones(3), 0) + x

        folded = self.fold(Model(), (torch.zeros(3),))
        self.assertIn(
            exir_ops.edge.aten.cumsum.default,
            [n.target for n in folded.graph.nodes],
        )

    def test_random_draws_are_not_folded(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                c = torch.ones(3)
                r = torch.cond(
                    torch.tensor(True),
                    lambda c: (torch.rand(3) * 100 + c).long(),
                    lambda c: c.long(),
                    (c,),
                )
                direct = (torch.rand(3) * 100).long()
                return torch.cumsum(r, 0) + torch.cumsum(direct, 0) + x

        folded = self.fold(Model(), (torch.zeros(3, dtype=torch.int64),))
        targets = [n.target for n in folded.graph.nodes]
        self.assertEqual(targets.count(exir_ops.edge.aten.cumsum.default), 2)

    def test_cumsum_over_a_split_constant_is_folded(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                part = torch.ones(6, dtype=torch.int64).split(3)[1]
                return torch.cumsum(part, 0) + x

        x = torch.zeros(3, dtype=torch.int64)
        folded = self.fold(Model(), (x,))
        self.assertNotIn(
            exir_ops.edge.aten.cumsum.default,
            [n.target for n in folded.graph.nodes],
        )
        torch.testing.assert_close(folded(x), Model()(x))

    def test_constants_are_folded_without_a_gpu(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                ones = torch.ones(3, dtype=torch.int64, device="cuda")
                pinned = torch.ones(3, dtype=torch.int64, pin_memory=True)
                return torch.cumsum(ones, 0).cpu() + torch.cumsum(pinned, 0) + x

        x = torch.zeros(3, dtype=torch.int64)
        folded = self.fold(Model(), (x,))
        self.assertNotIn(
            exir_ops.edge.aten.cumsum.default,
            [n.target for n in folded.graph.nodes],
        )
        torch.testing.assert_close(folded(x), torch.tensor([2, 4, 6]))
