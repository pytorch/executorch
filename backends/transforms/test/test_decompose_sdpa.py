# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from torch.export import export


class TestDecomposeScaledDotProductAttention(unittest.TestCase):
    def test_decompose_sdpa_requires_enable_gqa_for_head_mismatch(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, enable_gqa=True
                )

        q = torch.randn(1, 4, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)

        graph_module = export(Model().eval(), (q, k, v), strict=True).module()
        for node in graph_module.graph.nodes:
            if node.target == torch.ops.aten.scaled_dot_product_attention.default:
                node.kwargs = {**node.kwargs, "enable_gqa": False}
                break

        with self.assertRaisesRegex(ValueError, "enable_gqa=True"):
            DecomposeScaledDotProductAttention()(graph_module)

    def test_decompose_sdpa_preserves_kwargs(self) -> None:
        class Block(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    scale=0.25,
                )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = Block()

            def forward(self, q, k, v, mask):
                return self.block(q, k, v, mask)

        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        mask = torch.tensor(
            [[[[True, False, True], [True, True, False], [False, True, True]]]]
        )

        graph_module = export(Model().eval(), (q, k, v, mask), strict=True).module()

        before_output = graph_module(q, k, v, mask)
        original_nn_module_stack = None
        self.assertTrue(
            any(
                node.target == torch.ops.aten.scaled_dot_product_attention.default
                for node in graph_module.graph.nodes
                if node.op == "call_function"
            )
        )
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and (
                node.target == torch.ops.aten.scaled_dot_product_attention.default
            ):
                original_nn_module_stack = node.meta.get("nn_module_stack")
                break

        self.assertIsNotNone(original_nn_module_stack)

        DecomposeScaledDotProductAttention()(graph_module)

        self.assertFalse(
            any(
                node.target == torch.ops.aten.scaled_dot_product_attention.default
                for node in graph_module.graph.nodes
                if node.op == "call_function"
            )
        )
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(
                    node.meta.get("nn_module_stack"), original_nn_module_stack
                )
        torch.testing.assert_close(graph_module(q, k, v, mask), before_output)

    def test_skip_safe_softmax_guard_uses_regular_softmax(self) -> None:
        q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                model = _SDPA(is_causal=is_causal).eval()
                graph_module = export(model, (q, k, v), strict=True).module()

                result = DecomposeScaledDotProductAttention(
                    skip_safe_softmax_guard=True
                )(graph_module)

                self.assertTrue(result.modified)
                targets = _call_targets(result.graph_module)
                self.assertIn(torch.ops.aten.softmax.int, targets)
                self.assertNotIn(torch.ops.aten._safe_softmax.default, targets)
                self.assertNotIn(
                    torch.ops.aten.scaled_dot_product_attention.default, targets
                )
                torch.testing.assert_close(result.graph_module(q, k, v), model(q, k, v))

    def test_skip_safe_softmax_guard_leaves_maskable_sdpa(self) -> None:
        q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
        mask = torch.zeros(5, 5)
        mask[0] = float("-inf")
        cases = {
            "mask": (_SDPA(), (q, k, v, mask)),
            "dropout": (_SDPA(dropout_p=0.1), (q, k, v)),
        }
        for name, (model, inputs) in cases.items():
            with self.subTest(name):
                graph_module = export(model.eval(), inputs, strict=True).module()

                result = DecomposeScaledDotProductAttention(
                    skip_safe_softmax_guard=True
                )(graph_module)

                self.assertFalse(result.modified)
                self.assertIn(
                    torch.ops.aten.scaled_dot_product_attention.default,
                    _call_targets(result.graph_module),
                )


class _SDPA(torch.nn.Module):
    def __init__(self, is_causal: bool = False, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.dropout_p = dropout_p

    def forward(self, q, k, v, mask=None):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p, is_causal=self.is_causal
        )


def _call_targets(graph_module: torch.fx.GraphModule) -> set:
    return {n.target for n in graph_module.graph.nodes if n.op == "call_function"}
