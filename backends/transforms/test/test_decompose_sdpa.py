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
