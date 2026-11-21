# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from torch.export import export
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@pytest.mark.parametrize("strict", [False, True])
def test_decompose_sdpa_preserves_source_ancestry_and_partition_identity(strict):
    class Model(torch.nn.Module):
        def forward(self, q, k, v):
            first = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            second = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return first + second, torch.matmul(q, k.transpose(-2, -1))

    inputs = tuple(torch.randn(1, 2, 3, 4) for _ in range(3))
    graph_module = export(Model(), inputs, strict=strict).module()
    sdpa = torch.ops.aten.scaled_dot_product_attention.default
    parents = {
        node.name: list(node.meta.get("source_fn_stack", []))
        for node in graph_module.graph.nodes
        if node.target == sdpa
    }
    expected = graph_module(*inputs)
    DecomposeScaledDotProductAttention()(graph_module)
    descendants = {name: [] for name in parents}
    for node in graph_module.graph.nodes:
        stack = node.meta.get("source_fn_stack", [])
        for name, prefix in parents.items():
            if (name, sdpa) in stack:
                descendants[name].append(node)
                assert stack[: len(prefix)] == prefix
                assert stack[-1] == (node.name, node.target)
    assert all(descendants.values())
    first, second = descendants.values()
    assert not set(first) & set(second)
    bmm = torch.ops.aten.bmm.default
    partitions = get_source_partitions(graph_module.graph, [bmm])
    assert len(partitions[bmm]) == 4
    assert all(len(partition.nodes) == 1 for partition in partitions[bmm])
    torch.testing.assert_close(graph_module(*inputs), expected)


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
