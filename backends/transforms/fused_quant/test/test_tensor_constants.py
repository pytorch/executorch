# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.transforms.fused_quant.pre_quantize_passes.tensor_constants import (
    add_constant,
    fold_constants,
    get_constant,
    get_kind,
)
from executorch.backends.transforms.fused_quant.test.helpers import trace
from parameterized import parameterized
from torch import fx, nn
from torch.export.graph_signature import InputKind


class _Scale(nn.Module):
    """Scales by ``1 + weight``, a value computed only from a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.weight)


class _Counter(nn.Module):
    """Reads a buffer it also writes, as a KV cache does."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count.add_(1)
        return x + self.count * 2


class _Noise(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand(4) * 2


def _targets(graph_module: fx.GraphModule) -> set[object]:
    return {
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    }


class FoldConstantsTest(unittest.TestCase):
    def test_folds_a_parameter_derived_value(self) -> None:
        model = _Scale()
        x = torch.randn(4)
        graph_module = trace(model, (x,)).module()

        fold_constants(graph_module)

        self.assertEqual({torch.ops.aten.mul.Tensor}, _targets(graph_module))
        self.assertEqual([], list(graph_module.named_parameters()))
        torch.testing.assert_close(graph_module(x), model(x))

    def test_leaves_a_buffer_written_in_place(self) -> None:
        x = torch.randn(4)
        graph_module = trace(_Counter(), (x,)).module()
        targets = _targets(graph_module)

        fold_constants(graph_module)

        self.assertEqual(targets, _targets(graph_module))
        model = _Counter()
        for _ in range(2):
            torch.testing.assert_close(graph_module(x), model(x))

    def test_leaves_a_random_draw(self) -> None:
        graph_module = trace(_Noise(), (torch.randn(4),)).module()

        fold_constants(graph_module)

        self.assertIn(torch.ops.aten.rand.default, _targets(graph_module))


class AddConstantTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("parameter", InputKind.PARAMETER),
            ("buffer", InputKind.BUFFER),
            ("constant", InputKind.CONSTANT_TENSOR),
        ]
    )
    def test_adds_an_attribute_of_the_kind(self, _name: str, kind: InputKind) -> None:
        graph_module = trace(_Scale(), (torch.randn(4),)).module()
        tensor = torch.arange(3.0)

        node = add_constant(graph_module, "layer.table", tensor, kind)

        self.assertEqual("layer_table", node.target)
        self.assertEqual(kind, get_kind(node))
        torch.testing.assert_close(get_constant(node), tensor)
        self.assertEqual(tensor.shape, node.meta["val"].shape)

    def test_names_are_unique(self) -> None:
        graph_module = trace(_Scale(), (torch.randn(4),)).module()

        first = add_constant(
            graph_module, "table", torch.zeros(2), InputKind.CONSTANT_TENSOR
        )
        second = add_constant(
            graph_module, "table", torch.ones(2), InputKind.CONSTANT_TENSOR
        )

        self.assertNotEqual(first.target, second.target)
        torch.testing.assert_close(get_constant(second), torch.ones(2))
