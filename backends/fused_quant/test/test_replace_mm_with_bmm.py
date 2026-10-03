# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import unittest

import torch
from executorch.backends.cadence.aot.compiler import trace
from executorch.backends.fused_quant.pre_quantize_passes.replace_mm_with_bmm import (
    ReplaceMmWithBmm,
)
from torch import nn
from torch.export import ExportedProgram

_MM = torch.ops.aten.mm.default
_BMM = torch.ops.aten.bmm.default


class _ConstantMmModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, self.proj)


class _DynamicMmModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, y)


class _MultiMmModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj_a = nn.Parameter(torch.randn(dim, dim))
        self.proj_b = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.mm(x, self.proj_a)
        return torch.mm(a, self.proj_b)


def _count(ep: ExportedProgram, target: object) -> int:
    return len(ep.graph_module.graph.find_nodes(op="call_function", target=target))


class ReplaceMmWithBmmTest(unittest.TestCase):
    def test_constant_weight_replaced(self) -> None:
        torch.manual_seed(0)
        model = _ConstantMmModel(8, 4).eval()
        inp = torch.randn(2, 8)
        expected = model(inp)
        ep = trace(model, (inp,))

        self.assertEqual(_count(ep, _MM), 1)

        result = ReplaceMmWithBmm().call(ep.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(ep, _MM), 0)
        self.assertEqual(_count(ep, _BMM), 1)
        actual = ep.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_dynamic_inputs_replaced(self) -> None:
        torch.manual_seed(0)
        model = _DynamicMmModel().eval()
        x = torch.randn(3, 5)
        y = torch.randn(5, 4)
        expected = model(x, y)
        ep = trace(model, (x, y))

        self.assertEqual(_count(ep, _MM), 1)

        result = ReplaceMmWithBmm().call(ep.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(ep, _MM), 0)
        self.assertEqual(_count(ep, _BMM), 1)
        actual = ep.module()(x, y)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_multiple_mm_replaced(self) -> None:
        torch.manual_seed(0)
        model = _MultiMmModel(6).eval()
        inp = torch.randn(2, 6)
        expected = model(inp)
        ep = trace(model, (inp,))

        self.assertEqual(_count(ep, _MM), 2)

        result = ReplaceMmWithBmm().call(ep.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(ep, _MM), 0)
        self.assertEqual(_count(ep, _BMM), 2)
        actual = ep.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_no_mm_not_modified(self) -> None:
        model = nn.Linear(4, 3).eval()
        inp = torch.randn(2, 4)
        ep = trace(model, (inp,))

        result = ReplaceMmWithBmm().call(ep.graph_module)

        self.assertFalse(result.modified)
