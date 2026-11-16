# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for RecomposeRmsNorm on edge graphs of ``torch.nn.RMSNorm``."""

import unittest

import torch
from executorch.backends.samsung._passes.compose_rms_norm import RecomposeRmsNorm
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops

RMS_NORM = exir_ops.edge.aten.rms_norm.default


class Norm(torch.nn.Module):
    def __init__(self, affine: bool, dtype: torch.dtype, add_first: bool) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(8, elementwise_affine=affine, dtype=dtype)
        self.add_first = add_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + 1.0 if self.add_first else x)


def _recompose(affine: bool, dtype: torch.dtype, add_first: bool):
    module = Norm(affine, dtype, add_first).eval()
    exported = torch.export.export(module, (torch.randn(2, 8, dtype=dtype),))
    graph_module = to_edge(exported).exported_program().graph_module
    RecomposeRmsNorm()(graph_module)
    nodes = [
        n
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target == RMS_NORM
    ]
    assert len(nodes) == 1, f"expected one rms_norm, found {len(nodes)}"
    return nodes[0], graph_module


class TestRecomposeRmsNorm(unittest.TestCase):
    def test_affine_norm_keeps_its_weight(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for add_first in (True, False):
                with self.subTest(dtype=dtype, add_first=add_first):
                    node, _ = _recompose(True, dtype, add_first)
                    _, shape, gamma, _ = node.args
                    self.assertEqual(shape, [8])
                    self.assertEqual(gamma.op, "placeholder")
                    self.assertTrue(gamma.name.endswith("weight"), gamma.name)

    def test_activation_is_the_input_even_when_it_is_a_graph_input(self):
        node, _ = _recompose(True, torch.float32, add_first=False)
        self.assertEqual(node.args[0].op, "placeholder")
        self.assertEqual(node.args[0].name, "x")
        node, _ = _recompose(True, torch.float32, add_first=True)
        self.assertEqual(node.args[0].op, "call_function")

    def test_norm_without_weight_gets_an_all_ones_gamma_of_the_norm_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                node, graph_module = _recompose(False, dtype, add_first=True)
                gamma = node.args[2]
                self.assertEqual(gamma.op, "get_attr")
                value = getattr(graph_module, gamma.target)
                self.assertEqual(value.shape, (8,))
                self.assertEqual(value.dtype, dtype)
                self.assertTrue(bool((value == 1).all()))


if __name__ == "__main__":
    unittest.main()
