# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for SpecPropPass dim_order propagation to out TensorSpec (Fix #16032).
Run from ExecuTorch repo root: python -m pytest exir/tests/test_spec_prop_dim_order.py -v
"""

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.dim_order_utils import (
    dim_order_from_fake_tensor,
    should_propagate_dim_order,
)
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from torch.export import export


# Clone ops that may appear in the graph: aten (pre-OpReplacePass) or edge (after to_edge).
_CLONE_OPS = (
    torch.ops.aten.clone.default,
    torch.ops.aten.clone.out,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.dim_order_ops._clone_dim_order.default,
)
if hasattr(exir_ops.edge.dim_order_ops._clone_dim_order, "out"):
    _CLONE_OPS = _CLONE_OPS + (exir_ops.edge.dim_order_ops._clone_dim_order.out,)


def _find_clone_nodes(graph_module):
    """
    Return list of (node, self_node, output_spec) for each clone in graph.
    to_edge uses edge ops (edge.aten.clone or edge.dim_order_ops._clone_dim_order).
    output_spec is node.meta['spec'] for single-output, or out_node.meta['spec'] for .out.
    """
    result = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in _CLONE_OPS:
            continue
        if not node.args:
            continue
        self_node = node.args[0]
        if "out" in node.kwargs:
            out_node = node.kwargs["out"]
            output_spec = out_node.meta.get("spec") if isinstance(out_node, torch.fx.Node) else None
        else:
            output_spec = node.meta.get("spec")
        if output_spec is not None:
            result.append((node, self_node, output_spec))
    return result


class TestDimOrderFromFakeTensor(unittest.TestCase):
    def test_contiguous_4d(self) -> None:
        t = torch.randn(2, 3, 4, 5)
        self.assertTrue(t.is_contiguous())
        dim_order = dim_order_from_fake_tensor(t)
        self.assertIsNotNone(dim_order)
        self.assertEqual(dim_order, [0, 1, 2, 3])

    def test_channels_last_4d(self) -> None:
        t = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
        dim_order = dim_order_from_fake_tensor(t)
        self.assertIsNotNone(dim_order)
        self.assertEqual(dim_order, [0, 2, 3, 1])


class TestShouldPropagateDimOrder(unittest.TestCase):
    def test_clone_out(self) -> None:
        self.assertTrue(should_propagate_dim_order(torch.ops.aten.clone.out))

    def test_clone_default(self) -> None:
        self.assertTrue(should_propagate_dim_order(torch.ops.aten.clone.default))

    def test_conv_not_format_preserving(self) -> None:
        self.assertFalse(
            should_propagate_dim_order(torch.ops.aten.convolution.default)
        )


class TestSpecPropPassDimOrder(unittest.TestCase):
    """SpecPropPass must propagate primary input dim_order to out TensorSpec for clone.out."""

    def test_fp32_contiguous_clone(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.clone()

        m = M().eval()
        example = (torch.randn(1, 3, 8, 8),)
        ep = export(m, example)
        edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))
        gm = edge.exported_program().graph_module
        pass_result = SpecPropPass()(gm)
        gm = pass_result.graph_module
        clone_nodes = _find_clone_nodes(gm)
        self.assertGreater(len(clone_nodes), 0, "graph should contain clone")
        for _node, self_node, output_spec in clone_nodes:
            self_spec = self_node.meta.get("spec")
            self.assertIsNotNone(self_spec)
            self.assertIsNotNone(output_spec)
            self.assertEqual(
                output_spec.dim_order,
                self_spec.dim_order,
                "out dim_order should match self (contiguous)",
            )
            self.assertEqual(list(output_spec.dim_order), [0, 1, 2, 3])

    def test_fp16_conv_clone_channels_last(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Explicit channels_last so the traced FakeTensor has channels_last strides.
                return self.conv(x).to(memory_format=torch.channels_last).clone()

        m = M().to(torch.float16).eval()
        example = (torch.randn(1, 3, 16, 16, dtype=torch.float16),)
        ep = export(m, example)
        edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))
        gm = edge.exported_program().graph_module
        pass_result = SpecPropPass()(gm)
        gm = pass_result.graph_module
        clone_nodes = _find_clone_nodes(gm)
        self.assertGreater(len(clone_nodes), 0)
        for _node, self_node, output_spec in clone_nodes:
            self_spec = self_node.meta.get("spec")
            self.assertIsNotNone(self_spec)
            self.assertIsNotNone(output_spec)
            self.assertEqual(
                output_spec.dim_order,
                self_spec.dim_order,
                "out dim_order should match self (channels_last from conv)",
            )
            self.assertEqual(
                list(output_spec.dim_order),
                [0, 2, 3, 1],
                "conv output is channels_last",
            )

    def test_fp16_conv_relu_clone(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
                self.relu = torch.nn.ReLU(inplace=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.conv(x)).clone()

        m = M().to(torch.float16).eval()
        example = (torch.randn(1, 3, 16, 16, dtype=torch.float16),)
        ep = export(m, example)
        edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))
        gm = edge.exported_program().graph_module
        pass_result = SpecPropPass()(gm)
        gm = pass_result.graph_module
        clone_nodes = _find_clone_nodes(gm)
        self.assertGreater(len(clone_nodes), 0)
        for _node, self_node, output_spec in clone_nodes:
            self_spec = self_node.meta.get("spec")
            self.assertIsNotNone(self_spec)
            self.assertIsNotNone(output_spec)
            self.assertEqual(
                output_spec.dim_order,
                self_spec.dim_order,
                "dim_order should propagate through relu to clone",
            )


if __name__ == "__main__":
    unittest.main()
