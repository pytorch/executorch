# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.aoti.passes.replace_view_copy_with_view import (
    ReplaceViewCopyWithViewPass,
)
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.passes.copy_returned_views import CopyReturnedViewsPass
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


def _lower(module, x):
    """Runs the view passes as the CUDA backend does, then checks the program
    as the backend does, here with the default EdgeCompileConfig."""
    program = to_edge(export(module, (x,), strict=True)).exported_program()
    ReplaceViewCopyWithViewPass()(program.graph_module)
    CopyReturnedViewsPass()(program.graph_module)
    program.validate()
    return program


def _targets(graph_module):
    return [n.target for n in graph_module.graph.nodes if n.op == "call_function"]


class TestCopyReturnedViewsPass(unittest.TestCase):
    def test_registered_in_cuda_backend(self):
        self.assertIn(
            "CopyReturnedViewsPass",
            {type(p).__name__ for p in CudaBackend.get_custom_passes([])},
        )

    def test_returned_view_is_copied(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return (x * 2)[:, :6]

        class Select(torch.nn.Module):
            def forward(self, x):
                return (x * 2)[:, 2]

        x = torch.randn(4, 32)
        for module, view in (
            (Slice(), exir_ops.edge.aten.slice.Tensor),
            (Select(), exir_ops.edge.aten.select.int),
        ):
            with self.subTest(module=type(module).__name__):
                program = _lower(module, x)
                (returned,) = program.graph_module.graph.output_node().args[0]
                self.assertEqual(returned.target, exir_ops.edge.aten.alias_copy.default)
                self.assertEqual(returned.args[0].target, view)
                output = program.module()(x)
                self.assertTrue(output.is_contiguous())
                torch.testing.assert_close(output, module(x), rtol=0, atol=0)

    def test_intermediate_view_is_not_copied(self):
        class SliceThenScale(torch.nn.Module):
            def forward(self, x):
                return x[:, :6] * 2

        targets = _targets(_lower(SliceThenScale(), torch.randn(4, 32)).graph_module)
        self.assertIn(exir_ops.edge.aten.slice.Tensor, targets)
        self.assertNotIn(exir_ops.edge.aten.alias_copy.default, targets)

    def test_returned_detach_is_copied_once(self):
        """ReplaceViewCopyWithViewPass already turns a detach into a copy."""

        class Detach(torch.nn.Module):
            def forward(self, x):
                y = x * 2
                return y, y.detach()

        targets = _targets(_lower(Detach(), torch.randn(4, 32)).graph_module)
        self.assertEqual(targets.count(exir_ops.edge.aten.alias_copy.default), 1)


if __name__ == "__main__":
    unittest.main()
