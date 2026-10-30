# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.passes.replace_masked_copy_with_where import (
    ReplaceMaskedCopyWithWherePass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


def _targets(graph_module):
    return [n.target for n in graph_module.graph.nodes if n.op == "call_function"]


class TestReplaceMaskedCopyWithWherePass(unittest.TestCase):
    def test_registered_in_cuda_backend(self):
        self.assertIn(
            "ReplaceMaskedCopyWithWherePass",
            {type(p).__name__ for p in CudaBackend.get_custom_passes([])},
        )

    def test_masked_copy_becomes_where(self):
        """Also removes the index and the checks on its size."""

        class Copy(torch.nn.Module):
            def forward(self, x, mask):
                out = torch.zeros_like(x)
                # A leading None and a trailing dim, and the mask computed twice,
                # as Transformers' SmolVLM writes its position ids.
                out[:, mask.view(-1)] = x[:, mask.view(-1)]
                return out

        x = torch.randn(2, 5, 3)
        masks = (
            torch.tensor([True, False, True, True, False]),
            torch.zeros(5, dtype=torch.bool),
            torch.ones(5, dtype=torch.bool),
        )
        program = to_edge(export(Copy(), (x, masks[0]))).exported_program()
        ReplaceMaskedCopyWithWherePass()(program.graph_module)

        targets = _targets(program.graph_module)
        self.assertIn(exir_ops.edge.aten.where.self, targets)
        for removed in (
            exir_ops.edge.aten.index.Tensor,
            exir_ops.edge.aten.index_put.default,
            torch.ops.aten._assert_scalar.default,
        ):
            self.assertNotIn(removed, targets)
        for mask in masks:
            torch.testing.assert_close(
                program.module()(x, mask), Copy()(x, mask), rtol=0, atol=0
            )

    def test_other_writes_are_kept(self):
        class Kept(torch.nn.Module):
            def forward(self, x, mask):
                scaled = torch.zeros_like(x)
                # The values are not the mask's picks of a tensor.
                scaled[mask] = x[mask] * 2
                checked = torch.zeros_like(x)
                picked = x[mask]
                # A check on the count that does not hold for every mask.
                torch._check(picked.shape[0] >= 1)
                checked[mask] = picked
                summed = torch.ones_like(x)
                summed.index_put_((mask,), x[mask], accumulate=True)
                other = torch.zeros_like(x)
                other[mask] = x[mask.logical_not()]
                return scaled, checked, summed, other

        x = torch.randn(4, 3)
        mask = torch.tensor([True, False, True, False])
        program = to_edge(export(Kept(), (x, mask))).exported_program()
        ReplaceMaskedCopyWithWherePass()(program.graph_module)

        targets = _targets(program.graph_module)
        self.assertEqual(targets.count(exir_ops.edge.aten.index_put.default), 3)
        self.assertEqual(targets.count(exir_ops.edge.aten.where.self), 1)
        # The index stays for the checked count.
        self.assertIn(exir_ops.edge.aten.index.Tensor, targets)
        for got, expected in zip(program.module()(x, mask), Kept()(x, mask)):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
