# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.passes.replace_masked_copy_with_where import (
    ReplaceMaskedCopyWithWherePass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export


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

    def test_long_equal_masks(self):
        """Two long computations of one mask, with inputs shared many times."""

        class Copy(torch.nn.Module):
            def forward(self, x, mask):
                def compute(m):
                    for _ in range(1200):
                        m = m.logical_not()
                    for _ in range(30):
                        m = m & m
                    return m

                out = torch.zeros_like(x)
                out[compute(mask)] = x[compute(mask)]
                return out

        x = torch.randn(4, 3)
        mask = torch.tensor([True, False, True, False])
        program = to_edge(export(Copy(), (x, mask))).exported_program()
        ReplaceMaskedCopyWithWherePass()(program.graph_module)
        self.assertIn(exir_ops.edge.aten.where.self, _targets(program.graph_module))

    def test_other_writes_are_kept(self):
        class Kept(torch.nn.Module):
            def forward(self, x, mask, idx):
                checked = torch.zeros_like(x)
                # First, since export attaches the check to the first x[mask].
                picked = x[mask]
                # A check on the count that does not hold for every mask.
                torch._check(picked.shape[0] >= 1)
                checked[mask] = picked
                scaled = torch.zeros_like(x)
                # The values are not the mask's picks of a tensor.
                scaled[mask] = x[mask] * 2
                summed = torch.ones_like(x)
                summed.index_put_((mask,), x[mask], accumulate=True)
                other = torch.zeros_like(x)
                other[mask] = x[mask.logical_not()]
                # An integer index is not a mask, so where cannot take it.
                rows = torch.zeros_like(x)
                rows[idx] = x[idx]
                # The picks are also read elsewhere, so the index stays.
                shared = torch.zeros_like(x)
                read = x[mask]
                shared[mask] = read
                return scaled, checked, summed, other, rows, shared, read.sum()

        x = torch.randn(4, 3)
        mask = torch.tensor([True, False, True, False])
        idx = torch.tensor([2, 0])
        program = to_edge(export(Kept(), (x, mask, idx))).exported_program()
        ReplaceMaskedCopyWithWherePass()(program.graph_module)

        targets = _targets(program.graph_module)
        self.assertEqual(targets.count(exir_ops.edge.aten.index_put.default), 4)
        self.assertEqual(targets.count(exir_ops.edge.aten.where.self), 2)
        # The checked count and the shared picks keep their index.
        self.assertEqual(targets.count(exir_ops.edge.aten.index.Tensor), 6)
        for got, expected in zip(
            program.module()(x, mask, idx), Kept()(x, mask, idx), strict=True
        ):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_check_on_another_size_keeps_the_index(self):
        """A check that reads the count but tests another size does not hold
        for every mask."""

        class Copy(torch.nn.Module):
            def forward(self, x, mask, y):
                out = torch.zeros_like(x)
                out[mask] = x[mask]
                return out, y * 2

        x = torch.randn(4, 3)
        mask = torch.tensor([True, False, True, False])
        y = torch.randn(3)
        program = to_edge(
            export(Copy(), (x, mask, y), dynamic_shapes=(None, None, {0: Dim("n")}))
        ).exported_program()
        graph = program.graph_module.graph
        index = next(
            n for n in graph.nodes if n.target == exir_ops.edge.aten.index.Tensor
        )
        y_node = next(n for n in graph.nodes if n.name == "y")
        # count * 0 + y.size(0) <= 4 has the one symbol of y's size.
        with graph.inserting_before(index.next):
            count = graph.call_function(torch.ops.aten.sym_size.int, (index, 0))
            size = graph.call_function(torch.ops.aten.sym_size.int, (y_node, 0))
            zero = graph.call_function(operator.mul, (count, 0))
            total = graph.call_function(operator.add, (zero, size))
            check = graph.call_function(operator.le, (total, 4))
            graph.call_function(torch.ops.aten._assert_scalar.default, (check, ""))
        count.meta["val"] = index.meta["val"].shape[0]
        size.meta["val"] = y_node.meta["val"].shape[0]
        zero.meta["val"] = count.meta["val"] * 0
        total.meta["val"] = zero.meta["val"] + size.meta["val"]
        check.meta["val"] = total.meta["val"] <= 4
        ReplaceMaskedCopyWithWherePass()(program.graph_module)

        targets = _targets(program.graph_module)
        self.assertIn(exir_ops.edge.aten.where.self, targets)
        self.assertIn(exir_ops.edge.aten.index.Tensor, targets)


if __name__ == "__main__":
    unittest.main()
