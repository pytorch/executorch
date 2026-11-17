# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.replace_in_place_ops import replace_in_place_ops


def _targets(gm: torch.fx.GraphModule):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class TestReplaceInPlaceOps(unittest.TestCase):
    def _export(self, module, *inputs):
        return torch.export.export(module.eval(), inputs, strict=True).module()

    def test_write_through_view_becomes_scatter(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3, 4)
                buf.select(0, 1)[:, 1:3].copy_(x)
                return buf * 2

        x = torch.randn(3, 2)
        gm = replace_in_place_ops(self._export(M(), x))
        targets = _targets(gm)
        self.assertNotIn(torch.ops.aten.copy_.default, targets)
        self.assertIn(torch.ops.aten.select_scatter.default, targets)
        self.assertIn(torch.ops.aten.slice_scatter.default, targets)
        torch.testing.assert_close(gm(x), M()(x))

    def test_direct_in_place_op_becomes_functional(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                y.relu_()
                return y * 2

        x = torch.randn(4)
        gm = replace_in_place_ops(self._export(M(), x))
        self.assertNotIn(torch.ops.aten.relu_.default, _targets(gm))
        torch.testing.assert_close(gm(x), M()(x))

    def test_write_to_input_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                x.select(0, 0).copy_(y)
                return x + 1

        x, y = torch.randn(2, 3), torch.randn(3)
        gm = replace_in_place_ops(self._export(M(), x.clone(), y))
        self.assertIn(torch.ops.aten.copy_.default, _targets(gm))

    def test_alias_read_after_write_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3)
                row = buf.select(0, 1)
                buf.select(0, 0).copy_(x)
                return row + buf.sum()

        x = torch.randn(3)
        gm = replace_in_place_ops(self._export(M(), x))
        # `row` was taken before the write and is read after it, so rewriting
        # would leave it pointing at the old value.
        self.assertIn(torch.ops.aten.copy_.default, _targets(gm))
        torch.testing.assert_close(gm(x), M()(x))


if __name__ == "__main__":
    unittest.main()
