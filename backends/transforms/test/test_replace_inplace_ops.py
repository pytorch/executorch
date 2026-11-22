# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.replace_inplace_ops import ReplaceInplaceOpsPass
from parameterized import parameterized


def _targets(gm: torch.fx.GraphModule):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class TestReplaceInplaceOps(unittest.TestCase):
    def _run(self, module, *inputs, strict=True):
        """Export `module`, run the pass, and check it against eager."""
        module = module.eval()
        gm = torch.export.export(
            module, tuple(i.clone() for i in inputs), strict=strict
        ).module()
        result = ReplaceInplaceOpsPass()(gm)
        torch.testing.assert_close(
            result.graph_module(*[i.clone() for i in inputs]),
            module(*[i.clone() for i in inputs]),
        )
        return result

    @parameterized.expand([(True,), (False,)])
    def test_write_through_view_becomes_scatter(self, strict):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3, 4)
                buf.select(0, 1)[:, 1:3].copy_(x)
                return buf * 2

        result = self._run(M(), torch.randn(3, 2), strict=strict)
        self.assertTrue(result.modified)
        targets = _targets(result.graph_module)
        self.assertNotIn(torch.ops.aten.copy_.default, targets)
        self.assertIn(torch.ops.aten.select_scatter.default, targets)
        self.assertIn(torch.ops.aten.slice_scatter.default, targets)

    @parameterized.expand([(True,), (False,)])
    def test_writes_in_a_loop(self, strict):
        # Like torchvision's batch_images.
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(3, 2, 5)
                for i in range(3):
                    buf[i, :, : 2 + i].copy_(x[:, : 2 + i] * i)
                return buf + 1

        result = self._run(M(), torch.randn(2, 4), strict=strict)
        self.assertNotIn(torch.ops.aten.copy_.default, _targets(result.graph_module))

    def test_write_through_diagonal(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(3, 3)
                buf.diagonal().add_(x)
                return buf * 2

        result = self._run(M(), torch.randn(3))
        self.assertIn(
            torch.ops.aten.diagonal_scatter.default, _targets(result.graph_module)
        )

    def test_new_nodes_get_their_own_meta(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3, 4)
                buf.select(0, 1)[:, 1:3].copy_(x)
                return buf * 2

        gm = self._run(M(), torch.randn(3, 2)).graph_module
        nodes = {n.target: n for n in gm.graph.nodes}
        for target, shape in [
            (torch.ops.aten.copy.default, (3, 2)),
            (torch.ops.aten.slice_scatter.default, (3, 4)),
            (torch.ops.aten.select_scatter.default, (2, 3, 4)),
        ]:
            self.assertEqual(tuple(nodes[target].meta["val"].shape), shape)
        self.assertIsNot(
            nodes[torch.ops.aten.copy.default].meta["nn_module_stack"],
            nodes[torch.ops.aten.select_scatter.default].meta["nn_module_stack"],
        )

    def test_second_write_through_the_same_view_is_kept(self):
        def first(x):
            buf = x.new_zeros(2, 3)
            row = buf.select(0, 0)
            row.copy_(x)
            row.mul_(2)
            return buf + 0

        def augmented(x):
            buf = x.new_zeros(2, 3)
            row = buf.select(0, 0)
            row += x
            row *= 2
            return buf + 0

        def chained(x):
            buf = x.new_zeros(2, 3)
            buf[0].copy_(x).add_(1)
            return buf + 0

        def base_written_later(x):
            buf = x.new_zeros(2, 3)
            row = buf.select(0, 0)
            row.copy_(x)
            buf.add_(1)
            return row * 2

        for fn in (first, augmented, chained, base_written_later):
            with self.subTest(fn.__name__):
                M = type(fn.__name__, (torch.nn.Module,), {"forward": staticmethod(fn)})
                self.assertFalse(self._run(M(), torch.randn(3)).modified)

    def test_scalar_write_uses_scalar_overload(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x * 1
                buf.select(0, 0).mul_(0.5)
                buf.select(0, 1).add_(2, alpha=3)
                return buf + 0

        result = self._run(M(), torch.randn(2, 3))
        targets = _targets(result.graph_module)
        self.assertIn(torch.ops.aten.mul.Scalar, targets)
        self.assertIn(torch.ops.aten.add.Scalar, targets)

    def test_alias_read_after_write_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3)
                row = buf.select(0, 0)
                buf.select(0, 0).copy_(x)
                return row + buf.sum()

        result = self._run(M(), torch.randn(3))
        self.assertFalse(result.modified)

    def test_unbind_alias_read_after_write_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3)
                first, _ = buf.unbind(0)
                buf.select(0, 0).copy_(x)
                return first * 2

        result = self._run(M(), torch.randn(3))
        self.assertFalse(result.modified)

    def test_inplace_view_op_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x):
                buf = x.new_zeros(2, 3) + x
                buf.select(0, 0).unsqueeze_(0)
                return buf * 2

        self.assertFalse(self._run(M(), torch.randn(3)).modified)

    def test_direct_in_place_ops_are_kept(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                y.relu_()
                y.gt_(0.5)
                return y * 2

        result = self._run(M(), torch.randn(4))
        self.assertFalse(result.modified)
        self.assertIn(torch.ops.aten.relu_.default, _targets(result.graph_module))

    def test_write_to_input_is_kept(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                x.select(0, 0).copy_(y)
                return x + 1

        x, y = torch.randn(2, 3), torch.randn(3)
        gm = torch.export.export(M(), (x.clone(), y), strict=True).module()
        self.assertFalse(ReplaceInplaceOpsPass()(gm).modified)
        gm(x, y)
        torch.testing.assert_close(x[0], y)

    def test_write_to_buffer_is_kept(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(2, 3))

            def forward(self, x):
                self.buf.select(0, 0).copy_(x)
                return self.buf * 2

        x = torch.randn(3)
        gm = torch.export.export(M(), (x,), strict=True).module()
        self.assertFalse(ReplaceInplaceOpsPass()(gm).modified)
        gm(x)
        torch.testing.assert_close(gm.buf[0], x)


if __name__ == "__main__":
    unittest.main()
