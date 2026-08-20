# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from backends.cuda.passes.replace_int64_floordiv import (
    ReplaceInt64FloorDivWithFloatPass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


_INT_DIV_OPS = (
    exir_ops.edge.aten.floor_divide.default,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.div.Scalar_mode,
)


def _count_int_floordiv(graph_module) -> int:
    """Count integer floor-division nodes remaining in the graph."""
    n = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target not in _INT_DIV_OPS:
            continue
        if node.target in (
            exir_ops.edge.aten.div.Tensor_mode,
            exir_ops.edge.aten.div.Scalar_mode,
        ):
            rmode = node.kwargs.get("rounding_mode", None)
            if rmode != "floor":
                continue
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor) and val.dtype in (
            torch.int64,
            torch.int32,
        ):
            n += 1
    return n


class TestReplaceInt64FloorDivWithFloatPass(unittest.TestCase):
    """Test the ReplaceInt64FloorDivWithFloatPass transformation pass."""

    def _edge_gm(self, module, inputs):
        ep = to_edge(export(module, inputs, strict=True))
        return ep, ep.exported_program().graph_module

    def test_tensor_tensor_floordiv_rewritten(self):
        """int64 a // b (tensor/tensor), including negative numerators."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a // b

        a = torch.tensor([-5, 7, -8, 9, -1, 0], dtype=torch.long)
        b = torch.tensor([2, 3, 4, 5, 3, 7], dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (a, b))

        self.assertGreater(_count_int_floordiv(gm), 0)
        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(a, b)
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(torch.equal(out, a // b))

    def test_scalar_divisor_floordiv_rewritten(self):
        """int64 a // 3 (scalar divisor lifted to a 0-d tensor constant)."""

        class M(torch.nn.Module):
            def forward(self, a):
                return a // 3

        a = torch.tensor([-5, 7, -8, 9, -1, 0], dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (a,))

        self.assertGreater(_count_int_floordiv(gm), 0)
        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(a)
        self.assertTrue(torch.equal(out, a // 3))

    def test_div_rounding_mode_floor_rewritten(self):
        """torch.div(..., rounding_mode='floor') on int64 is rewritten."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.div(a, b, rounding_mode="floor")

        a = torch.tensor([-5, 7, -8, 9], dtype=torch.long)
        b = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (a, b))

        self.assertGreater(_count_int_floordiv(gm), 0)
        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(a, b)
        self.assertTrue(torch.equal(out, torch.div(a, b, rounding_mode="floor")))

    def test_int32_floordiv_rewritten(self):
        """int32 floor-division is also rewritten and stays int32."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a // b

        a = torch.tensor([-5, 7, -8, 9], dtype=torch.int32)
        b = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
        ep, gm = self._edge_gm(M().eval(), (a, b))

        self.assertGreater(_count_int_floordiv(gm), 0)
        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(a, b)
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(torch.equal(out, a // b))

    def test_float_division_untouched(self):
        """Real float division must not be rewritten."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a / b

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 3.0, 4.0])
        ep, gm = self._edge_gm(M().eval(), (a, b))

        before = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        result = ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertFalse(result.modified)
        after = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(before, after)

    def test_trunc_rounding_mode_untouched(self):
        """div with rounding_mode='trunc' must not be rewritten."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.div(a, b, rounding_mode="trunc")

        a = torch.tensor([-5, 7, -8, 9], dtype=torch.long)
        b = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (a, b))

        result = ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertFalse(result.modified)

    def test_floor_divide_default_branch(self):
        """Exercise the floor_divide.default match/rewrite branch.

        This pin lowers ``//`` to ``div.Tensor_mode``; floor_divide.default does
        not appear naturally, so we synthesize it by retargeting a node.
        """

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a // b

        a = torch.tensor([-5, 7, -8, 9], dtype=torch.long)
        b = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (a, b))

        # Retarget the div.Tensor_mode node to floor_divide.default.
        for node in list(gm.graph.nodes):
            if node.target == exir_ops.edge.aten.div.Tensor_mode:
                with gm.graph.inserting_before(node):
                    new = gm.graph.call_function(
                        exir_ops.edge.aten.floor_divide.default, args=node.args
                    )
                    new.meta = node.meta.copy()
                node.replace_all_uses_with(new)
                gm.graph.erase_node(node)
        gm.recompile()

        self.assertGreater(_count_int_floordiv(gm), 0)
        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(a, b)
        self.assertTrue(torch.equal(out, a // b))

    def test_ring_buffer_mask_analog(self):
        """gemma4_31b sliding-window analog: negative numerators + scalar divisor."""

        class M(torch.nn.Module):
            def forward(self, input_pos):
                buf_size = 8
                seq_len = input_pos.shape[0]
                total_written = input_pos[0] + seq_len
                j = torch.arange(buf_size, dtype=torch.long)
                wraps = (total_written - 1 - j) // buf_size
                return j + wraps * buf_size

        input_pos = torch.arange(3, dtype=torch.long)
        ep, gm = self._edge_gm(M().eval(), (input_pos,))

        ReplaceInt64FloorDivWithFloatPass()(gm)
        self.assertEqual(_count_int_floordiv(gm), 0)

        out = ep.exported_program().module()(input_pos)
        ref = M()(input_pos)
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
