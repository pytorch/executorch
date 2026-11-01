# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.convert_dtype_pass import I64toI32
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class TestConvertDtypePass(unittest.TestCase):
    def test_i64_to_i32_casts_topk_indices_once_for_cats(self) -> None:
        class TopKIndices(torch.nn.Module):
            def forward(self, x: torch.Tensor, ids: torch.Tensor):
                indices = torch.topk(x, 2).indices
                return (
                    torch.cat([ids, indices], dim=1),
                    torch.cat([indices, ids], dim=1),
                    torch.gather(ids, 1, indices),
                )

        module = TopKIndices()
        inputs = (torch.randn(2, 3), torch.randint(0, 8, (2, 3)))
        edge = to_edge(export(module, inputs, strict=True)).transform([I64toI32()])

        nodes = edge.exported_program().graph.nodes
        cats = [n for n in nodes if n.target == exir_ops.edge.aten.cat.default]
        gather = next(n for n in nodes if n.target == exir_ops.edge.aten.gather.default)
        for cat in cats:
            self.assertEqual(
                [t.meta["val"].dtype for t in cat.args[0]], [torch.int32] * 2
            )
        self.assertIs(cats[0].args[0][1], cats[1].args[0][0])
        # The pass rewrites gather to int32 too, but only casts tensor lists, so
        # gather keeps the int64 index it needs.
        self.assertEqual(gather.args[2].meta["val"].dtype, torch.int64)
        torch.testing.assert_close(
            edge.exported_program().module()(*inputs), module(*inputs)
        )
