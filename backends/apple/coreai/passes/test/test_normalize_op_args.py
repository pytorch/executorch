# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`NormalizeCoreAIOpArgsPass`."""

import operator
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.apple.coreai import (
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.partition.partitioner import (
    CoreAIPartitioner,
    is_coreai_supported_variant,
)
from executorch.backends.apple.coreai.passes import NormalizeCoreAIOpArgsPass
from executorch.backends.apple.coreai.passes.normalize_op_args import coreai_aten_op
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import executorch_call_delegate


class _Fn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _edge_gm(fn, x):
    ep = torch.export.export(_Fn(fn).eval(), (x,))
    edge = to_edge(ep, compile_config=get_default_compile_config())
    return edge.exported_program().graph_module


def _node(gm, aten_op):
    return next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and coreai_aten_op(n.target) == aten_op
    )


aten = torch.ops.aten


class NormalizeCoreAIOpArgsPassTest(unittest.TestCase):
    def _assert_normalized(self, fn, x, aten_op, expected_args):
        gm = _edge_gm(fn, x)
        node = _node(gm, aten_op)
        self.assertFalse(is_coreai_supported_variant(node))

        self.assertTrue(NormalizeCoreAIOpArgsPass()(gm).modified)
        self.assertEqual(list(node.args[1:]), expected_args)
        self.assertTrue(is_coreai_supported_variant(node))
        torch.testing.assert_close(gm(x)[0], fn(x))

    def test_reductions_over_all_dims(self):
        x = torch.randn(2, 3, 4)
        cases = {
            "mean_dim_none": (lambda t: t.mean(dim=None), aten.mean.dim),
            "amax_no_dim": (torch.amax, aten.amax.default),
            "amin_empty_dim": (lambda t: torch.amin(t, dim=[]), aten.amin.default),
            "amax_keepdim": (
                lambda t: torch.amax(t, dim=[], keepdim=True),
                aten.amax.default,
            ),
        }
        for name, (fn, op) in cases.items():
            with self.subTest(case=name):
                gm = _edge_gm(fn, x)
                node = _node(gm, op)
                self.assertFalse(is_coreai_supported_variant(node))
                self.assertTrue(NormalizeCoreAIOpArgsPass()(gm).modified)
                self.assertEqual(node.args[1], [0, 1, 2])
                self.assertTrue(is_coreai_supported_variant(node))
                torch.testing.assert_close(gm(x)[0], fn(x))

    def test_avg_pool_defaults_filled(self):
        # On 4x4 input ceil_mode doesn't change the output shape, so it's cleared.
        self._assert_normalized(
            lambda t: F.avg_pool2d(t, 2, ceil_mode=True),
            torch.randn(1, 2, 4, 4),
            aten.avg_pool2d.default,
            [[2, 2], [2, 2], [0, 0], False, True],
        )
        self._assert_normalized(
            lambda t: F.avg_pool3d(t, 2, ceil_mode=True),
            torch.randn(1, 2, 4, 4, 4),
            aten.avg_pool3d.default,
            [[2, 2, 2], [2, 2, 2], [0, 0, 0], False, True],
        )

    def test_shape_changing_ceil_mode_left_unsupported(self):
        fn = lambda t: F.avg_pool2d(t, 2, ceil_mode=True)  # noqa: E731
        x = torch.randn(1, 2, 5, 5)
        gm = _edge_gm(fn, x)
        node = _node(gm, aten.avg_pool2d.default)
        NormalizeCoreAIOpArgsPass()(gm)
        self.assertEqual(list(node.args[1:]), [[2, 2], [2, 2], [0, 0], True, True])
        self.assertFalse(is_coreai_supported_variant(node))
        torch.testing.assert_close(gm(x)[0], fn(x))

    def test_squeeze_drops_non_singleton_dims(self):
        self._assert_normalized(
            lambda t: t.squeeze((0, 1)),
            torch.randn(1, 3, 4),
            aten.squeeze.dims,
            [[0]],
        )

    def test_no_effect_squeeze_becomes_clone(self):
        x = torch.randn(3, 4, 5)
        for fn in (lambda t: t.squeeze(1), torch.squeeze):
            gm = _edge_gm(fn, x)
            node = _node(gm, aten.squeeze.dims)
            self.assertFalse(is_coreai_supported_variant(node))
            self.assertTrue(NormalizeCoreAIOpArgsPass()(gm).modified)
            self.assertEqual(coreai_aten_op(node.target), aten.clone.default)
            torch.testing.assert_close(gm(x)[0], fn(x))

    def test_already_explicit_unchanged(self):
        x = torch.randn(1, 2, 4, 4)
        for fn in (
            lambda t: t.mean(dim=[1, 2]),
            lambda t: torch.amax(t, dim=1),
            lambda t: F.avg_pool2d(t, 2, 2, 0, False, False),
            lambda t: t.squeeze(0),
        ):
            gm = _edge_gm(fn, x)
            before = str(gm.graph)
            self.assertFalse(NormalizeCoreAIOpArgsPass()(gm).modified)
            self.assertEqual(str(gm.graph), before)

    def test_fully_delegated_with_default_passes(self):
        x = torch.randn(1, 2, 4, 4)
        for fn in (
            lambda t: t.mean(dim=None),
            torch.amin,
            lambda t: F.avg_pool2d(t, 2, ceil_mode=True),
            lambda t: t.squeeze((0, 1)),
            lambda t: t.squeeze(2),
            torch.squeeze,
        ):
            ep = to_edge_transform_and_lower(
                torch.export.export(_Fn(fn), (x,)),
                partitioner=[CoreAIPartitioner()],
                transform_passes=get_default_passes(),
                compile_config=get_default_compile_config(),
            ).exported_program()
            targets = {n.target for n in ep.graph.nodes if n.op == "call_function"} - {
                operator.getitem
            }
            self.assertEqual(targets, {executorch_call_delegate})


if __name__ == "__main__":
    unittest.main()
