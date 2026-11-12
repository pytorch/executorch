# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import unittest

import torch
from executorch.backends.transforms.fused_quant.pre_quantize_passes.replace_adaptive_avg_pool_with_avg_pool import (
    ReplaceAdaptiveAvgPoolWithAvgPool,
)
from executorch.backends.transforms.fused_quant.test.helpers import trace
from parameterized import parameterized
from torch import nn
from torch.export import ExportedProgram

_ADAPTIVE_AVG_POOL = torch.ops.aten._adaptive_avg_pool2d.default
_AVG_POOL = torch.ops.aten.avg_pool2d.default


class _AdaptiveAvgPoolModel(nn.Module):
    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(x))


def _count(ep: ExportedProgram, target: object) -> int:
    return len(ep.graph_module.graph.find_nodes(op="call_function", target=target))


class ReplaceAdaptiveAvgPoolWithAvgPoolTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("downsample", (8, 8), (4, 4), [2, 2], [2, 2]),
            ("non_square", (12, 8), (3, 2), [4, 4], [4, 4]),
            ("identity", (6, 6), (6, 6), [1, 1], [1, 1]),
            ("single_window_dim", (8, 6), (2, 1), [4, 6], [4, 6]),
        ]
    )
    def test_divisible_replaced(
        self,
        _name: str,
        input_hw: tuple[int, int],
        output_size: tuple[int, int],
        kernel_size: list[int],
        stride: list[int],
    ) -> None:
        torch.manual_seed(0)
        model = _AdaptiveAvgPoolModel(output_size).eval()
        inp = torch.randn(1, 3, *input_hw)
        expected = model(inp)
        ep = trace(model, (inp,))
        self.assertEqual(_count(ep, _ADAPTIVE_AVG_POOL), 1)

        result = ReplaceAdaptiveAvgPoolWithAvgPool().call(ep.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(ep, _ADAPTIVE_AVG_POOL), 0)
        (avg_pool,) = ep.graph_module.graph.find_nodes(
            op="call_function", target=_AVG_POOL
        )
        self.assertEqual(avg_pool.args[1:4], (kernel_size, stride, [0, 0]))
        self.assertEqual(avg_pool.meta["val"].shape, expected.shape)
        torch.testing.assert_close(ep.module()(inp), expected)

    @parameterized.expand(
        [
            ("non_divisible", (7, 7), (3, 3)),
            ("upsample", (3, 3), (6, 6)),
        ]
    )
    def test_non_divisible_not_modified(
        self,
        _name: str,
        input_hw: tuple[int, int],
        output_size: tuple[int, int],
    ) -> None:
        model = _AdaptiveAvgPoolModel(output_size).eval()
        inp = torch.randn(1, 3, *input_hw)
        ep = trace(model, (inp,))

        result = ReplaceAdaptiveAvgPoolWithAvgPool().call(ep.graph_module)

        self.assertFalse(result.modified)
        self.assertEqual(_count(ep, _ADAPTIVE_AVG_POOL), 1)
        self.assertEqual(_count(ep, _AVG_POOL), 0)
