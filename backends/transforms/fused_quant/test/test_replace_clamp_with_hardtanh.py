# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import unittest

import torch
from executorch.backends.transforms.fused_quant.pre_quantize_passes.replace_clamp_with_hardtanh import (
    ReplaceClampWithHardtanh,
)
from executorch.backends.transforms.fused_quant.test.helpers import trace
from parameterized import parameterized
from torch import nn
from torch.export import ExportedProgram

_CLAMP = torch.ops.aten.clamp.default
_HARDTANH = torch.ops.aten.hardtanh.default


class _ClampModel(nn.Module):
    def __init__(self, min_val: float | None, max_val: float | None) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * 2, self.min_val, self.max_val)


def _count(ep: ExportedProgram, target: object) -> int:
    return len(ep.graph_module.graph.find_nodes(op="call_function", target=target))


class ReplaceClampWithHardtanhTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("unit_range", 0.0, 1.0),
            ("int_bounds", -1, 6),
            ("equal_bounds", 0.5, 0.5),
        ]
    )
    def test_clamp_with_both_bounds_replaced(
        self, _name: str, min_val: float, max_val: float
    ) -> None:
        model = _ClampModel(min_val, max_val).eval()
        inp = torch.randn(2, 8)
        expected = model(inp)
        ep = trace(model, (inp,))
        self.assertEqual(_count(ep, _CLAMP), 1)

        result = ReplaceClampWithHardtanh().call(ep.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(ep, _CLAMP), 0)
        (hardtanh,) = ep.graph_module.graph.find_nodes(
            op="call_function", target=_HARDTANH
        )
        self.assertEqual(hardtanh.args[1:], (float(min_val), float(max_val)))
        torch.testing.assert_close(ep.module()(inp), expected)

    @parameterized.expand(
        [
            ("min_only", 0.0, None),
            ("max_only", None, 1.0),
            ("min_above_max", 1.0, 0.0),
        ]
    )
    def test_other_clamps_not_modified(
        self, _name: str, min_val: float | None, max_val: float | None
    ) -> None:
        ep = trace(_ClampModel(min_val, max_val).eval(), (torch.randn(2, 8),))

        result = ReplaceClampWithHardtanh().call(ep.graph_module)

        self.assertFalse(result.modified)
        self.assertEqual(_count(ep, _CLAMP), 1)
        self.assertEqual(_count(ep, _HARDTANH), 0)

    def test_integer_clamp_not_modified(self) -> None:
        model = _ClampModel(0, 4).eval()
        ep = trace(model, (torch.arange(16).reshape(2, 8),))

        result = ReplaceClampWithHardtanh().call(ep.graph_module)

        self.assertFalse(result.modified)
        self.assertEqual(_count(ep, _CLAMP), 1)
