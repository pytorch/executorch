# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import unittest

import torch
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.passes import get_default_passes
from executorch.exir import to_edge


class ConvBatchNorm(torch.nn.Module):
    def __init__(self, *, groups=1, bias=True, affine=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1, groups=groups, bias=bias)
        self.bn = torch.nn.BatchNorm2d(4, affine=affine)
        with torch.no_grad():
            self.bn.running_mean.copy_(torch.tensor([-0.5, 0.2, 0.7, 1.0]))
            self.bn.running_var.copy_(torch.tensor([0.3, 0.8, 1.2, 2.0]))
            if affine:
                self.bn.weight.copy_(torch.tensor([0.4, 0.8, 1.1, 1.5]))
                self.bn.bias.copy_(torch.tensor([-0.7, 0.1, 0.3, 0.9]))

    def forward(self, x):
        return self.bn(self.conv(x))


class FuseConvBatchNormPassTest(unittest.TestCase):
    def _edge(self, model, inputs):
        return to_edge(
            torch.export.export(model, inputs),
            compile_config=get_default_compile_config(),
        )

    def _batch_norms(self, program):
        return [
            node for node in program.graph.nodes if "batch_norm" in str(node.target)
        ]

    def test_default_pipeline_folds_eval_batch_norm(self):
        for config in (
            {},
            {"groups": 2, "bias": False},
            {"groups": 4, "affine": False},
        ):
            with self.subTest(config=config):
                model = ConvBatchNorm(**config).eval()
                inputs = (torch.randn(1, 4, 8, 8),)
                state = copy.deepcopy(model.state_dict())
                expected = model(*inputs)
                edge = self._edge(model, inputs)
                self.assertEqual(len(self._batch_norms(edge.exported_program())), 1)
                program = edge.transform(get_default_passes()).exported_program()
                self.assertFalse(self._batch_norms(program))
                torch.testing.assert_close(program.module()(*inputs), expected)
                torch.testing.assert_close(model.state_dict(), state)
