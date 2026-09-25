# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

_BN_OPS = {
    exir_ops.edge.aten.native_batch_norm.default,
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
}


class _ConvBn(torch.nn.Module):
    def __init__(self, conv: torch.nn.Module, bn: torch.nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x):
        return self.bn(self.conv(x))


def _seeded(module: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(0)
    bn = module.bn
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.uniform_(0.5, 2)
    bn.bias.data.uniform_(-1, 1)
    return module.eval()


class TestFuseBatchNormWithConv(unittest.TestCase):
    def _check(self, conv, bn, input_shape):
        module = _seeded(_ConvBn(conv, bn))
        x = torch.randn(input_shape)
        expected = module(x)

        ep = to_edge(
            export(module, (x,), strict=True),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()
        FuseBatchNormWithConvPass(ep)(ep.graph_module)

        self.assertFalse(any(n.target in _BN_OPS for n in ep.graph_module.graph.nodes))
        torch.testing.assert_close(ep.module()(x), expected, rtol=1e-5, atol=1e-5)

    def test_conv2d(self):
        self._check(torch.nn.Conv2d(4, 8, 3), torch.nn.BatchNorm2d(8), (1, 4, 8, 8))

    def test_grouped_conv2d(self):
        self._check(
            torch.nn.Conv2d(4, 8, 3, groups=2, bias=False),
            torch.nn.BatchNorm2d(8),
            (1, 4, 8, 8),
        )

    def test_conv_transpose1d(self):
        self._check(
            torch.nn.ConvTranspose1d(4, 8, 3), torch.nn.BatchNorm1d(8), (1, 4, 16)
        )

    def test_conv_transpose2d_same_channels(self):
        # in == out, so folding on the wrong axis runs but gives wrong values.
        self._check(
            torch.nn.ConvTranspose2d(8, 8, 3, bias=False),
            torch.nn.BatchNorm2d(8),
            (1, 8, 8, 8),
        )

    def test_conv_transpose2d(self):
        self._check(
            torch.nn.ConvTranspose2d(4, 8, 3, stride=2),
            torch.nn.BatchNorm2d(8),
            (1, 4, 8, 8),
        )

    def test_grouped_conv_transpose2d(self):
        self._check(
            torch.nn.ConvTranspose2d(4, 6, 3, groups=2),
            torch.nn.BatchNorm2d(6),
            (1, 4, 8, 8),
        )
