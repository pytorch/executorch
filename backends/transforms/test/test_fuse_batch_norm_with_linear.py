# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.backends.transforms.fuse_batch_norm_with_linear import (
    FuseBatchNormWithLinearPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

_BN_OPS = {
    exir_ops.edge.aten.native_batch_norm.default,
    exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
}


class _LinearBn(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module, bn: torch.nn.Module):
        super().__init__()
        self.linear = linear
        self.bn = bn

    def forward(self, x):
        return self.bn(self.linear(x))


class _LinearBnWithRuntimeParameter(_LinearBn):
    def __init__(self, parameter: str):
        super().__init__(torch.nn.Linear(16, 8), torch.nn.BatchNorm1d(8))
        self.parameter = parameter

    def forward(self, x, parameter):
        x = torch.nn.functional.linear(
            x,
            self.linear.weight,
            parameter if self.parameter == "linear_bias" else self.linear.bias,
        )
        return torch.nn.functional.batch_norm(
            x,
            parameter if self.parameter == "running_mean" else self.bn.running_mean,
            parameter if self.parameter == "running_var" else self.bn.running_var,
            self.bn.weight,
            parameter if self.parameter == "bn_bias" else self.bn.bias,
            training=False,
            eps=self.bn.eps,
        )


def _seeded(module: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(0)
    bn = module.bn
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.weight.data.uniform_(0.5, 2)
    if bn.bias is not None:
        bn.bias.data.uniform_(-1, 1)
    return module.eval()


class TestFuseBatchNormWithLinear(unittest.TestCase):
    def _check(self, module, inputs, *, fuse=True):
        module = _seeded(module)
        expected = module(*inputs)

        edge = to_edge(
            export(module, inputs, strict=True),
            compile_config=EdgeCompileConfig(
                preserve_ops=[torch.ops.aten.linear.default],
            ),
        )
        self.assertTrue(
            any(n.target in _BN_OPS for n in edge.exported_program().graph.nodes)
        )
        result = FuseBatchNormWithLinearPass(edge.exported_program())(
            copy.deepcopy(edge.exported_program().graph_module)
        )
        self.assertEqual(result.modified, fuse)
        edge = edge.transform([FuseBatchNormWithLinearPass(edge.exported_program())])
        ep = edge.exported_program()

        self.assertEqual(any(n.target in _BN_OPS for n in ep.graph.nodes), not fuse)
        torch.testing.assert_close(ep.module()(*inputs), expected, rtol=1e-5, atol=1e-5)

    def test_linear(self):
        self._check(
            _LinearBn(torch.nn.Linear(16, 8), torch.nn.BatchNorm1d(8)),
            (torch.randn(4, 16),),
        )

    def test_linear_no_bias(self):
        self._check(
            _LinearBn(torch.nn.Linear(16, 8, bias=False), torch.nn.BatchNorm1d(8)),
            (torch.randn(4, 16),),
        )

    def test_linear_wide(self):
        self._check(
            _LinearBn(torch.nn.Linear(64, 32), torch.nn.BatchNorm1d(32)),
            (torch.randn(2, 64),),
        )

    def test_batch_norm_no_bias(self):
        self._check(
            _LinearBn(torch.nn.Linear(16, 8), torch.nn.BatchNorm1d(8, bias=False)),
            (torch.randn(4, 16),),
        )

    def test_non_rank_two_is_not_fused(self):
        for bn_type, spatial_shape in (
            (torch.nn.BatchNorm1d, ()),
            (torch.nn.BatchNorm2d, (3,)),
            (torch.nn.BatchNorm3d, (3, 2)),
        ):
            for channels in (4, 8):
                with self.subTest(batch_norm=bn_type.__name__, channels=channels):
                    self._check(
                        _LinearBn(torch.nn.Linear(16, 8), bn_type(channels)),
                        (torch.randn(2, channels, *spatial_shape, 16),),
                        fuse=False,
                    )

    def test_runtime_parameters_are_not_fused(self):
        for parameter in ("linear_bias", "bn_bias", "running_mean", "running_var"):
            with self.subTest(parameter=parameter):
                self._check(
                    _LinearBnWithRuntimeParameter(parameter),
                    (torch.randn(4, 16), torch.rand(8) + 0.5),
                    fuse=False,
                )
