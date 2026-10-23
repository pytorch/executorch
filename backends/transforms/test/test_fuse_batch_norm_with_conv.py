# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
    FuseBatchNormWithConvProgramPass,
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

        for program_pass in (False, True):
            with self.subTest(program_pass=program_pass):
                ep = to_edge(
                    export(module, (x,), strict=True),
                    compile_config=EdgeCompileConfig(_check_ir_validity=False),
                ).exported_program()
                if program_pass:
                    FuseBatchNormWithConvProgramPass()(ep)
                    ep.validate()
                else:
                    FuseBatchNormWithConvPass(ep)(ep.graph_module)
                self.assertFalse(any(n.target in _BN_OPS for n in ep.graph.nodes))
                torch.testing.assert_close(
                    ep.module()(x), expected, rtol=1e-5, atol=1e-5
                )

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


class TestBatchNormFusionSafety(unittest.TestCase):
    def _edge(self, model, inputs):
        return to_edge(
            export(model, inputs),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

    def _batch_norms(self, program):
        return [
            node for node in program.graph.nodes if "batch_norm" in str(node.target)
        ]

    def _apply(self, program, mutable_buffers=()):
        return FuseBatchNormWithConvProgramPass(mutable_buffers)(program).modified

    def test_shared_convolution_output_is_preserved(self):
        class SharedOutput(ConvBatchNorm):
            def forward(self, x):
                y = self.conv(x)
                return y, self.bn(y)

        model = SharedOutput().eval()
        inputs = (torch.randn(1, 4, 8, 8),)
        program = self._edge(model, inputs).exported_program()
        self.assertFalse(self._apply(program))
        torch.testing.assert_close(program.module()(*inputs), model(*inputs))

    def test_shared_weights_are_not_modified(self):
        class SharedWeight(ConvBatchNorm):
            def forward(self, x):
                return self.bn(self.conv(x)), self.conv(x + 1)

        model = SharedWeight().eval()
        inputs = (torch.randn(1, 4, 8, 8),)
        expected = model(*inputs)
        program = self._edge(model, inputs).exported_program()
        self.assertTrue(self._apply(program))
        self.assertFalse(self._batch_norms(program))
        torch.testing.assert_close(program.module()(*inputs), expected)

    def test_channels_last_weights_preserve_strided_output(self):
        class StridedOutput(ConvBatchNorm):
            def forward(self, x):
                return super().forward(x).as_strided((4, 64), (1, 4))

        model = StridedOutput().eval().to(memory_format=torch.channels_last)
        inputs = (torch.randn(1, 4, 8, 8),)
        expected = model(*inputs)
        program = self._edge(model, inputs).exported_program()
        torch.testing.assert_close(program.module()(*inputs), expected)
        self.assertTrue(self._apply(program))
        self.assertFalse(self._batch_norms(program))
        torch.testing.assert_close(program.module()(*inputs), expected)

    def test_training_batch_norm_is_not_folded(self):
        program = self._edge(
            ConvBatchNorm().train(), (torch.randn(2, 4, 8, 8),)
        ).exported_program()
        self.assertTrue(self._batch_norms(program))
        self.assertFalse(self._apply(program))

    def test_runtime_statistics_are_not_folded(self):
        class RuntimeStats(ConvBatchNorm):
            def forward(self, x, mean, var):
                return torch.nn.functional.batch_norm(
                    self.conv(x),
                    mean,
                    var,
                    self.bn.weight,
                    self.bn.bias,
                    training=False,
                )

        inputs = (torch.randn(1, 4, 8, 8), torch.zeros(4), torch.ones(4))
        program = self._edge(RuntimeStats().eval(), inputs).exported_program()
        self.assertTrue(self._batch_norms(program))
        self.assertFalse(self._apply(program))

    def test_mutated_statistics_are_not_folded(self):
        class MutableStats(ConvBatchNorm):
            def forward(self, x):
                self.bn.running_mean.add_(1)
                return super().forward(x)

        program = self._edge(
            MutableStats().eval(), (torch.randn(1, 4, 8, 8),)
        ).exported_program()
        self.assertTrue(self._batch_norms(program))
        self.assertFalse(self._apply(program))

    def test_statistics_mutated_by_another_method_are_not_folded(self):
        program = self._edge(
            ConvBatchNorm().eval(), (torch.randn(1, 4, 8, 8),)
        ).exported_program()
        self.assertFalse(self._apply(program, {"bn.running_mean"}))
        self.assertTrue(self._batch_norms(program))

    def test_auxiliary_batch_norm_outputs_are_preserved(self):
        class AuxiliaryOutputs(ConvBatchNorm):
            def forward(self, x):
                return torch.ops.aten.native_batch_norm.default(
                    self.conv(x),
                    self.bn.weight,
                    self.bn.bias,
                    self.bn.running_mean,
                    self.bn.running_var,
                    False,
                    0.1,
                    1e-5,
                )

        inputs = (torch.randn(1, 4, 8, 8),)
        model = AuxiliaryOutputs().eval()
        program = self._edge(model, inputs).exported_program()
        self.assertFalse(self._apply(program))
        torch.testing.assert_close(program.module()(*inputs), model(*inputs))

    def test_optional_bias_and_affine(self):
        model = ConvBatchNorm(groups=4, bias=False, affine=False).eval()
        inputs = (torch.randn(1, 4, 8, 8),)
        expected = model(*inputs)
        program = self._edge(model, inputs).exported_program()
        self.assertTrue(self._apply(program))
        self.assertFalse(self._batch_norms(program))
        torch.testing.assert_close(program.module()(*inputs), expected)
        self.assertFalse(self._apply(program))


class TestGraphModuleBatchNormFusionSafety(TestBatchNormFusionSafety):
    def _apply(self, program, mutable_buffers=()):
        fusion = FuseBatchNormWithConvPass(program, mutable_buffers=mutable_buffers)
        return fusion(program.graph_module).modified
