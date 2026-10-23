# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.fuse_pad_into_conv_pass import (
    FusePadIntoConvPass,
)
from executorch.backends.xnnpack._passes.insert_pad_qdq import InsertPadQDQPass
from executorch.backends.xnnpack._passes.remove_redundant_copy_pass import (
    RemoveRedundantCopyPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass


class TestFusePadIntoConvPass(unittest.TestCase):
    PassStage = RunPasses(
        [
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            InsertPadQDQPass,
            ChannelsLastTaggedReshapePass,
            RemoveRedundantCopyPass,
            FusePadIntoConvPass,
        ]
    )

    def setUp(self):
        torch._dynamo.reset()

    class ConvPadConv(torch.nn.Module):
        """Asymmetric pad before a 4x4 filter, as in StyleGAN-style blur. The
        leading conv puts the activation in NHWC so the pad reaches the second
        conv without a layout copy in between."""

        def __init__(self, value=0.0, depthwise=False):
            super().__init__()
            self.value = value
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = (
                torch.nn.Conv2d(8, 8, 4, groups=8, bias=False)
                if depthwise
                else torch.nn.Conv2d(8, 4, 4, padding=1)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.pad(x, (2, 1, 2, 1), value=self.value)
            return self.conv2(x)

    class ConvPadConvTranspose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = torch.nn.ConvTranspose2d(8, 4, 3)

        def forward(self, x):
            x = self.conv1(x)
            return self.conv2(torch.nn.functional.pad(x, (1, 1, 1, 1)))

    class PadOnInput(torch.nn.Module):
        """The pad sits on the graph input, so the NCHW -> NHWC copy lands
        between the pad and the conv and nothing folds."""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(8, 4, 4, padding=1)

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (2, 1, 2, 1)))

    def _run_passes(self, module, inputs, quantize=False):
        tester = Tester(module.eval(), inputs)
        if quantize:
            tester.quantize()
        tester.export().to_edge().run_passes(self.PassStage)
        return tester.get_artifact().exported_program().graph_module

    def _fused_paddings(self, graph_module):
        return [
            FusePadIntoConvPass.get_fused_padding(n)
            for n in graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
            and FusePadIntoConvPass.get_fused_padding(n) is not None
        ]

    def _folded_pads(self, graph_module):
        return [
            n
            for n in graph_module.graph.nodes
            if n.target == exir_ops.edge.aten.constant_pad_nd.default
            and FusePadIntoConvPass.is_folded_pad(n)
        ]

    def _assert_folded(self, module, inputs, expected_trbl):
        graph_module = self._run_passes(module, inputs)
        self.assertEqual(len(self._folded_pads(graph_module)), 1)
        self.assertEqual(self._fused_paddings(graph_module), [expected_trbl])

    def _assert_not_folded(self, module, inputs, quantize=False):
        graph_module = self._run_passes(module, inputs, quantize)
        self.assertEqual(self._folded_pads(graph_module), [])
        self.assertEqual(self._fused_paddings(graph_module), [])

    def _run(self, module, inputs, atol=1e-03, rtol=1e-03, quantize=False):
        tester = Tester(module.eval(), inputs)
        if quantize:
            tester.quantize()
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=atol, rtol=rtol, qtol=1)
        )

    def test_fp32_conv_pad_conv(self):
        inputs = (torch.randn(1, 3, 10, 12),)
        self._assert_folded(self.ConvPadConv(), inputs, (3, 2, 2, 3))
        self._run(self.ConvPadConv(), inputs)

    def test_fp32_conv_pad_depthwise_conv(self):
        inputs = (torch.randn(1, 3, 10, 12),)
        self._assert_folded(self.ConvPadConv(depthwise=True), inputs, (2, 1, 1, 2))
        self._run(self.ConvPadConv(depthwise=True), inputs)

    def test_fp16_conv_pad_conv(self):
        inputs = (torch.randn(1, 3, 10, 12).half(),)
        self._assert_folded(self.ConvPadConv().half(), inputs, (3, 2, 2, 3))
        self._run(self.ConvPadConv().half(), inputs, atol=2e-2, rtol=2e-2)

    def test_fp32_nonzero_pad_value_not_fused_still_correct(self):
        inputs = (torch.randn(1, 3, 10, 12),)
        self._assert_not_folded(self.ConvPadConv(value=1.5), inputs)
        self._run(self.ConvPadConv(value=1.5), inputs)

    def test_fp32_conv_transpose_not_fused_still_correct(self):
        inputs = (torch.randn(1, 3, 5, 5),)
        self._assert_not_folded(self.ConvPadConvTranspose(), inputs)
        self._run(self.ConvPadConvTranspose(), inputs)

    def test_fp32_pad_on_graph_input_not_fused_still_correct(self):
        inputs = (torch.randn(1, 8, 9, 7),)
        self._assert_not_folded(self.PadOnInput(), inputs)
        self._run(self.PadOnInput(), inputs)

    def test_qs8_conv_pad_conv_not_fused_still_correct(self):
        inputs = (torch.randn(1, 3, 10, 12),)
        self._assert_not_folded(self.ConvPadConv(), inputs, quantize=True)
        self._run(self.ConvPadConv(), inputs, quantize=True)
