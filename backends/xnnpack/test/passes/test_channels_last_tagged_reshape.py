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
from executorch.backends.xnnpack.test.test_xnnpack_utils_classes import (
    OpSequencesAddConv2d,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    PassStage = RunPasses([ChannelsLastTaggedReshapePass])
    # Dictionary mapping modules to expected number of reshapes
    modules = {
        OpSequencesAddConv2d(0, 0).eval(): 0,
        OpSequencesAddConv2d(1, 1).eval(): 2,
        OpSequencesAddConv2d(2, 2).eval(): 2,
        OpSequencesAddConv2d(0, 0, True).eval(): 0,
        OpSequencesAddConv2d(1, 1, True).eval(): 2,
        OpSequencesAddConv2d(2, 2, True).eval(): 2,
    }
    to_copy_name = "executorch_exir_dialects_edge__ops_aten__to_copy_default"
    quant_name = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
    dequant_name = "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
    conv_name = "executorch_exir_dialects_edge__ops_aten_convolution_default"
    relu_name = "executorch_exir_dialects_edge__ops_aten_relu_default"

    def test_fp32_channels_last_tagged_reshape_pass(self):
        for module, num_reshape in self.modules.items():
            (
                Tester(module, (torch.randn(1, 1, 6, 6),))
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check_count(
                    {
                        self.to_copy_name: num_reshape,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_qs8_channels_last_tagged_reshape_pass(self):
        for module, num_reshape in self.modules.items():
            (
                Tester(module, (torch.randn(1, 1, 6, 6),))
                .quantize()
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check(
                    [
                        self.quant_name,
                        self.dequant_name,
                        self.to_copy_name,
                        self.quant_name,
                        self.dequant_name,
                    ]
                    * num_reshape
                )
                .run_method_and_compare_outputs()
            )

    class ConvRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    def test_fp32_channels_last_tagged_reshape_pass_conv_relu(self):
        (
            Tester(self.ConvRelu().eval(), (torch.randn(1, 1, 6, 6),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check(
                [self.to_copy_name, self.conv_name, self.relu_name, self.to_copy_name]
            )
            .run_method_and_compare_outputs()
        )

    def test_qs8_channels_last_tagged_reshape_pass_conv_relu(self):
        (
            Tester(self.ConvRelu().eval(), (torch.randn(1, 1, 6, 6),))
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check(
                [
                    self.to_copy_name,
                    self.quant_name,
                    self.dequant_name,
                    self.conv_name,
                    self.relu_name,
                    self.quant_name,
                    self.dequant_name,
                    self.to_copy_name,
                ]
            )
            .run_method_and_compare_outputs()
        )

    class Conv2dBnHardtanhMeanSequenceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 3),
                stride=[2, 2],
                padding=[1, 1],
                groups=1,
                dilation=[1, 1],
                bias=True,
            )
            self.native_batchnorm = torch.nn.BatchNorm2d(1)
            self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)
            self.eval()

        def forward(self, x):
            x = self.conv(x)
            x = self.native_batchnorm(x)
            x = self.hardtanh(x)
            x = torch.mean(x, (-1, -2), keepdim=True)
            return x

    def test_fp32_channels_last_tagged_reshape_pass_conv_bn_hardtanh_mean_seq(self):
        # Copy #1 is for input to conv, nchw -> nhwc
        # Copy #2 is for conv to _native_batch_norm_legit_no_training, nhwc -> nchw
        # Copy #3 is for input to mean, nchw -> nhwc
        # Copy #4 is for output, nhwc -> nchw

        # The graph looks like:
        # graph():
        #     %arg0_1 : [#users=1] = placeholder[target=arg0_1]
        #     %aten__to_copy_default : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._to_copy.default](args = (%arg0_1,), kwargs = {memory_format: torch.channels_last})
        #     %_param_constant0 : [#users=1] = get_attr[target=_param_constant0]
        #     %_param_constant1 : [#users=1] = get_attr[target=_param_constant1]
        #     %aten_convolution_default : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten__to_copy_default, %_param_constant0, %_param_constant1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
        #     %aten__to_copy_default_1 : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._to_copy.default](args = (%aten_convolution_default,), kwargs = {memory_format: torch.contiguous_format})
        #     %_param_constant2 : [#users=1] = get_attr[target=_param_constant2]
        #     %_param_constant3 : [#users=1] = get_attr[target=_param_constant3]
        #     %_tensor_constant0 : [#users=1] = get_attr[target=_tensor_constant0]
        #     %_tensor_constant1 : [#users=1] = get_attr[target=_tensor_constant1]
        #     %aten__native_batch_norm_legit_no_training_default : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten__to_copy_default_1, %_param_constant2, %_param_constant3, %_tensor_constant0, %_tensor_constant1, 0.1, 1e-05), kwargs = {})
        #     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default, 0), kwargs = {})
        #     %aten_hardtanh_default : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem, 0, 6), kwargs = {})
        #     %aten__to_copy_default_2 : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._to_copy.default](args = (%aten_hardtanh_default,), kwargs = {memory_format: torch.channels_last})
        #     %aten_mean_dim : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mean.dim](args = (%aten__to_copy_default_2, [-1, -2], True), kwargs = {})
        #     %aten__to_copy_default_3 : [#users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._to_copy.default](args = (%aten_mean_dim,), kwargs = {memory_format: torch.contiguous_format})
        #     return [aten__to_copy_default_3]
        (
            Tester(
                self.Conv2dBnHardtanhMeanSequenceModule().eval(),
                (torch.randn(1, 1, 6, 6),),
            )
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    self.to_copy_name: 4,
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_fp32_channels_last_tagged_reshape_pass_nhwc_view(self):
        # View can run in NHWC because channel and batch are unchanged.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                y = self.conv1(x)
                y = y.view((1, 3, 3, -1))
                return self.conv2(y)

        inputs = (torch.randn(1, 3, 8, 8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                    exir_ops.edge.aten._to_copy.default: 2,
                }
            )
        )

    def test_fp32_channels_last_tagged_reshape_pass_nchw_view_channel_modified(self):
        # View cannot run in NHWC because channel and/or batch are modified.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(6, 3, 3)

            def forward(self, x):
                y = self.conv1(x)
                y = y.view((1, 6, 6, -1))
                return self.conv2(y)

        inputs = (torch.randn(1, 3, 8, 8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                    exir_ops.edge.aten._to_copy.default: 4,
                }
            )
        )

    def test_fp32_channels_last_tagged_reshape_pass_nchw_view_batch_modified(self):
        # View cannot run in NHWC because channel and/or batch are modified.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                y = self.conv1(x)
                y = y.view((2, 3, 6, -1))
                return self.conv2(y)

        inputs = (torch.randn(1, 3, 8, 8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 2,
                    exir_ops.edge.aten.view_copy.default: 1,
                    exir_ops.edge.aten._to_copy.default: 4,
                }
            )
        )

    def test_fp32_channels_last_tagged_reshape_pass_flatten_view(self):
        # View cannot run in NHWC because tensor rank changes.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.linear1 = torch.nn.Linear(36 * 3, 1)

            def forward(self, x):
                y = self.conv1(x)
                y = y.view((x.shape[0], -1))
                return self.linear1(y)

        inputs = (torch.randn(1, 3, 8, 8),)
        tester = (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 1,
                    exir_ops.edge.aten.view_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
            .check_node_count(
                {
                    exir_ops.edge.aten.convolution.default: 1,
                    exir_ops.edge.aten.view_copy.default: 1,
                    exir_ops.edge.aten._to_copy.default: 2,
                }
            )
        )

        # Verify view is not tagged.
        graph = tester.get_artifact().exported_program().module().graph
        view_nodes = [
            n for n in graph.nodes if n.target == exir_ops.edge.aten.view_copy.default
        ]
        self.assertEqual(1, len(view_nodes))
        self.assertTrue(ChannelsLastTaggedReshapePass(None).is_nchw_node(view_nodes[0]))
