# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.test.harness.stages.stage import StageType
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.test_xnnpack_utils_classes import (
    OpSequencesAddConv2d,
)
from executorch.backends.xnnpack.test.tester import Quantize, RunPasses, Tester
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_quant,
    is_tagged_as_implicit_q_dq,
)


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

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
    choose_qparams_name = (
        "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_tensor"
    )
    dynamic_quant_name = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_tensor"

    def run_tester(self, module, inputs):
        tester = Tester(
            module.eval(),
            inputs,
        )
        tester.export().to_edge_transform_and_lower().check_not(
            ["executorch_exir_dialects_edge__ops_aten__to_copy_default"]
        ).to_executorch().serialize().run_method_and_compare_outputs()

    class LinearConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.linear1 = torch.nn.Linear(4, 3)

        def forward(self, x):
            y = self.linear1(x)
            return self.conv1(y)

    LinearConvModule = LinearConv()

    class ConvLinearConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.linear1 = torch.nn.Linear(4, 4)

        def forward(self, x):
            y = self.conv1(x)
            return self.linear1(y)

    ConvLinearConvModule = ConvLinearConv()

    class Bilinear(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )

    BilinearModule = Bilinear()

    class TwoConvAdd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(5, 16, 3, padding=1)

        def forward(self, x1, x2):
            y1 = self.conv1(x1)
            y2 = self.conv2(x2)
            return torch.add(y1, y2)

    TwoConvAddModule = TwoConvAdd()

    def test_two_conv_add(self):
        x1 = torch.randn(1, 3, 8, 8)
        x2 = torch.randn(1, 5, 8, 8)

        # Test with regular format inputs
        self.run_tester(self.TwoConvAddModule, (x1, x2))

        # Test with channels_last format inputs
        x1_cl = x1.to(memory_format=torch.channels_last)
        x2_cl = x2.to(memory_format=torch.channels_last)
        self.run_tester(self.TwoConvAddModule, (x1_cl, x2_cl))

        # Test with mixed format inputs
        self.run_tester(self.TwoConvAddModule, (x1_cl, x2))
        self.run_tester(self.TwoConvAddModule, (x1, x2_cl))

        # Verify the pass adds the expected number of to_copy operations
        (
            Tester(self.TwoConvAddModule, (x1, x2))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    self.to_copy_name: 3,  # 2 for inputs to conv, 1 for outputs from add
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_conv_linear_dim_order_swaps(self):
        self.run_tester(self.LinearConvModule, (torch.randn(1, 3, 6, 4),))
        self.run_tester(
            self.LinearConvModule,
            (torch.randn(1, 3, 6, 4).to(memory_format=torch.channels_last),),
        )

    def test_linear_conv_dim_order_swaps(self):
        self.run_tester(self.ConvLinearConvModule, (torch.randn(1, 3, 6, 6),))
        self.run_tester(
            self.ConvLinearConvModule,
            (torch.randn(1, 3, 6, 6).to(memory_format=torch.channels_last),),
        )

    def test_nhwc_nchw_input_on_nhwc_op(self):
        self.run_tester(
            self.BilinearModule,
            (
                torch.arange(8)
                .reshape(1, 2, 2, 2)
                .to(torch.float32)
                .to(memory_format=torch.channels_last),
            ),
        )

        self.run_tester(
            self.BilinearModule,
            (torch.arange(8).reshape(1, 2, 2, 2).to(torch.float32),),
        )

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

    class LinearConvDimSwap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.linear1 = torch.nn.Linear(4, 3)

        def forward(self, x):
            y = self.linear1(x)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            return self.conv1(y)

    LinearConvDimSwapModule = LinearConvDimSwap()

    def test_conv_linear_dim_order_swap_partitioner(self):
        self.run_tester(self.LinearConvDimSwapModule, (torch.randn(1, 3, 6, 4),))

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

    class Conv2dDynamicQuant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 10, 3)

        def forward(self, x):
            return self.conv(x)

    def test_dq_conv2d_channels_last_tagged_reshape_pass(self) -> None:
        (
            Tester(self.Conv2dDynamicQuant().eval(), (torch.randn(1, 3, 8, 8),))
            .quantize(
                Quantize(
                    quantization_config=get_symmetric_quantization_config(
                        is_dynamic=True
                    )
                )
            )
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check(
                [
                    self.to_copy_name,
                    self.choose_qparams_name,
                    self.dynamic_quant_name,
                    self.dequant_name,
                    self.conv_name,
                    self.to_copy_name,
                ]
            )
            .run_method_and_compare_outputs()
        )

    class ConvAddConvOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3)
            self.conv2 = torch.nn.Conv2d(16, 16, 3)

        def forward(self, x):
            y = self.conv1(x)
            z = torch.add(y, 1.0)
            out1 = self.conv2(z)
            out2 = z
            return out1, out2

    ConvAddConvOutputModule = ConvAddConvOutput()

    def test_conv_add_conv_output(self):
        x = torch.randn(1, 3, 8, 8)

        self.run_tester(self.ConvAddConvOutput().eval(), (x,))

        x_cl = x.to(memory_format=torch.channels_last)
        self.run_tester(self.ConvAddConvOutput().eval(), (x_cl,))

    class ThreeOutputsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(6, 6)

        def forward(self, x):
            conv1_out = self.conv1(x)
            conv2_out = self.conv2(x)
            linear_out = self.linear(x)

            return linear_out, conv1_out, conv2_out

    ThreeOutputsModelModule = ThreeOutputsModel()

    def test_three_outputs_model(self):
        x = torch.randn(1, 3, 6, 6)

        self.run_tester(self.ThreeOutputsModelModule.eval(), (x,))

        x_cl = x.to(memory_format=torch.channels_last)
        self.run_tester(self.ThreeOutputsModelModule.eval(), (x_cl,))

    class ConvQDQModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    def _check_implicit_q_dq_tagging(
        self, graph_module: torch.fx.GraphModule, expected_tagging: list[bool]
    ):
        q_dq_nodes = []
        for node in graph_module.graph.nodes:
            if is_quant(node) or is_dequant(node):
                q_dq_nodes.append(node)

        # Check that we have the expected number of nodes
        self.assertEqual(
            len(q_dq_nodes),
            len(expected_tagging),
            f"Expected {len(expected_tagging)} q/dq nodes but found {len(q_dq_nodes)}",
        )

        actual_tagging = []
        for node in q_dq_nodes:
            is_tagged = is_tagged_as_implicit_q_dq(node)
            actual_tagging.append(is_tagged)

        self.assertEqual(
            actual_tagging,
            expected_tagging,
            f"Q/DQ node tagging mismatch. Expected: {expected_tagging}, Actual: {actual_tagging}",
        )

    def test_q_dq_nodes_around_copy_are_tagged(self):
        # Create a model with conv operation
        model = self.ConvQDQModule().eval()
        input_tensor = torch.randn(1, 3, 8, 8)

        tester = (
            Tester(model, (input_tensor,))
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check(
                [
                    self.dequant_name,
                    self.quant_name,
                    self.dequant_name,
                    self.to_copy_name,
                    self.quant_name,
                    self.dequant_name,
                    self.conv_name,
                    self.quant_name,
                    self.dequant_name,
                    self.to_copy_name,
                    self.quant_name,
                    self.dequant_name,
                ]
            )
        )

        artifact = tester.get_artifact(StageType.RUN_PASSES)
        graph_module = artifact.exported_program().graph_module

        # Check implicit q/dq tagging
        expected_tagging = [False, False, True, True, False, False, True, True, False]
        self._check_implicit_q_dq_tagging(graph_module, expected_tagging)

        # Compare outputs
        tester.run_method_and_compare_outputs()
