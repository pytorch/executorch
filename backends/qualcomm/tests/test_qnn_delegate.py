# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
import unittest

import torch

from executorch.backends.qualcomm.tests.qnn_test_utils import (
    get_qdq_module,
    save_model_and_expected_output,
    TestQNN,
)
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.models.inception_v4 import InceptionV4Model
from executorch.examples.models.mobilebert import MobileBertModelExample
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.examples.models.mobilenet_v3 import MV3Model
from executorch.exir.backend.backend_api import disable_validation


@unittest.skip("skip this for now until e2e test is enabled")
class TestQNNFloatingPoint(TestQNN):
    def test_qnn_backend_sequential_conv2d(self):
        class TwoConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=True,
                )
                self.second = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=True,
                )

            def forward(self, x):
                return self.second(self.first(x))

        instance = TwoConv()
        example_inputs = (torch.ones([1, 1, 3, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_two_conv2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_element_wise_add(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add = torch.add

            def forward(self, x, y):
                return self.add(x, y)

        instance = Add()
        example_inputs = (torch.randn(2, 5, 1, 3), torch.randn(2, 5, 1, 3))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_add_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_relu(self):
        class Relu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        instance = Relu()
        example_inputs = (torch.ones([2, 5, 1, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_relu_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_hardtanh(self):
        class Hardtanh(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)

            def forward(self, x):
                return self.hardtanh(x)

        instance = Hardtanh()
        example_inputs = (torch.ones([2, 5, 1, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_hardtanh_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_mean_dim(self):
        class MeanDim(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (-1, -2), keepdim=True)

        instance = MeanDim()
        example_inputs = (torch.ones([2, 5, 1, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_mean_dim_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_linear(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5).eval()

            def forward(self, x):
                return self.linear(x)

        instance = Linear()
        example_inputs = (torch.ones([3, 4]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_linear_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_reshape(self):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.reshape(1, 12)

        instance = Reshape()
        example_inputs = (torch.ones([3, 4]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_reshape_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_sequential_conv2d_bn_hardtanh_mean(self):
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 1
        out_channels = 1

        class Conv2dBnHardtanhMeanSequenceModule(torch.nn.Module):
            def __init__(self):
                super(Conv2dBnHardtanhMeanSequenceModule, self).__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.conv.weight = torch.nn.Parameter(
                    10 * torch.randn(self.conv.weight.size())
                )
                self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=100)
                self.eval()

            def forward(self, x):
                x1 = self.conv(x)
                x2 = self.native_batchnorm(x1)
                x3 = self.hardtanh(x2)
                x4 = torch.mean(x3, (1), keepdim=True)
                return x4

        instance = Conv2dBnHardtanhMeanSequenceModule()
        example_inputs = (torch.ones(1, 1, 6, 6),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_conv2d_bn_hardtanh_mean_sequence_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_residual_block(self):
        groups = 1
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 32
        out_channels = 32

        class ResidualBlockModule(torch.nn.Module):
            def __init__(self):
                super(ResidualBlockModule, self).__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.conv.weight = torch.nn.Parameter(
                    10 * torch.randn(self.conv.weight.size())
                )
                self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=1000)
                self.eval()

            def forward(self, x):
                x1 = self.conv(x)
                x2 = self.native_batchnorm(x1)
                x3 = self.conv(x2)
                x4 = self.native_batchnorm(x3)
                x5 = self.hardtanh(x4)
                x6 = torch.add(x5, x2)
                return x6

        instance = ResidualBlockModule()
        example_inputs = (torch.ones(1, in_channels, 28, 28),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_residual_block_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_mobilenet_v2(self):
        from executorch.examples.models.mobilenet_v2 import MV2Model

        instance = MV2Model().get_eager_model().eval()
        example_inputs = MV2Model().get_example_inputs()
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_mobilenet_v2_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_select_copy(self):
        class SelectCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=True,
                )

                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=1000)

            def forward(self, x):
                return self.conv(x)[0, 1, 1:2]

        instance = SelectCopy()
        example_inputs = (torch.randn([1, 3, 3, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_select_copy_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.unsqueeze(0)

        instance = Unsqueeze()
        example_inputs = (torch.randn([1, 3, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_unsqueeze_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_mul(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x * y

        instance = Mul()
        example_inputs = (torch.randn([1, 3, 3]), torch.randn([1, 3, 3]))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_mul_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_cat2(self):
        class Cat(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.cat((x, y), axis=2)

        instance = Cat()
        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_cat2_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_cat3(self):
        class Cat(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.concat((y, y, x), axis=2)

        instance = Cat()
        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_cat3_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_cat4(self):
        class Cat(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.cat((y, y, x, x), axis=2)

        instance = Cat()
        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_cat4_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_deeplabv3(self):
        instance = DeepLabV3ResNet101Model().get_eager_model().eval()
        example_inputs = DeepLabV3ResNet101Model().get_example_inputs()
        # TODO: Due to trigger maximum recursion depth exceeded, need to check it.
        disable_validation()
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_deeplabv3_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_maxpool2d(self):
        class MaxPool2d(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    ceil_mode=True,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)

        instance = MaxPool2d()
        example_inputs = (torch.randn(4, 3, 24, 24),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_maxpool2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_arange(self):
        class Arange(torch.nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self, y):
                return torch.arange(self.x, dtype=torch.float32) + y

        instance = Arange(5)
        example_inputs = (torch.randn(5),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_arange_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_clamp(self):
        class Clamp(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.clamp(x, max=0)

        instance = Clamp()
        example_inputs = (torch.randn((9, 4, 5, 3)),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_clamp_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_cast(self):
        class Cast(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.type(torch.IntTensor)

        instance = Cast()
        example_inputs = (10 * torch.rand((9, 4, 5, 3)),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_cast_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_element_wise_sub(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = torch.sub

            def forward(self, x, y):
                return self.sub(x, y)

        instance = Sub()
        example_inputs = (
            torch.ones([2, 5, 1, 3]),
            torch.ones([4, 1]),
        )
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_sub_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_element_wise_ceil(self):
        class Ceil(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ceil = torch.ceil

            def forward(self, x):
                return self.ceil(x)

        instance = Ceil()
        example_inputs = (torch.randn([2, 5, 1, 3]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_ceil_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_element_wise_sub_constant(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, extended_attention_mask):
                dtype = extended_attention_mask.dtype
                return (1.0 - extended_attention_mask) * torch.finfo(dtype).min

        instance = Sub()
        example_inputs = (torch.ones([28], dtype=torch.float32),)
        buffer = self.lower_module_and_test_output(
            instance,
            example_inputs,
            is_fp16=True,
        )
        model_name = "qnn_sub_constant_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_interpolate(self):
        class StaticResizeBilinear2DSizeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                output_shape = [dim * 2 for dim in x.shape[-2:]]
                a = torch.nn.functional.interpolate(
                    x,
                    size=list(torch.randn(output_shape).shape),
                    mode="bilinear",
                    align_corners=False,
                )
                return a

        instance = StaticResizeBilinear2DSizeModule()
        example_inputs = (torch.randn(2, 3, 4, 5),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_interpolate_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_inceptionv4(self):
        model = InceptionV4Model()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_inceptionv4_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_avg_pool2d(self):
        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(1, 1),
                    count_include_pad=False,
                )

            def forward(self, x):
                return self.avgPool(x)

        instance = AvgPoolModule()

        example_inputs = (torch.ones(1, 3, 2, 2),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_avg_pool2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_element_wise_div(self):
        class Div(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.div = torch.divide

            def forward(self, x, y):
                return self.div(x, y)

        instance = Div()
        example_inputs = (
            torch.randn([2, 5, 1, 3]),
            torch.randn([4, 1]),
        )
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_div_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_slice_copy(self):
        class SliceCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.position_ids = torch.randn([1, 512])

            def forward(self, x, y):
                seq_length = y.size()[1]
                return x[:, :seq_length] + self.position_ids[:, :seq_length]

        instance = SliceCopy()
        example_inputs = (
            torch.randn([1, 512]),
            torch.randn([1, 8]),
        )
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_slice_copy_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_softmax(self):
        class Softmax(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softmax(x, dim=-1)

        instance = Softmax()
        example_inputs = (torch.randn([1, 4, 8, 8]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_softmax_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_pad(self):
        class Pad(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(
                    x[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0, mode="constant"
                )

        instance = Pad()
        example_inputs = (torch.randn([1, 8, 128]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_pad_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_bmm(self):
        class Bmm(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        instance = Bmm()
        example_inputs = (
            torch.randn([4, 8, 32]),
            torch.randn([4, 32, 8]),
        )
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_bmm_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    @unittest.expectedFailure
    def test_qnn_backend_embedding(self):
        class Embedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 3)

            def forward(self, x):
                return self.embedding(x)

        instance = Embedding()
        # QNN does not support int64 datatype
        example_inputs = (torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_embedding_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_view(self):
        class View(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first_size = 2
                self.second_size = 256

            def forward(self, x, y):
                new_shape = x.size()[:-1] + (self.first_size, self.second_size)
                x = x.view(new_shape)
                x = x.permute(0, 2, 1, 3)
                return torch.matmul(x, y.transpose(-1, -2))

        instance = View()
        example_inputs = (torch.randn([1, 8, 512]), torch.randn([1, 2, 8, 256]))
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_view_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_expand_copy(self):
        class ExpandCopy(torch.nn.Module):
            def forward(self, x):
                return x.expand(3, 4)

        instance = ExpandCopy()
        example_inputs = (torch.randn([3, 1]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_expand_copy_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_mobilebert(self):
        instance = MobileBertModelExample().get_eager_model().eval()
        example_inputs = MobileBertModelExample().get_example_inputs()
        # TODO: Due to triggering maximum recursion depth exceeded, need to check it.
        disable_validation()
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_mobilebert_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_sqrt(self):
        class Sqrt(torch.nn.Module):
            def forward(self, x):
                return x / math.sqrt(64)

        instance = Sqrt()
        example_inputs = (torch.ones([3, 1]),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_sqrt_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_hardswish(self):
        class Hardswish(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardswish = torch.nn.Hardswish()

            def forward(self, x):
                return self.hardswish(x)

        instance = Hardswish()
        example_inputs = (torch.randn(2, 5, 1, 3),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_hardswish_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_hardsigmoid(self):
        class Hardswigmoid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardsigmoid = torch.nn.Hardsigmoid()

            def forward(self, x):
                return self.hardsigmoid(x)

        instance = Hardswigmoid()
        example_inputs = (torch.randn(2, 5, 1, 3),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_hardsigmoid_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_tanh(self):
        class Tanh(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.tanh(x)

        instance = Tanh()
        example_inputs = (torch.randn(2, 5, 1, 3),)
        buffer = self.lower_module_and_test_output(
            instance, example_inputs, is_fp16=True
        )
        model_name = "qnn_tanh_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_mobilenet_v3(self):
        model = MV3Model()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        buffer = self.lower_module_and_test_output(instance, example_inputs)
        model_name = "qnn_mobilenet_v3_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_edsr(self):
        model = EdsrModel()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        buffer = self.lower_module_and_test_output(instance, example_inputs)
        model_name = "qnn_edsr_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)


@unittest.skip("skip this for now until e2e test is enabled")
class TestQNNQuantized(TestQNN):
    def test_qnn_backend_ptq_sequential_conv2d(self):
        class TwoConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=True,
                )
                self.second = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=True,
                )

            def forward(self, x):
                return self.second(self.first(x))

        instance = TwoConv()
        example_inputs = (torch.randn([1, 1, 3, 3]),)
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_two_conv2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_element_wise_add(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add = torch.add

            def forward(self, x, y):
                return self.add(x, y)

        example_inputs = (
            torch.ones([2, 5, 1, 3]),
            torch.ones([4, 1]),
        )
        quant_instance = get_qdq_module(Add(), example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_add_model"
        save_model_and_expected_output(Add(), buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_relu(self):
        class Relu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        example_inputs = (torch.ones([2, 5, 1, 3]),)
        quant_instance = get_qdq_module(Relu(), example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_relu_model"
        save_model_and_expected_output(Relu(), buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_linear(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5).eval()

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.ones([3, 4]),)
        instance = Linear()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_linear_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_simple_model(self):
        class SimpleModel(torch.nn.Module):
            k_sz = 32

            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    self.k_sz, self.k_sz, 3, padding=1, bias=True
                )
                self.conv2 = torch.nn.Conv2d(
                    self.k_sz, self.k_sz, 3, padding=1, bias=True
                )
                self.conv3 = torch.nn.Conv2d(
                    self.k_sz, self.k_sz, 3, padding=1, bias=False
                )
                self.conv4 = torch.nn.Conv2d(
                    self.k_sz, self.k_sz, 3, padding=1, bias=False
                )
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)
                self.relu = torch.nn.ReLU()
                self.batch_norm = torch.nn.BatchNorm2d(self.k_sz)
                self.add = torch.add
                self.mean = torch.mean
                self.reshape = torch.reshape
                self.linear = torch.nn.Linear(4, 10)
                self.permute = torch.permute
                self.eval()

            def forward(self, x, y):
                x1 = self.conv1(x)
                x2 = self.batch_norm(x1)
                x3 = self.relu(x2)
                x4 = self.conv2(x3)
                x5 = self.relu(x4)
                y1 = self.conv3(y)
                y2 = self.batch_norm(y1)
                y3 = self.relu(y2)
                y4 = self.conv4(y3)
                y5 = self.relu(y4)
                z = self.add(x5, y5)
                z1 = self.permute(z, (0, 3, 2, 1))
                z2 = torch.mean(z1, [1, 2], True)
                z3 = self.reshape(z2, (8, -1))
                z4 = self.linear(z3)
                z5 = self.hardtanh(z4)
                return z5

        instance = SimpleModel()
        k_sz = instance.k_sz
        example_inputs = (torch.ones(1, k_sz, 28, 28), torch.ones(1, k_sz, 28, 28))
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_simple_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_maxpool2d(self):
        class ConvMaxPool2d(torch.nn.Module):
            def __init__(self):
                super(ConvMaxPool2d, self).__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=2,
                    out_channels=2,
                    kernel_size=(1, 1),
                    padding=1,
                    bias=True,
                )
                self.pool = torch.nn.MaxPool2d(1, 1)

            def forward(self, x):
                return self.pool(self.conv(x))

        example_inputs = (torch.rand(1, 2, 14, 14),)
        instance = ConvMaxPool2d()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_maxpool2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_avgpool2d(self):
        class Conv2dAvgPool2d(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)

            def forward(self, x):
                return self.avgpool(self.conv(x))

        example_inputs = (torch.randn(16, 3, 16, 16),)
        instance = Conv2dAvgPool2d()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_avgpool2d_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_cat(self):
        class Conv2dWithCat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                z = torch.cat([x, y], dim=1)
                return z

        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        instance = Conv2dWithCat()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_cat_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_interpolate(self):
        class StaticResizeBilinear2DSizeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                output_shape = [dim * 2 for dim in x.shape[-2:]]
                a = torch.nn.functional.interpolate(
                    x,
                    size=list(torch.randn(output_shape).shape),
                    mode="bilinear",
                    align_corners=False,
                )
                return a

        instance = StaticResizeBilinear2DSizeModule()
        example_inputs = (torch.randn(2, 3, 4, 5),)
        quant_instance = get_qdq_module(instance, example_inputs)

        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_interpolate_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return x[:, None, None, :]

        example_inputs = (torch.ones([3, 4]),)
        instance = Unsqueeze()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_unsqueeze_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_constant_mul(self):
        class ConstantMul(torch.nn.Module):
            def forward(self, x):
                return x * 255

        example_inputs = (torch.ones([3, 4]),)
        instance = ConstantMul()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_constant_mul_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_hardswish(self):
        class Hardswish(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardswish = torch.nn.Hardswish()

            def forward(self, x):
                return self.hardswish(x)

        instance = Hardswish()
        example_inputs = (torch.randn(2, 5, 1, 3),)
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_hardswish_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_hardsigmoid(self):
        class Hardswigmoid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardsigmoid = torch.nn.Hardsigmoid()

            def forward(self, x):
                return self.hardsigmoid(x)

        instance = Hardswigmoid()
        example_inputs = (torch.randn(2, 5, 3, 3),)
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_hardsigmoid_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_bmm(self):
        class Bmm(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        instance = Bmm()
        example_inputs = (
            torch.randn([4, 8, 32]),
            torch.randn([4, 32, 8]),
        )
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_bmm_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_mean_dim(self):
        class MeanDim(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (-1, -2), keepdim=True)

        instance = MeanDim()
        example_inputs = (torch.ones([2, 5, 1, 3]),)
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_mean_dim_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_pixel_shuffle(self):
        class PixelShuffle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pixel_shuffle = torch.nn.PixelShuffle(2)

            def forward(self, x):
                return self.pixel_shuffle(x)

        instance = PixelShuffle()
        example_inputs = (torch.ones([2, 4, 3, 3]),)
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_pixel_shuffle_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_mobilenet_v2(self):
        model = MV2Model()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_mobilenet_v2_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_mobilenet_v3(self):
        model = MV3Model()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_mobilenet_v3_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_inceptionv4(self):
        instance = InceptionV4Model().get_eager_model().eval()
        example_inputs = InceptionV4Model().get_example_inputs()
        quant_instance = get_qdq_module(instance, example_inputs)
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_inceptionv4_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_deeplabv3(self):
        instance = DeepLabV3ResNet101Model().get_eager_model().eval()
        example_inputs = DeepLabV3ResNet101Model().get_example_inputs()
        quant_instance = get_qdq_module(instance, example_inputs)
        # TODO: Due to trigger maximum recursion depth exceeded, need to check it.
        disable_validation()
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_deeplabv3_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_edsr(self):
        from executorch.examples.qualcomm.scripts.edsr import annotate_forward

        model = EdsrModel()
        instance = model.get_eager_model().eval()
        example_inputs = model.get_example_inputs()
        quant_instance = get_qdq_module(
            instance,
            example_inputs,
            is_conv_per_channel=False,
            custom_quant_annotations=(annotate_forward,),
        )
        buffer = self.lower_module_and_test_output(quant_instance, example_inputs)
        model_name = "ptq_qnn_edsr_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)

    def test_qnn_backend_ptq_mobilebert(self):
        instance = MobileBertModelExample().get_eager_model().eval()
        example_inputs = MobileBertModelExample().get_example_inputs()
        quant_instance = get_qdq_module(instance, example_inputs)
        # TODO: Due to triggering maximum recursion depth exceeded, need to check it.
        disable_validation()
        buffer = self.lower_module_and_test_output(
            quant_instance,
            example_inputs,
        )
        model_name = "ptq_qnn_mobilebert_model"
        save_model_and_expected_output(instance, buffer, example_inputs, model_name)


if __name__ == "__main__":
    unittest.main()
