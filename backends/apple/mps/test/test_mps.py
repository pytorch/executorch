#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import inspect
import logging
import random
import unittest
from enum import Enum

import torch
from executorch.backends.apple.mps.test.test_mps_models import MPS_MODEL_NAME_TO_MODEL
from executorch.backends.apple.mps.test.test_mps_utils import (
    OpSequencesAddConv2d,
    randomize_bn,
    TestMPS,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.exir.tests.models import (
    BasicSinMax,
    CompositeDelegateModule,
    ElementwiseAdd,
    Emformer,
    MLP,
    ModelWithUnusedArg,
    Mul,
    Repeat,
)


class MODEL_TYPE(Enum):
    EXIR_DEFAULT_MODEL = 0
    EXIR_TEST_MODEL = 1
    MPS_TEST_MODEL = 2


EXIR_MODEL_NAME_TO_MODEL = {
    "repeat": lambda: (Repeat(), Repeat().get_random_inputs()),
    "model_with_unused_arg": lambda: (
        ModelWithUnusedArg(),
        ModelWithUnusedArg().get_random_inputs(),
    ),
    "mlp": lambda: (MLP(), MLP().get_random_inputs()),
    "mul_2": lambda: (Mul(), Mul().get_random_inputs()),
    "element_wise_add": lambda: (
        ElementwiseAdd(),
        ElementwiseAdd().get_random_inputs(),
    ),
    "basic_sin_max": lambda: (BasicSinMax(), BasicSinMax().get_random_inputs()),
    "composite_delegate_module": lambda: (
        CompositeDelegateModule(),
        CompositeDelegateModule().get_random_inputs(),
    ),
    "emformer": lambda: (Emformer(), Emformer().get_random_inputs()),
}


def run_model(
    model: str,
    model_type: MODEL_TYPE = MODEL_TYPE.EXIR_DEFAULT_MODEL,
    use_fp16: bool = False,
    lowering_func=None,
):
    logging.info(f"Step 1: Retrieving model: {model}...")
    if model_type == MODEL_TYPE.EXIR_DEFAULT_MODEL:
        m, m_inputs = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model])
    elif model_type == MODEL_TYPE.EXIR_TEST_MODEL:
        m, m_inputs = EXIR_MODEL_NAME_TO_MODEL.get(model)()
    elif model_type == MODEL_TYPE.MPS_TEST_MODEL:
        m, m_inputs = MPS_MODEL_NAME_TO_MODEL.get(model)()

    lowering_func(m, m_inputs, model)


class TestMPSBackendExirModels(TestMPS):
    def test_model_with_unused_arg(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.EXIR_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_mlp(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.EXIR_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_mul_2(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.EXIR_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_element_wise_add(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.EXIR_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_emformer(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.EXIR_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )


class TestMPSBackendMPSModels(TestMPS):
    def test_conv2D(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_norm(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_module_add(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_toy_model_for_mem_planning(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_mem_planning_with_scratch_tensor(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_module_ops_return_tensor_list(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_module_contiguous_tensor(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )

    def test_module_input_dynamic_shape(self):
        run_model(
            inspect.stack()[0].function[5:],
            MODEL_TYPE.MPS_TEST_MODEL,
            lowering_func=self.lower_and_test_with_partitioner,
        )


class TestMPSUnitOpTesting(TestMPS):
    def test_mps_backend_split_copy(self):
        class SplitCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2, 1)

        example_inputs = (torch.randn(3, 5, 4, 7),)
        self.lower_and_test_with_partitioner(
            SplitCopy(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_unbind_copy(self):
        class UnbindCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.unbind(x, 1)

        example_inputs = (torch.randn(3, 5, 4, 7),)
        self.lower_and_test_with_partitioner(
            UnbindCopy(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_pixel_shuffle(self):
        class PixelShuffle(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.pixel_shuffle(x, 2)

        example_inputs = (torch.randn(3, 8, 4, 7),)
        self.lower_and_test_with_partitioner(
            PixelShuffle(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_cumsum(self):
        class CumulativeSum(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *x):
                return torch.cumsum(x[0], dim=0)

        example_inputs = (torch.randn(3, 5, 4, 7),)
        self.lower_and_test_with_partitioner(
            CumulativeSum(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_stack(self):
        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *x):
                return torch.stack((x), 0)

        example_inputs = (
            torch.randn(1, 5, 1, 8),
            torch.randn(1, 5, 1, 8),
        )
        self.lower_and_test_with_partitioner(
            Stack(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_cat(self):
        class Cat(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *x):
                return torch.cat((x), 1)

        example_inputs = (
            torch.randn(1, 5, 1, 8),
            torch.randn(1, 5, 1, 8),
        )
        self.lower_and_test_with_partitioner(
            Cat(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_expand_copy(self):
        class ExpandCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.example_inputs = [7, 5, 4, 8]

            def forward(self, x):
                return x.expand(self.example_inputs)

        example_inputs = (torch.randn(1, 5, 1, 8),)
        self.lower_and_test_with_partitioner(
            ExpandCopy(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_select(self):
        class Select(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.select(x, 3, 2)

        example_inputs = (torch.randn(3, 5, 4, 7),)
        self.lower_and_test_with_partitioner(
            Select(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_view_copy(self):
        class ViewCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.example_inputs = [2, 10, 2, 4]

            def forward(self, x):
                return x.view(self.example_inputs)

        example_inputs = (torch.randn(1, 5, 4, 8),)
        self.lower_and_test_with_partitioner(
            ViewCopy(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_mean_dim_2(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (-1, -2), keepdim=True)

        example_inputs = (torch.randn(1, 5, 4, 4),)
        self.lower_and_test_with_partitioner(
            Mean(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_squeeze_dim_1(self):
        class Squeeze(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.squeeze(x, 2)
                return torch.squeeze(y, 0)

        example_inputs = (torch.randn(1, 5, 1, 1, 4),)
        self.lower_and_test_with_partitioner(
            Squeeze(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_unsqueeze_dim_1(self):
        class Squeeze(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.unsqueeze(x, 1)

        example_inputs = (torch.randn(1, 5, 1, 4),)
        self.lower_and_test_with_partitioner(
            Squeeze(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_mean_dim_no_keepdim(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (-1, -2), keepdim=False)

        example_inputs = (torch.randn(1, 5, 4, 4),)
        self.lower_and_test_with_partitioner(
            Mean(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_mean_dim_unsupported(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (3), keepdim=True)

        example_inputs = (torch.randn(1, 5, 4, 4),)
        self.lower_and_test_with_partitioner(
            Mean(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_static_transpose(self):
        class PermuteModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nchw_to_nhwc = [0, 2, 3, 1]

            def forward(self, x):
                return torch.permute(x, self.nchw_to_nhwc)

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.lower_module_and_test_output(
            PermuteModule(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sequential_conv2d(self):
        class TwoConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )
                self.second = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x):
                return self.second(self.first(x))

        example_inputs = (torch.randn(1, 1, 3, 3),)
        self.lower_and_test_with_partitioner(
            TwoConv(), example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_conv2d_bn_1(self):
        class ModelConvBN(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)
                self.bn = randomize_bn(out_features)

            def forward(self, x):
                y = self.conv2d(x)
                y = self.bn(y)
                return y

        model = ModelConvBN(2, 2, (2, 2)).eval()

        self.lower_and_test_with_partitioner(
            model, (torch.randn(2, 2, 4, 4),), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_conv2d(self):
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_conv1d(self):
        example_inputs = (torch.randn(1, 57, 40),)
        stride = random.randint(1, 4)
        padding = random.randint(1, 4)
        conv = torch.nn.Conv1d(
            57,
            20,
            stride=stride,
            padding=padding,
            kernel_size=3,
            bias=random.choice([True, False]),
        )
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_conv2d_simple(self):
        N = 10
        C = 10
        H = 4
        W = 6
        groups = 2
        input_memory_format = torch.contiguous_format
        weight_memory_format = torch.contiguous_format
        strideX = random.randint(1, 4)
        strideY = random.randint(1, 4)
        example_inputs = (
            torch.randn(N, C, H, W).to(memory_format=input_memory_format),
        )
        conv = torch.nn.Conv2d(
            in_channels=N,
            out_channels=C,
            kernel_size=H,
            groups=groups,
            stride=(strideX, strideY),
            bias=False,
        )
        conv.weight.data = conv.weight.to(memory_format=weight_memory_format)
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_conv2d_to_depthwise_conv_3d(self):
        N = 10
        C = 10
        H = 4
        W = 6
        groups = 10
        input_memory_format = torch.contiguous_format
        weight_memory_format = torch.contiguous_format
        strideX = random.randint(1, 4)
        strideY = random.randint(1, 4)
        example_inputs = (
            torch.randn(N, C, H, W).to(memory_format=input_memory_format),
        )
        conv = torch.nn.Conv2d(
            in_channels=N,
            out_channels=C,
            kernel_size=H,
            groups=groups,
            stride=(strideX, strideY),
        )
        conv.weight.data = conv.weight.to(memory_format=weight_memory_format)
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_conv2d_single_int_params(self):
        groups = 1
        stride = 2
        padding = "valid"
        dilation = 1
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_conv2d_dw(self):
        # Depthwise Convolution Requirements:
        # - Groups must equal In Channels
        # - Out Channels must be a positive multiple of In Channels
        groups = 2
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = groups
        out_channels = 3 * in_channels
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(
            conv, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_mm(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=False).eval()
            example_input = (torch.randn(in_size, input_size),)

            self.lower_and_test_with_partitioner(
                linear, example_input, func_name=inspect.stack()[0].function[5:]
            )

    def test_mps_backend_bmm(self):
        class BmmModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.bmm = torch.bmm

            def forward(self, x, y):
                return self.bmm(x, y)

        mul_module = BmmModule()
        model_inputs = (
            torch.randn((3, 1, 8)),
            torch.randn((3, 8, 1)),
        )

        self.lower_and_test_with_partitioner(
            mul_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_addmm(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=True).eval()
            example_input = (torch.randn(in_size, input_size),)

            self.lower_and_test_with_partitioner(
                linear, example_input, func_name=inspect.stack()[0].function[5:]
            )

    def test_mps_backend_full_ones_default(self):
        class Ones(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                size = (4, 37, 17)
                return torch.ones(size)

        self.lower_and_test_with_partitioner(
            Ones(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_full_zeros_default(self):
        class Zeros(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                size = (4, 37, 17)
                return torch.zeros(size=size)

        self.lower_and_test_with_partitioner(
            Zeros(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_full_default(self):
        class Full(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                size = (4, 37, 17)
                return torch.full(size=size, fill_value=2.0)

        self.lower_and_test_with_partitioner(
            Full(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_full_like(self):
        class Full_Like(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.full_like(x, fill_value=2.0)

        const_module = Full_Like()
        model_inputs = (torch.randn(4, 37, 17),)

        self.lower_and_test_with_partitioner(
            const_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_logit_1(self):
        class LogitModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = torch.ops.aten.logit.default(x)
                return z

        logit_module = LogitModule()
        model_inputs = (torch.rand(5),)

        self.lower_and_test_with_partitioner(
            logit_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_logit_2(self):
        class LogitModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = torch.ops.aten.logit.default(x, eps=1e-6)
                return z

        logit_module = LogitModule()
        model_inputs = (torch.rand(5),)

        self.lower_and_test_with_partitioner(
            logit_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_round(self):
        class RoundModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.round(x)
                return out

        module = RoundModule()
        model_inputs = (torch.randn(5, 2),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_amax(self):
        class AmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.amax(x, 1)
                return out

        module = AmaxModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_amin(self):
        class AminModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.amin(x, 1)
                return out

        module = AminModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    @unittest.skip
    def test_mps_backend_min_dim(self):
        class MinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.min(x, 1)
                return out

        module = MinModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_argmax_1(self):
        class ArgmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out1 = torch.argmax(x, 1)
                return out1

        module = ArgmaxModule()
        model_inputs = (torch.randn(5, 10),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_argmax_2(self):
        class ArgmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out1 = torch.argmax(x)
                return out1

        module = ArgmaxModule()
        model_inputs = (torch.randn(5, 10),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_argmin_1(self):
        class ArgminModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out1 = torch.argmin(x, 1)
                return out1

        module = ArgminModule()
        model_inputs = (torch.randn(5, 10),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_argmin_2(self):
        class ArgminModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out1 = torch.argmin(x)
                return out1

        module = ArgminModule()
        model_inputs = (torch.randn(5, 10),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_minimum(self):
        class MinimumModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.minimum_module = torch.minimum

            def forward(self, x, y):
                return self.minimum_module(x, y)

        module = MinimumModule()
        model_inputs = (
            torch.randn(1, 3, 6),
            torch.randn(1, 3, 6),
        )
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_eq_tensor_1(self):
        class EqModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.eq(x, y)
                return out

        module = EqModule()
        model_inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_eq_tensor_2(self):
        class EqModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.eq(x, y)
                return out

        module = EqModule()
        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_eq_scalar(self):
        class EqModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.eq(x, 1.0)
                return out

        module = EqModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ne_tensor_1(self):
        class NeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.ne(x, y)
                return out

        module = NeModule()
        model_inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ne_tensor_2(self):
        class NeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.ne(x, y)
                return out

        module = NeModule()
        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ne_scalar(self):
        class NeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.ne(x, 1.0)
                return out

        module = NeModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ge_tensor_1(self):
        class GeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.ge(x, y)
                return out

        module = GeModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ge_tensor_2(self):
        class GeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.ge(x, y)
                return out

        module = GeModule()

        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ge_scalar(self):
        class GeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.ge(x, 1.0)
                return out

        module = GeModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_gt_tensor_1(self):
        class GtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.gt(x, y)
                return out

        module = GtModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_gt_tensor_2(self):
        class GtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.gt(x, y)
                return out

        module = GtModule()
        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_gt_scalar(self):
        class GtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.gt(x, 1.0)
                return out

        module = GtModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_isnan(self):
        class IsNanModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.isnan(x)

        module = IsNanModule()
        model_inputs = (
            torch.randn(8, 3, 4, 5).index_put_(
                indices=[torch.tensor([random.randrange(0, 8)])],
                values=torch.tensor(float("nan")),
            ),
        )
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_partitioner(self):
        # `index.Tensor`` is not yet natively supported
        # It will fall back to MPSPartitioner. Once implemented,
        # replace the op with an unsupported one.
        class IndexTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.indices = torch.tensor([0, 5, 2, 3])

            def forward(self, x):
                y = torch.add(x, 2.0)
                z = y[self.indices]
                r = z + x[self.indices]
                d = r - 2
                p = torch.pow(d, 4)
                return p / 10

        module = IndexTensorModule()

        model_inputs = (torch.randn(8, 3, 4, 5),)
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_1(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 1, 2], [0, 1, 0]]

        module = IndexGet()
        model_inputs = (torch.tensor([[1, 2], [3, 4], [5, 6]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_2(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [0, 4, 2]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_3(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [[0, 1], [4, 3]]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_4(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 4, 2]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_5(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 2, 1], :, 0]

        module = IndexGet()
        model_inputs = (torch.ones(3, 2, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indices2d(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, rows, columns):
                return x[rows, columns]

        module = IndexGet()
        x = torch.arange(0, 12).resize(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]])
        columns = torch.tensor([[0, 2], [0, 2]])
        model_inputs = (
            x,
            rows,
            columns,
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_slicing_using_advanced_index_for_column_0(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:4]

        module = IndexGet()
        model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_slicing_using_advanced_index_for_column_1(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # using advanced index for column
                return x[1:4, [1, 2]]

        module = IndexGet()
        model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    @unittest.skip
    def test_boolean_array_indexing(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[x > 5]

        module = IndexGet()
        model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_isinf(self):
        class IsInfModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.isinf(x)

        module = IsInfModule()
        model_inputs = (
            torch.randn(8, 3, 4, 5).index_put_(
                indices=[torch.tensor([random.randrange(0, 8)])],
                values=torch.tensor(float("inf")),
            ),
        )
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_le_tensor_1(self):
        class LeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.le(x, y)
                return out

        module = LeModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_le_tensor_2(self):
        class LeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.le(x, y)
                return out

        module = LeModule()
        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_le_scalar(self):
        class LeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.le(x, 1.0)
                return out

        module = LeModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_lt_tensor_1(self):
        class LtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.lt(x, y)
                return out

        module = LtModule()
        model_inputs = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_lt_tensor_2(self):
        class LtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                out = torch.le(x, y)
                return out

        module = LtModule()
        input_tensor = torch.randn(2, 3, 4)
        model_inputs = (input_tensor, input_tensor)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_lt_scalar(self):
        class LtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.lt(x, 1.0)
                return out

        module = LtModule()
        model_inputs = (torch.randn(2, 3, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    @torch.inference_mode()  # TODO Use  for capturing.
    def test_mps_backend_linear(self):
        in_size = 2
        input_size = 3
        output_size = 4
        linear = torch.nn.Linear(input_size, output_size).eval()
        example_input = (torch.randn(in_size, input_size),)

        self.lower_and_test_with_partitioner(
            linear, example_input, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_glu(self):
        class GLUModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.glu = torch.nn.GLU(dim=dim)

            def forward(self, x):
                return self.glu(x)

        shape = (4, 2)
        for dim in list(range(len(shape))) + [-1]:
            model_inputs = (torch.rand(shape),)
            glu_module = GLUModule(dim)
            self.lower_and_test_with_partitioner(
                glu_module, model_inputs, func_name=inspect.stack()[0].function[5:]
            )

    def test_mps_backend_softmax(self):
        class SoftMaxModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.softmax = torch.nn.Softmax(dim=dim)

            def forward(self, x):
                return self.softmax(x)

        shape = (3, 5, 7)
        for dim in list(range(len(shape))) + [-1]:
            model_inputs = (torch.rand(shape),)
            softmax_module = SoftMaxModule(dim)
            self.lower_and_test_with_partitioner(
                softmax_module, model_inputs, func_name=inspect.stack()[0].function[5:]
            )

    def test_mps_backend_log_softmax(self):
        class LogSoftMaxModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.logsoftmax = torch.nn.LogSoftmax(dim=dim)

            def forward(self, x):
                return self.logsoftmax(x)

        shape = (3, 5, 7)
        for dim in list(range(len(shape))) + [-1]:
            model_inputs = (torch.rand(shape),)
            logsoftmax_module = LogSoftMaxModule(dim)

            self.lower_and_test_with_partitioner(
                logsoftmax_module,
                model_inputs,
                func_name=inspect.stack()[0].function[5:],
            )

    def test_mps_backend_hardtanh(self):
        class HardTanhModule(torch.nn.Module):
            def __init__(self, min_val=-1.0, max_val=1.0):
                super().__init__()
                self.hardtanh = torch.nn.Hardtanh(min_val, max_val)

            def forward(self, x):
                return self.hardtanh(x)

        inputs = [torch.randn(2, 3, 4), torch.randn(7, 5, 2), torch.randn(2, 9)]
        for test_input in inputs:
            hardtanh_model = HardTanhModule()
            self.lower_and_test_with_partitioner(
                hardtanh_model, (test_input,), func_name=inspect.stack()[0].function[5:]
            )

        for test_input in inputs:
            hardtanh_model = HardTanhModule(-2, 2)
            self.lower_and_test_with_partitioner(
                hardtanh_model, (test_input,), func_name=inspect.stack()[0].function[5:]
            )

    def test_mps_backend_Relu(self):
        class ReluModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        example_input = torch.randn(2, 3, 4)
        self.lower_and_test_with_partitioner(
            ReluModule(), (example_input,), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_GELU(self):
        class GELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU()
                self.gelu_tanh = torch.nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.gelu(x)
                # MPS TODO: MPS Gelu tanh fails
                # return self.gelu_tanh(y)

        example_input = torch.randn(2, 3, 4)
        self.lower_and_test_with_partitioner(
            GELUModule(), (example_input,), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_leaky_Relu(self):
        class LeakyReluModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaky_relu = torch.nn.LeakyReLU()
                self.leaky_relu_2 = torch.nn.LeakyReLU(1.0)

            def forward(self, x):
                out = self.leaky_relu(x)
                out = self.leaky_relu_2(out)
                return out

        example_input = torch.randn(2, 3, 4)
        self.lower_and_test_with_partitioner(
            LeakyReluModule(),
            (example_input,),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_sigmoid(self):
        class SigmoidModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        model_inputs = (torch.rand(7, 5, 3),)
        sigmoid_module = SigmoidModule()
        self.lower_and_test_with_partitioner(
            sigmoid_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_constant_pad_nd(self):
        class PadModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant_pad = torch.nn.ConstantPad2d((1, 2), 0)

            def forward(self, x):
                return self.constant_pad(x)

        model_inputs = (torch.rand(1, 2, 3, 4),)
        pad_module = PadModule()
        self.lower_and_test_with_partitioner(
            pad_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_index_select(self):
        class IndexSelectModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.index_select(input, dim=2, index=index)

        model_inputs = (torch.rand(2, 8, 4, 5), torch.tensor([3, 0, 1]))
        module = IndexSelectModule()
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_empty(self):
        class EmptyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                return torch.empty((3, 4, 5), dtype=torch.float32)

        self.lower_and_test_with_partitioner(
            EmptyModule(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_static_constant_pad(self):
        class StaticConstantPadModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                pad_6 = (1, 2, 3, 4, 5, 6)
                pad_4 = (1, 2, 3, 4)
                pad_2 = (1, 2)
                a = torch.nn.functional.pad(
                    input=x,
                    pad=pad_6,
                    mode="constant",
                    value=2.3,
                )
                b = torch.nn.functional.pad(
                    input=x,
                    pad=pad_4,
                    mode="constant",
                    value=1.3,
                )
                c = torch.nn.functional.pad(
                    input=x,
                    pad=pad_2,
                    mode="constant",
                    value=2.1,
                )
                d = torch.nn.functional.pad(
                    input=y,
                    pad=pad_6,
                    mode="constant",
                    value=2.7,
                )
                e = torch.nn.functional.pad(
                    input=y,
                    pad=pad_4,
                    mode="constant",
                    value=1.9,
                )
                f = torch.nn.functional.pad(
                    input=y,
                    pad=pad_2,
                    mode="constant",
                    value=3.1,
                )
                g = torch.nn.functional.pad(
                    input=z,
                    pad=pad_4,
                    mode="constant",
                    value=2.9,
                )
                h = torch.nn.functional.pad(
                    input=z,
                    pad=pad_2,
                    mode="constant",
                    value=1.2,
                )
                return (a, b, c, d, e, f, g, h)

        example_inputs = (
            torch.randn(size=(5, 4, 3, 2)),
            torch.randn(size=(5, 3, 2)),
            torch.randn(size=(4, 3)),
        )
        self.lower_and_test_with_partitioner(
            StaticConstantPadModule(),
            example_inputs,
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_clamp_min_max(self):
        class Clamp(torch.nn.Module):
            def __init__(self, min_val, max_val):
                super().__init__()
                self.clamp = torch.clamp
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, *x):
                out1 = self.clamp(x[0], min=-0.5, max=0.5)
                out2 = self.clamp(x[0], min=-5, max=5)
                return out1, out2

        model_inputs = (
            torch.randn(1, 4, 122, 122) * 2,
            torch.randint(-100, 100, (1, 4, 15, 20)),
        )
        module = Clamp(-0.5, 0.5)
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_clamp_min(self):
        class Clamp(torch.nn.Module):
            def __init__(self, min_val, max_val):
                super().__init__()
                self.clamp = torch.clamp
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, x):
                return self.clamp(x, min=self.min_val, max=self.max_val)

        model_inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = Clamp(-0.5, None)
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_clamp_max(self):
        class Clamp(torch.nn.Module):
            def __init__(self, min_val, max_val):
                super().__init__()
                self.clamp = torch.clamp
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, x):
                return self.clamp(x, min=self.min_val, max=self.max_val)

        model_inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = Clamp(None, 0.5)
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_maxpool2d_default(self):
        class MaxPool2dModule(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)

        maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
        model_inputs = (torch.randn(4, 3, 24, 24),)

        self.lower_and_test_with_partitioner(
            maxpool2d_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_maxpool2d_unsupported(self):
        class MaxPool2dModule(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    return_indices=True,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)[1]

        maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
        model_inputs = (torch.randn(4, 3, 24, 24),)

        self.lower_and_test_with_partitioner(
            maxpool2d_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_max_dim_vals(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                max_vals, _ = torch.max(x, dim=3, keepdim=True)
                return max_vals

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(
            max_dim_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_max_dim(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                x = torch.add(x, x)
                max_values_1, max_indices_1 = torch.max(x, dim=2, keepdim=True)
                max_values_2, max_indices_2 = torch.max(x, dim=3, keepdim=True)
                return (max_values_1, max_indices_1, max_values_2, max_indices_2)

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(
            max_dim_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_multiply(self):
        class MulModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.mul = torch.mul

            def forward(self, x, y):
                return self.mul(x, y)

        mul_module = MulModule()
        model_inputs = (
            torch.randn((1, 8)),
            torch.randn((8, 1)),
        )

        self.lower_and_test_with_partitioner(
            mul_module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sub(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = torch.sub

            def forward(self, x, y):
                return self.sub(x, y)

        module = Sub()
        M = torch.randn(2, 3)
        N = torch.randn(2, 3)
        model_inputs = (
            M,
            N,
        )
        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_clone(self):
        class Clone(torch.nn.Module):
            def forward(self, x):
                return torch.clone(x)

        model_inputs = (torch.randn(1, 3, 3),)
        self.lower_and_test_with_partitioner(
            Clone(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_floor(self):
        class Floor(torch.nn.Module):
            def forward(self, x):
                return torch.floor(x)

        model_inputs = (torch.randn(1, 3, 3),)
        self.lower_and_test_with_partitioner(
            Floor(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sqrt(self):
        class Sqrt(torch.nn.Module):
            def forward(self, x):
                return torch.sqrt(x)

        model_inputs = (torch.randn(1, 3, 3).abs(),)
        self.lower_and_test_with_partitioner(
            Sqrt(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_ceil(self):
        class Ceil(torch.nn.Module):
            def forward(self, x):
                return torch.ceil(x)

        model_inputs = (torch.randn(1, 3, 3),)
        self.lower_and_test_with_partitioner(
            Ceil(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_hardswish(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class HardswishModule(torch.nn.Module):
            def __init__(self):
                super(HardswishModule, self).__init__()
                self.hardswish_out_of_place = torch.nn.Hardswish()
                self.hardswish_in_place = torch.nn.Hardswish(inplace=True)
                self.hardswish_functional = torch.nn.functional.hardswish

            def forward(self, x):
                a = self.hardswish_out_of_place(x)
                a = self.hardswish_in_place(a)
                a = self.hardswish_functional(a)
                return a

        # TODO(T158969708)
        self.lower_and_test_with_partitioner(
            HardswishModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_leaky_relu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super(LeakyReLUModule, self).__init__()
                self.leaky_relu_out_of_place = torch.nn.LeakyReLU(negative_slope=0.2)
                self.leaky_relu_in_place = torch.nn.LeakyReLU(
                    negative_slope=0.08, inplace=True
                )
                self.leaky_relu_functional_default = torch.nn.functional.leaky_relu

            def forward(self, x):
                a = self.leaky_relu_out_of_place(x)
                a = self.leaky_relu_in_place(a)
                a = self.leaky_relu_functional_default(a)
                return a

        self.lower_and_test_with_partitioner(
            LeakyReLUModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    @unittest.skip
    def test_mps_channels_last_tagged_reshape_pass_output(self):
        op_sequences = OpSequencesAddConv2d(2, 2)
        op_sequences.eval()

        example_inputs = (torch.ones(1, 1, 6, 6),)

        self.lower_and_test_with_partitioner(
            op_sequences, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_conv2d_bn_hardtanh_mean_sequence(self):
        """
        This test makes sure that we can fuse batchnorm and hardtanh
        even with inserting copy nodes at some spots in the graph to change
        memory format
        """
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
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
                self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)

            def forward(self, x):
                x = self.conv(x)
                x = self.native_batchnorm(x)
                x = self.hardtanh(x)
                x = torch.mean(x, (-1, -2), keepdim=True)
                return x

        test_module = TestModule()
        test_module.eval()
        self.lower_and_test_with_partitioner(
            test_module, example_inputs, func_name=inspect.stack()[0].function[5:]
        )

    @unittest.expectedFailure
    def test_mps_backend_maximum_no_broadcast(self):
        model_inputs_no_broadcast = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))

        self.lower_and_test_with_partitioner(
            torch.maximum,
            model_inputs_no_broadcast,
            func_name=inspect.stack()[0].function[5:],
        )

    @unittest.expectedFailure
    def test_mps_backend_maximum_broadcast(self):
        model_inputs_broadcast = (torch.randn(2, 3, 4), torch.randn(2, 1, 4))

        self.lower_and_test_with_partitioner(
            torch.maximum,
            model_inputs_broadcast,
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_negative(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class NegModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.neg(x)

        self.lower_and_test_with_partitioner(
            NegModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_remainder_1(self):
        model_inputs = (torch.randn(1, 3, 3), torch.randn(1, 3, 3))

        class RemainderModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.remainder(x, y)

        self.lower_and_test_with_partitioner(
            RemainderModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_remainder_2(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class RemainderModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.remainder(x, 0.5)

        self.lower_and_test_with_partitioner(
            RemainderModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_square(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class SquareModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.square(x)

        self.lower_and_test_with_partitioner(
            SquareModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_pow_1(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class PowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.pow(x, 4)

        self.lower_and_test_with_partitioner(
            PowModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_pow_2(self):
        model_inputs = (torch.randn(1, 3, 3), torch.tensor(4))

        class PowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.pow(x, y)

        self.lower_and_test_with_partitioner(
            PowModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_elu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.square(x)

        self.lower_and_test_with_partitioner(
            ELUModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_avg_pool_2d_1(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=False,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(
            AvgPoolModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_avg_pool_2d_2(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=True,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(
            AvgPoolModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_avg_pool_2d_3(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=False,
                    ceil_mode=True,
                    divisor_override=4,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(
            AvgPoolModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_abs(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class AbsModule(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        self.lower_and_test_with_partitioner(
            AbsModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_sign(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class SignModule(torch.nn.Module):
            def forward(self, x):
                return torch.sign(x)

        self.lower_and_test_with_partitioner(
            SignModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_rsqrt(self):
        model_inputs = (torch.randn(1, 3, 3).abs(),)

        class RsqrtModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        self.lower_and_test_with_partitioner(
            RsqrtModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_prelu(self):
        num_channels = 5
        model_inputs = (torch.randn(1, num_channels, 3, 2),)

        class PReLUModule(torch.nn.Module):
            def __init__(self):
                super(PReLUModule, self).__init__()
                self.prelu = torch.nn.PReLU()
                self.prelu_non_default = torch.nn.PReLU(
                    num_parameters=num_channels, init=0.2
                )

            def forward(self, x):
                a = self.prelu(x)
                a = self.prelu_non_default(a)
                return a

        self.lower_and_test_with_partitioner(
            PReLUModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

        # Should fail to be partitioned since constraint (input dim) is violated
        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            torch.nn.PReLU(),
            (torch.randn(1, 2),),
        )

    def test_mps_backend_concatenate2(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat((y, x), 0)

        self.lower_and_test_with_partitioner(
            Concat(),
            (torch.ones(4, 2, 3), torch.randn(1, 2, 3)),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_concatenate3(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.concat((y, y, x), 0)

        self.lower_and_test_with_partitioner(
            Concat(),
            (torch.ones(4, 2, 3), torch.randn(1, 2, 3)),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_concatenate4(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.concatenate((y, x, y, x), 2)

        self.lower_and_test_with_partitioner(
            Concat(),
            (torch.randn(1, 2, 3), torch.randn(1, 2, 5)),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_concatenate_nhwc(self):
        class Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x, y):
                x = self.conv(x)
                return torch.concatenate((y, x, y, x), 1)

        self.lower_and_test_with_partitioner(
            Concat(),
            (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3)),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_concatenate_nhwc2(self):
        class Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x, y):
                x = self.conv(x)
                y = self.conv(y)
                return torch.concatenate((y, x, y, x), 3)

        self.lower_and_test_with_partitioner(
            Concat(),
            (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3)),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_slice_copy(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[1:3, -2:, :-1]

        self.lower_and_test_with_partitioner(
            Slice(), (torch.randn(5, 5, 5),), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_slice_copy_stride_non_1(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[:3:-1, 2:, :3]

        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            Slice(),
            (torch.randn(5, 5, 5),),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_slice_copy_dim_0(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[-1:3, 2:, 3:3]

        self.lower_module_and_test_output(
            Slice(),
            (torch.randn(5, 5, 5),),
            use_partitioner=False,
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_slice_copy_memory_format(self):
        class ConvSlice(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x):
                y = self.conv(x)
                return y[:, :, 2:3, -2:]

        self.lower_and_test_with_partitioner(
            ConvSlice(),
            (torch.randn(1, 1, 3, 3),),
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_bitwise_and(self):
        class BitwiseAnd(torch.nn.Module):
            def forward(self, x, y):
                return torch.bitwise_and(x, y)

        model_inputs = (
            torch.tensor([-1, -2, 3], dtype=torch.int8),
            torch.tensor([1, 0, 3], dtype=torch.int8),
        )
        self.lower_and_test_with_partitioner(
            BitwiseAnd(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_bitwise_or(self):
        class BitwiseOr(torch.nn.Module):
            def forward(self, x, y):
                return torch.bitwise_or(x, y)

        model_inputs = (
            torch.tensor([-1, -2, 3], dtype=torch.int8),
            torch.tensor([1, 0, 3], dtype=torch.int8),
        )
        self.lower_and_test_with_partitioner(
            BitwiseOr(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_bitwise_xor(self):
        class BitwiseXor(torch.nn.Module):
            def forward(self, x, y):
                return torch.bitwise_xor(x, y)

        model_inputs = (
            torch.tensor([True, True, False]),
            torch.tensor([False, True, False]),
        )
        self.lower_and_test_with_partitioner(
            BitwiseXor(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_bitwise_not(self):
        class BitwiseNot(torch.nn.Module):
            def forward(self, x):
                return torch.bitwise_not(x)

        model_inputs = (torch.tensor([-1, -2, 3], dtype=torch.int8),)
        self.lower_and_test_with_partitioner(
            BitwiseNot(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_bitwise_not_with_bool(self):
        class BitwiseNot(torch.nn.Module):
            def forward(self, x):
                return torch.bitwise_not(x)

        model_inputs = (torch.tensor([True, True, False]),)
        self.lower_and_test_with_partitioner(
            BitwiseNot(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_bitwise_with_scalar(self):
        class BitwiseScalarModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar = 3

            def forward(self, x):
                out1 = torch.ops.aten.bitwise_and.Scalar(x, self._scalar)
                return out1

        model_inputs = (torch.tensor([-1, -2, 3], dtype=torch.int8),)
        self.lower_and_test_with_partitioner(
            BitwiseScalarModule(),
            model_inputs,
            func_name=inspect.stack()[0].function[5:],
        )

    def test_mps_backend_arange(self):
        class ArangeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._begin = 2.5
                self._end = 5
                self._step = 0.5

            def forward(self):
                out1 = torch.arange(end=self._end)
                out2 = torch.arange(start=self._begin, end=self._end, step=self._step)
                return out1 + out2

        self.lower_and_test_with_partitioner(
            ArangeModule(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_where(self):
        class Where(torch.nn.Module):
            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        x = torch.randn(3, 2)
        y = torch.ones(3, 2)
        cond = x > 0
        module_inputs = (cond, x, y)
        self.lower_and_test_with_partitioner(
            Where(), module_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_scalar_tensor(self):
        class ScalarTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._scalar = 3.0
                self._bool = True

            def forward(self):
                out1 = torch.ops.aten.scalar_tensor(self._scalar)
                out2 = torch.ops.aten.scalar_tensor(self._scalar, dtype=torch.int32)
                # issue 121117206
                out3 = torch.ops.aten.scalar_tensor(self._bool, dtype=torch.bool)
                return out1 + out2 + out3

        self.lower_and_test_with_partitioner(
            ScalarTensorModule(), (), func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_tril(self):
        class TrilModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._k = 1
                self._negK = -1

            def forward(self, x):
                out1 = torch.tril(x, diagonal=self._k)
                out2 = torch.tril(x, diagonal=self._negK)
                return out1 + out2

        model_inputs = (torch.randn(4, 6),)
        self.lower_and_test_with_partitioner(
            TrilModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_backend_embedding(self):
        class EmbeddingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._embedding = torch.nn.Embedding(10, 3)
                self._embedding_with_padding = torch.nn.Embedding(10, 3, padding_idx=2)

            def forward(self, x):
                return self._embedding(x) + self._embedding_with_padding(x)

        model_inputs = (torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]),)
        self.lower_and_test_with_partitioner(
            EmbeddingModule(), model_inputs, func_name=inspect.stack()[0].function[5:]
        )
