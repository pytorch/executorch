# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare

from executorch.backends.nxp.tests.ops_aliases import (
    AddMM,
    AddTensor,
    AvgPool2D,
    Convolution,
    ExecutorchDelegateCall,
    MM,
    PermuteCopy,
    Relu,
    ViewCopy,
)
from torch import nn
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ReshapeConvModule(nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape
        self.conv = nn.Conv2d(
            new_shape[1], new_shape[1], kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        x = self.conv(x)
        return x


class ConvViewConvModule(nn.Module):
    def __init__(self, input_shape: Sequence[int], new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape
        self.conv1 = nn.Conv2d(
            input_shape[1], input_shape[1], kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            new_shape[1], new_shape[1], kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.reshape(x, self.new_shape)
        x = self.conv2(x)
        return x


class AddReshapeModule(nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x + x
        x = torch.reshape(x, self.new_shape)
        return x


class ConvReshapeModule(nn.Module):
    def __init__(self, channels: int, new_shape: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, self.new_shape)
        return x


class LinearReshapeModule(torch.nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.linear = nn.Linear(64, 32, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, self.new_shape)
        return x


class ConvLinearViewModule(torch.nn.Module):
    def __init__(self, channels: int, channels_view_out: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2)
        self.linear = nn.Linear(channels_view_out, 32, bias=True)
        self.channels_view_out = channels_view_out
        self.avg_pool = nn.AvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(-1, self.channels_view_out)
        x = self.linear(x)
        return x


class ConvViewLinearModule(torch.nn.Module):
    def __init__(self, view_new_shape: Sequence[int], channels: int, bias: bool):
        super().__init__()
        self.view_new_shape = view_new_shape
        self.conv = nn.Conv2d(channels, channels, 1, 1)
        self.linear = nn.Linear(view_new_shape[1], 8, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(self.view_new_shape)
        x = self.linear(x)
        return x


class ViewViewModel(nn.Module):
    def __init__(self, new_shape_1: Sequence[int], new_shape_2: Sequence[int]):
        super().__init__()
        self.new_shape_1 = new_shape_1
        self.new_shape_2 = new_shape_2

    def forward(self, x):
        x = x.view(self.new_shape_1)
        return x.view(self.new_shape_2)


class ViewAddZeroModel(nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.view(self.new_shape)
        zero = torch.zeros(self.new_shape)
        return x + zero


class TestViewCopyNewFlow:
    @staticmethod
    def assert_delegated_and_correct(
        mocker,
        model,
        input_shape,
        request,
        exp_deleg_ops,
        exp_non_deleg_ops,
        use_qat=False,
    ):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=exp_deleg_ops,
            expected_non_delegated_ops=exp_non_deleg_ops,
        )

        dataset = RandomDatasetCreator(low=-128, high=128)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset,
            comparator,
            mocker=mocker,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    @staticmethod
    def assert_not_delegated(model, input_shape):
        delegated_ep = to_quantized_edge_program(
            model,
            input_shape,
        ).exported_program()

        # Make sure the partition was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [ViewCopy])

    @pytest.mark.parametrize(
        "input_shape, new_shape",
        [
            pytest.param((3, 7, 3, 2), (126,), id="1D view"),
            pytest.param((1, 4, 7, 9), (6, 42), id="2D view"),
            pytest.param((3, 3, 7, 7), (7, 7, 9), id="3D view"),
            pytest.param((1, 8, 6, 8), (6, 4, 2, 8), id="4D view"),
            pytest.param((2, 7, 5, 9), (3, 2, 3, 7, 5), id="5D view"),
        ],
    )
    def test__view_copy__channels_first_to_formatless(
        self,
        mocker,
        input_shape,
        new_shape,
        request,
        use_qat,
    ):
        model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={Convolution: 1, ViewCopy: 1},
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape, new_shape",
        [
            pytest.param((126,), (3, 7, 3, 2), id="1D view"),
            pytest.param((6, 42), (1, 4, 7, 9), id="2D view"),
            pytest.param((7, 7, 9), (3, 3, 7, 7), id="3D view"),
            pytest.param((6, 4, 2, 8), (1, 8, 6, 8), id="4D view"),
            pytest.param((3, 2, 3, 7, 5), (2, 7, 5, 9), id="5D view"),
        ],
    )
    def test__view_copy__formatless_to_channels_first(
        self, input_shape, new_shape, mocker, request, use_qat
    ):
        model = ReshapeConvModule(new_shape=new_shape)

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={Convolution: 1, ViewCopy: 1},
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape, new_shape",
        [
            pytest.param((3, 7, 3, 2), (126,), id="1D view"),
            pytest.param((1, 4, 7, 9), (6, 42), id="2D view"),
            pytest.param((3, 3, 7, 7), (7, 7, 9), id="3D view"),
            pytest.param((1, 8, 6, 8), (6, 4, 2, 8), id="4D view"),
            pytest.param((2, 7, 5, 9), (3, 2, 3, 7, 5), id="5D view"),
        ],
    )
    def test__view_copy__formatless_to_formatless(
        self, input_shape, new_shape, mocker, request, use_qat
    ):
        model = AddReshapeModule(new_shape=new_shape)

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={AddTensor: 1, ViewCopy: 1},
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape, new_shape",
        [
            pytest.param((6, 4, 2, 8), (1, 8, 6, 8), id="4D view"),
        ],
    )
    def test__view_copy__channels_first_to_channels_first(
        self, input_shape, new_shape, mocker, request, use_qat
    ):
        model = ConvViewConvModule(input_shape, new_shape)

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={Convolution: 2, ViewCopy: 1},
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    def test_view_copy_w_linear_quant_conversion(self, mocker, request, use_qat):
        input_shape = (8, 64)
        new_shape = (1, 16, 4, 4)

        model = LinearReshapeModule(new_shape=new_shape)

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={AddMM: 1, ViewCopy: 1, PermuteCopy: 1},
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    def test_view_w_conv_linear_quant_conversion(self, request, mocker, use_qat):
        input_shape = (1, 4, 16, 16)
        channels_view_out = 196

        model = ConvLinearViewModule(
            channels=input_shape[1], channels_view_out=channels_view_out
        )

        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={
                AddMM: 1,
                ViewCopy: 1,
                Convolution: 1,
                AvgPool2D: 1,
                Relu: 1,
                PermuteCopy: 1,
            },
            exp_non_deleg_ops={},
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "bias",
        [True, False],
    )
    def test__view_copy__context_dependent__channels_first_to_formatless__transpose_fused(
        self, bias, mocker, request
    ):
        input_shape = (1, 2, 3, 4)
        new_shape = (1, 2 * 3 * 4)
        model = ConvViewLinearModule(new_shape, 2, bias)

        converted_lin_op = AddMM if bias else MM
        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={
                converted_lin_op: 1,
                ViewCopy: 1,
                Convolution: 1,
                PermuteCopy: 1,
            },
            exp_non_deleg_ops={},
        )

    @pytest.mark.parametrize(
        "bias",
        [True, False],
    )
    def test__view_copy__context_dependent__channels_first_to_formatless__transpose_not_fusable(
        self, bias, mocker, request
    ):
        input_shape = (1, 2, 3, 4)
        new_shape = (
            2,
            3 * 4,
        )  # The batch size changes, which makes the optimization not applicable.
        model = ConvViewLinearModule(new_shape, 2, bias)

        converted_lin_op = AddMM if bias else MM
        self.assert_delegated_and_correct(
            mocker,
            model,
            input_shape,
            request,
            exp_deleg_ops={
                converted_lin_op: 1,
                ViewCopy: 1,
                Convolution: 1,
                PermuteCopy: 1,
            },
            exp_non_deleg_ops={},
        )

    def test__view_copy__noop_partitions__second_view(self):
        input_shape = (1, 2, 3, 4)
        new_shape1 = (2, 12)
        new_shape2 = (6, 4)
        model = ViewViewModel(new_shape1, new_shape2)

        self.assert_not_delegated(model, input_shape)

    def test__view_copy__noop_partitions__add_zeros(self):
        input_shape = (1, 2, 3, 4)
        new_shape = (2, 12)
        model = ViewAddZeroModel(new_shape)

        self.assert_not_delegated(model, input_shape)
