# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops

from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier

from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.models import SoftmaxModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import ExecutorchDelegateCall, Softmax


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ConvSoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 1)
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return self.softmax(x)


def assert_softmax_delegated(graph):
    assert graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(graph, [Softmax])


def assert_softmax_not_delegated(graph):
    assert not graph_contains_any_of_ops(graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(graph, [Softmax])


class TestSoftmax:
    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            # Dim must always be the last dimension.
            pytest.param((10,), -1, id="1D_dim_-1"),
            pytest.param((5, 21), -1, id="2D_dim_-1"),
            pytest.param((2, 3, 13), -1, id="3D_dim_-1"),
            pytest.param((1, 3, 3, 200), -1, id="4D_dim_-1"),
            pytest.param((5, 4, 3, 2, 180), -1, id="5D_dim_-1"),
        ],
    )
    def test__basic_nsys_inference(self, mocker, input_shape, dim):
        model = SoftmaxModule(dim)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Softmax: 1},
            expected_non_delegated_ops={},
        )
        output_comparator = NumericalStatsOutputComparator(
            max_mse_error=0.001, is_classification_task=True
        )
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            output_comparator=output_comparator,
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((4096, 8), -1, id="2D_spatial_size_limit"),
            pytest.param((2040,), -1, id="1D_channels_limit"),
            pytest.param((4096, 128), -1, id="2D_total_size_limit"),
            pytest.param((1, 64, 64, 8), -1, id="4D_spatial_size_limit_1x64x64"),
            pytest.param((2, 32, 64, 8), -1, id="4D_spatial_size_limit_2x32x64"),
        ],
    )
    def test__limits(self, input_shape, dim, mocker):
        model = SoftmaxModule(dim)
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `softmax` was delegated.
        assert_softmax_delegated(delegated_ep.graph)

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((4097, 8), -1, id="2D_spatial_size_exceeded"),
            pytest.param((2048,), -1, id="1D_channels_exceeded"),
            pytest.param((4096, 129), -1, id="2D_total_size_exceeded"),
            pytest.param((1, 64, 65, 8), -1, id="4D_spatial_size_exceeded_1x64x65"),
            pytest.param((2, 32, 65, 8), -1, id="4D_spatial_size_exceeded_2x32x65"),
        ],
    )
    def test__limits_exceeded(self, input_shape, dim):
        model = SoftmaxModule(dim)
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `softmax` was NOT delegated.

        assert_softmax_not_delegated(delegated_ep.graph)
