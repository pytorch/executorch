# Copyright 2024,2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import AvgPool2dModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AvgPool2D,
    ExecutorchDelegateCall,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class AvgPool1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool1d(
            kernel_size=3,
        )

    def forward(self, x):
        return self.avg_pool(x)


class TestAvgPool2D:
    def test__basic_nsys_inference(self, mocker, request):
        input_shape = (2, 4, 6, 7)
        model = AvgPool2dModule(False, 0)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shape, graph_verifier, request)

    def test__basic_nsys_inference_qat(self, mocker, request):
        input_shape = (2, 9, 6, 15)
        model = AvgPool2dModule(False, 0)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            use_qat=True,
        )

    def test__kernel_size_limit(self, mocker, request):
        kernel_size = (1, 4096)
        input_shape = (1, 4) + kernel_size
        model = AvgPool2dModule(False, 0, kernel_size)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shape, graph_verifier, request)

    def test__kernel_size_limit_exceeded(self):
        kernel_size = (1, 4097)  # Exceeds the kernel size limit.
        input_shape = (1, 4) + kernel_size
        model = AvgPool2dModule(False, 0, kernel_size)

        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `avg_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AvgPool2D])

    def test__stride_limit(self, mocker, request):
        stride = 4096
        input_shape = (1, 4, 1, 4096)
        model = AvgPool2dModule(False, 0, 1, stride)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shape, graph_verifier, request)

    def test__stride_limit_exceeded(self):
        stride = 4097  # Exceeds the stride limit.
        input_shape = (1, 4, 1, 4096)
        model = AvgPool2dModule(False, 0, 1, stride)

        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `avg_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AvgPool2D])


class TestAvgPool1D:

    # Just a basic test to verify that the operator gets extended to the 2D variant correctly.
    def test__basic_nsys_inference(self, mocker, request):
        input_shape = (2, 4, 6)  # The old flow limited the batch size to 1.
        model = AvgPool1DModule()
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AvgPool2D: 1, ViewCopy: 2},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shape, graph_verifier, request)
