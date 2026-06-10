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
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


class MaxPool1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
        )

    def forward(self, x):
        return self.max_pool(x)


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size: int | tuple[int, ...] = 3, **kwargs):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, **kwargs)

    def forward(self, x):
        return self.max_pool2d(x)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestMaxPool2D:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shape, graph_verifier)

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `max_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [MaxPool2DWithIndices])

    def test__basic_nsys_inference(self, mocker):
        input_shape = (2, 4, 6, 7)  # The old flow limited the batch size to 1.
        model = MaxPool2dModule()
        self.assert_delegated(model, input_shape, mocker)

    def test__basic_nsys_inference_qat(self, mocker):
        input_shape = (2, 11, 7, 16)  # The old flow limited the batch size to 1.
        model = MaxPool2dModule()
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            use_qat=True,
        )

    def test__large_kernel_size(self, mocker):
        kernel_size = (1, 5000)
        input_shape = (1, 4) + kernel_size
        model = MaxPool2dModule(kernel_size, stride=1)
        self.assert_delegated(model, input_shape, mocker)

    def test__stride_limit__no_padding(self, mocker):
        stride = 4096
        input_shape = (1, 4, 1, 4096)
        model = MaxPool2dModule(1, stride=stride)
        self.assert_delegated(model, input_shape, mocker)

    def test__stride_limit_exceeded__no_padding(self):
        stride = 4097  # Exceeds the stride limit.
        input_shape = (1, 4, 1, 4096)
        model = MaxPool2dModule(1, stride=stride)
        self.assert_not_delegated(model, input_shape)

    def test__stride_limit__padding(self, mocker):
        padding = 1
        stride = 4096
        input_shape = (1, 2, 3, stride)
        model = MaxPool2dModule(3, stride=stride, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__stride_limit_exceeded__padding(self):
        padding = 1
        stride = 4097  # Exceeds the stride limit.
        input_shape = (1, 2, 3, stride)
        model = MaxPool2dModule(3, stride=stride, padding=padding)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.skip(
        reason="Large padding requires large kernel size which results in an extremely slow test."
    )
    def test__padding_limit(self, mocker):
        # As the padding is added wia a `Pad` operator (not the `MaxPool` arguments), there is no limit to the padded
        #  value. But as padding can be at most half of the kernel size (PyTorch requirement) and kernel size is limited
        #  to 4096, padding of 2048 is the limit.
        padding = 2048
        kernel_size = padding * 2
        input_shape = (1, 1, 2, 3)
        model = MaxPool2dModule(kernel_size, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__padding__max_pool_limit_exceeded(self, mocker):
        # NeutronIR `MaxPool` padding is limited to 32. But as it is added by the `Pad` operator instead, there is no
        #  limit. This tests ensures the `MaxPool` padding limit is not a problem.
        padding = 33
        kernel_size = padding * 2
        input_shape = (1, 2, 3, 4)
        model = MaxPool2dModule(kernel_size, padding=padding)
        self.assert_delegated(model, input_shape, mocker)

    def test__padding_to_kernel_ratio_exceeded(self):
        # Both PyTorch and Neutron require the padding to be at most half of the kernel size.
        kernel_size = 3
        padding = 2  # More than half of the kernel size.
        input_shape = (1, 2, 3, 4)
        model = MaxPool2dModule(kernel_size, padding=padding)
        with pytest.raises(
            RuntimeError, match="pad should be at most half of effective kernel size"
        ):
            to_quantized_edge_program(model, input_shape)


class TestMaxPool1D:

    # Just a basic test to verify that the operator gets extended to the 2D variant correctly.
    def test__basic_nsys_inference__view_not_delegated(self, mocker):
        input_shape = (2, 4, 6)  # The old flow limited the batch size to 1.
        model = MaxPool1DModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1, ViewCopy: 2},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shape, graph_verifier)
