# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import LeakyRelu
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class LeakyReluModule(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(*args, **kwargs)

    def forward(self, x):
        return self.leaky_relu(x)


class TestLeakyRelu:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker, request, use_qat=False):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={LeakyRelu: 1},
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            (2,),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
        ],
        ids=lambda shape: f"{len(shape)}D",
    )
    def test__default_alpha__input_shapes(self, mocker, request, input_shape):
        model = LeakyReluModule()
        self.assert_delegated(model, input_shape, mocker, request)

    def test__default_alpha__qat(self, mocker, request, use_qat):
        model = LeakyReluModule()
        input_shape = (23,)
        self.assert_delegated(model, input_shape, mocker, request, use_qat)

    @pytest.mark.parametrize(
        "alpha",
        [0.01, 3.14159, 0, 1, float("inf")],
        ids=lambda alpha: f"alpha = {alpha}",
    )
    def test__specific_alpha(self, mocker, request, alpha):
        model = LeakyReluModule(negative_slope=alpha)
        self.assert_delegated(model, (23,), mocker, request)

    def test__inplace(self, mocker, request):
        model = LeakyReluModule(inplace=True)
        self.assert_delegated(
            model,
            (23,),
            mocker,
            request,
        )
