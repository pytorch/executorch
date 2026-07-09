# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import (
    LinearRampDatasetCreator,
    RandomDatasetCreator,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Convolution, Neg
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class NegModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return -x


class ConvNegModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x = self.conv(x)
        return -x


class TestNeg:
    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((8,), id="1D"),
            pytest.param((4, 2), id="2D"),
            pytest.param((1, 2, 6), id="3D"),
            pytest.param((1, 5, 3, 4), id="4D"),
        ],
    )
    def test__basic_nsys_inference(self, mocker, request, input_shape):
        model = NegModule()

        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Neg: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator=RandomDatasetCreator(low=-1.0, high=1.0),
        )

    def test__all_possible_values(self, mocker, request, use_qat):
        # Use 256 elements so that, after quantization to int8, the input can
        # cover the full discrete range [-128, 127].
        # The dataset is generated as a linear float ramp and later quantized,
        # which effectively exercises all int8 values.
        input_shape = (256,)
        model = NegModule()

        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Neg: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator=LinearRampDatasetCreator(low=-1.0, high=1.0),
            use_qat=use_qat,
        )

    def test__channels_first_input(self, mocker, request):
        # Use 256 elements so that, after quantization to int8, the input can
        # cover the full discrete range [-128, 127].
        # The dataset is generated as a linear float ramp and later quantized,
        # which effectively exercises all int8 values.
        input_shape = (1, 4, 8, 8)
        model = ConvNegModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Neg: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator=LinearRampDatasetCreator(low=-1.0, high=1.0),
        )
