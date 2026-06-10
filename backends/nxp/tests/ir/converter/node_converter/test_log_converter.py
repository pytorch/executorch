# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Log
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.backends.nxp.tests.dataset_creator import (
    LinearRampDatasetCreator,
    RandomDatasetCreator,
)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class LogModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


class TestLog:
    def test__basic_nsys_inference(self, mocker):
        # Use 256 elements so that, after quantization to int8, the input can
        # cover the full discrete range [-128, 127].
        # The dataset is generated as a linear float ramp and later quantized,
        # which effectively exercises all int8 values.
        input_shape = (256,)
        model = LogModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Log: 1}, expected_non_delegated_ops={}
        )
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator=LinearRampDatasetCreator(low=0.0, high=1.0),
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((17, 2), id="2D"),
            pytest.param((1, 3, 10), id="3D"),
            pytest.param((1, 3, 16, 16), id="4D"),
        ],
    )
    def test__basic_nsys_inference__qat(self, mocker, input_shape, use_qat):
        model = LogModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Log: 1}, expected_non_delegated_ops={}
        )
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator=RandomDatasetCreator(low=1.0, high=10.0),
            use_qat=use_qat,
        )
