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
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Rsqrt
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class RsqrtModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(x)


class TestRsqrt:
    def assert_delegated(self, model, input_shape, mocker, request, use_qat=False):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Rsqrt: 1},
            expected_non_delegated_ops={},
        )

        # Use positive-only values because rsqrt is only defined for x > 0.
        dataset_creator = RandomDatasetCreator(low=0.1, high=2.0)

        # Allow a single quantization bit error in the output.
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            use_qat=use_qat,
        )

    def test__basic_nsys_inference(self, mocker, request):
        input_shape = (2, 13, 7, 9)
        model = RsqrtModule()
        self.assert_delegated(model, input_shape, mocker, request)

    def test__basic_nsys_inference__qat(self, mocker, request, use_qat):
        input_shape = (3, 5, 7, 11)
        model = RsqrtModule()
        self.assert_delegated(model, input_shape, mocker, request, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((2,), id="1D"),
            pytest.param((2, 3), id="2D"),
            pytest.param((2, 3, 5), id="3D"),
            pytest.param((2, 3, 5, 7), id="4D"),
        ],
    )
    def test__input_shapes(self, mocker, request, input_shape):
        model = RsqrtModule()
        self.assert_delegated(model, input_shape, mocker, request)
