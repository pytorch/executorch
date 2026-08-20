# Copyright 2025-2026 NXP
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
from executorch.backends.nxp.tests.ops_aliases import Sigmoid
from torch import nn
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestSigmoid:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker, request, use_qat=False):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Sigmoid: 1},
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    def test__basic_nsys_inference__qat(self, mocker, request, use_qat):
        input_shape = (23,)
        model = nn.Sigmoid()
        self.assert_delegated(model, input_shape, mocker, request, use_qat=use_qat)

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
    def test__input_shapes(self, mocker, request, input_shape):
        model = nn.Sigmoid()

        self.assert_delegated(model, input_shape, mocker, request)
