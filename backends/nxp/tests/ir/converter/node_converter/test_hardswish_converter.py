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
from executorch.backends.nxp.tests.models import (
    ConvHardswishModule,
    HardswishModule,
    LinearHardswishModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddMM,
    Convolution,
    Hardswish,
    PermuteCopy,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestHardswishConverter:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        request,
        expected_delegated_ops=None,
        use_qat=False,
    ):
        rank = len(input_shape)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops
            or {
                Hardswish: 1,
                AddMM: 1,
                PermuteCopy: 1,
                ViewCopy: 0 if rank == 2 else 2,
            },
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-4, high=4)
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            output_comparator=comparator,
            remove_quant_io_ops=True,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((7, 83), id="2D."),
            pytest.param((7, 8, 12), id="3D."),
            pytest.param((1, 4, 7, 8), id="4D."),
            pytest.param((5, 4, 7, 8), id="4D batchsize != 1."),
            pytest.param((1, 4, 3, 4, 14), id="5D."),
        ],
    )
    def test__basic_nsys_inference(self, mocker, request, input_shape):
        channels = input_shape[-1]
        model = LinearHardswishModule(in_features=channels, out_features=channels)

        self.assert_delegated(model, input_shape, mocker, request)

    def test__basic_nsys_inference_qat(self, mocker, request):
        input_shape = (2, 4, 6, 7)
        channels = input_shape[-1]
        model = LinearHardswishModule(in_features=channels, out_features=channels)

        self.assert_delegated(model, input_shape, mocker, request, use_qat=True)

    def test__basic_nsys_inference_inplace(self, mocker, request, use_qat):
        input_shape = (2, 4, 6, 7)
        channels = input_shape[-1]
        model = LinearHardswishModule(
            in_features=channels, out_features=channels, inplace=True
        )

        self.assert_delegated(model, input_shape, mocker, request, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((3,), id="1D."),
            pytest.param((1, 4), id="2D."),
            pytest.param((4, 7, 4), id="3D."),
            pytest.param((1, 6, 4, 4), id="4D."),
            pytest.param((5, 4, 7, 8), id="4D batchsize != 1."),
            pytest.param((2, 3, 8, 3, 11), id="5D."),
        ],
    )
    def test__single_hardswish(self, mocker, request, input_shape):
        model = HardswishModule()
        expected_delegated_ops = {
            Hardswish: 1,
        }

        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            expected_delegated_ops=expected_delegated_ops,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            pytest.param((1, 8, 4, 4), id="4D."),
        ],
    )
    def test__channels_first(self, mocker, request, input_shape):
        channels = input_shape[1]
        model = ConvHardswishModule(in_channels=channels)
        expected_delegated_ops = {
            Hardswish: 1,
            Convolution: 1,
        }

        self.assert_delegated(
            model, input_shape, mocker, request, expected_delegated_ops
        )
