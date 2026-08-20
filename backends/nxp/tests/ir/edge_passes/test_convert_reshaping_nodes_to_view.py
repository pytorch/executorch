# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import SqueezeAddModel, UnsqueezeAddModel
from executorch.backends.nxp.tests.nsys_testing import (
    AllCloseOutputComparator,
    lower_run_compare,
)
from executorch.backends.nxp.tests.ops_aliases import AddTensor, ViewCopy


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class TestConvertToView:
    @staticmethod
    def assert_converted_to_view(model, input_shape, mocker, request, use_qat=False):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={ViewCopy: 1, AddTensor: 1},
            expected_non_delegated_ops={},
        )
        dataset = RandomDatasetCreator(low=-1.0, high=1.0)

        # Use quantized dataset and allow single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset,
            comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((2,), 0, id="1D."),
            pytest.param((7, 3, 5), 2, id="3D."),
            pytest.param((7, 3, 5, 7), -2, id="4D, negative dim."),
            pytest.param((7, 3, 5), 3, id="3D, dim arg is clipped."),
            pytest.param((7, 3, 5), -4, id="3D, dim arg is clipped."),
        ],
    )
    def test__convert_unsqueeze_to_view(self, mocker, input_shape, dim, request):
        model = UnsqueezeAddModel(dim)

        self.assert_converted_to_view(
            model, [input_shape, input_shape], mocker, request
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((7, 1, 1), None, id="3D, dim = None."),
            pytest.param((7, 3, 1), 2, id="3D, dim hit."),
            pytest.param((7, 3, 1), 1, id="3D, dim miss."),
            pytest.param((7, 3, 1), -1, id="3D, negative dim hit."),
            pytest.param((7, 1, 1, 7), [1, 2], id="4D, full dims overlap."),
            pytest.param((7, 1, 3, 7), [1, 2], id="4D, partial dims overlap."),
            pytest.param((1, 7, 3, 7), [1, 2], id="4D, no dims overlap."),
            pytest.param((7, 1, 1, 7), [-2, -3], id="4D, negative full dims overlap."),
            pytest.param(
                (7, 1, 3, 7), [-2, -3], id="4D, negative partial dims overlap."
            ),
            pytest.param((1, 7, 3, 7), [-2, -3], id="4D, negative no dims overlap."),
            pytest.param(
                (7, 1, 1, 7), (1, 2), id="4D, tuple instead of list, full dims overlap."
            ),
            pytest.param(
                (7, 1, 3, 7),
                (1, 2),
                id="4D, tuple instead of list, partial dims overlap.",
            ),
            pytest.param(
                (1, 7, 3, 7), (1, 2), id="4D, tuple instead of list, no dims overlap."
            ),
        ],
    )
    def test__convert_squeeze_to_view(self, mocker, input_shape, dim, request):
        model = SqueezeAddModel(dim)

        self.assert_converted_to_view(
            model, [input_shape, input_shape], mocker, request
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((7, 5, 7, 9), -2, id="4D."),
        ],
    )
    def test__convert_unsqueeze_to_view__qat(self, mocker, input_shape, dim, request):
        model = UnsqueezeAddModel(dim)

        self.assert_converted_to_view(
            model, [input_shape, input_shape], mocker, request, use_qat=True
        )

    @pytest.mark.parametrize(
        "input_shape, dim",
        [
            pytest.param((7, 3, 5, 3), -2, id="4D."),
        ],
    )
    def test__convert_squeeze_to_view__qat(self, mocker, input_shape, dim, request):
        model = SqueezeAddModel(dim)

        self.assert_converted_to_view(
            model, [input_shape, input_shape], mocker, request, use_qat=True
        )
