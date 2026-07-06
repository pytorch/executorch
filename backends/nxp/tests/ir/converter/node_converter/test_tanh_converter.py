# Copyright 2025-2026 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dWithActivation
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Convolution, Tanh
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


class TanhModule(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return torch.tanh_(x)
        else:
            return torch.tanh(x)


class TestTanh:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        request,
        use_qat=False,
        expected_delegated_ops=None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {Tanh: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            use_qat=use_qat,
        )

    @pytest.fixture(params=[True, False], ids=lambda inplace: f"inplace = {inplace}")
    def inplace(self, request):
        return request.param

    def test__qat__inplace(self, mocker, request, use_qat, inplace):
        shape = (23,)
        model = TanhModule(inplace)
        self.assert_delegated(model, shape, mocker, request, use_qat=use_qat)

    @pytest.mark.parametrize(
        "shape",
        [
            (16,),
            (3, 5),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 2, 3, 2),
        ],
        ids=lambda shape: f"{len(shape)}D",
    )
    def test__shapes(self, mocker, request, shape):
        model = TanhModule()
        self.assert_delegated(model, shape, mocker, request)

    def test__with_convolution(self, mocker, request):
        input_shape = (1, 3, 12, 16)
        channels = input_shape[1]
        model = Conv2dWithActivation(
            activation=torch.tanh, in_channels=channels, out_channels=channels
        )
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            expected_delegated_ops={Tanh: 1, Convolution: 1},
        )
