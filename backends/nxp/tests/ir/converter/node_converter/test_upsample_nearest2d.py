# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    UpsampleNearest2D,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class UpsampleNearestModule(torch.nn.Module):

    def __init__(self, size=None, scale=None):
        super().__init__()
        self.upsample = torch.nn.Upsample(size=size, scale_factor=scale, mode="nearest")

    def forward(self, x):
        return self.upsample(x)


class UpsampleNearestAddModule(UpsampleNearestModule):

    def forward(self, x):
        x = super().forward(x)
        return x + x


class TestUpsampleNearest2D:

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
            expected_delegated_ops = {UpsampleNearest2D: 1}

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

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [UpsampleNearest2D])

    def test__qat(self, mocker, request, use_qat):
        input_shape = (1, 2, 3, 4)
        output_size = (6, 8)
        model = UpsampleNearestModule(size=output_size)
        self.assert_delegated(model, input_shape, mocker, request, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 2, 3, 4), (6, 8), id="batch=1, scale_h=scale_w=2"),
            pytest.param((1, 2, 3, 3), 6, id="batch=1, scale_h=scale_w=2, scalar size"),
            pytest.param(
                (3, 3, 3, 5),
                (6, 5),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (3, 16), id="batch=2, scale_h=1, scale_w=4"),
            pytest.param((2, 2, 3, 4), (24, 8), id="batch=2, scale_h=8, scale_w=2"),
        ],
    )
    def test__output_size(self, mocker, request, input_shape, output_size):
        model = UpsampleNearestModule(size=output_size)
        self.assert_delegated(model, input_shape, mocker, request)

    def test__output_size__unsupported(self):
        input_shape = (1, 2, 3, 4)
        output_size = (9, 12)  # scale = (3, 3)
        model = UpsampleNearestModule(size=output_size)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "input_shape, scale",
        [
            pytest.param((1, 2, 3, 4), (2, 2), id="batch=1, scale_h=scale_w=2"),
            pytest.param(
                (1, 2, 3, 4), 4, id="batch=1, scale_h=scale_w=4, scalar scale"
            ),
            pytest.param(
                (3, 3, 3, 5),
                (2, 1),
                id="batch=3, scale_h=2, scale_w=1 (no num_macs multiples)",
            ),
            pytest.param((2, 2, 3, 4), (4, 1), id="batch=2, scale_h=4, scale_w=1"),
            pytest.param((2, 2, 3, 4), (2, 8), id="batch=2, scale_h=2, scale_w=8"),
        ],
    )
    def test__scales(self, mocker, request, input_shape, scale):
        model = UpsampleNearestModule(scale=scale)
        self.assert_delegated(model, input_shape, mocker, request)

    def test__scales__unsupported(self):
        input_shape = (1, 2, 3, 4)
        scale = (3, 3)
        model = UpsampleNearestModule(scale=scale)
        self.assert_not_delegated(model, input_shape)

    def test__noop__alone_in_partition__not_delegated(self):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleNearestModule(scale=scale)
        self.assert_not_delegated(model, input_shape)

    def test__noop__not_alone_in_partition__delegated(self, mocker, request):
        input_shape = (1, 2, 3, 4)
        scale = 1
        model = UpsampleNearestAddModule(scale=scale)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            expected_delegated_ops={UpsampleNearest2D: 1, AddTensor: 1},
        )
