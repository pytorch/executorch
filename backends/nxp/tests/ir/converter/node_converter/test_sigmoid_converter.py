# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import DequantizePerTensor, Sigmoid
from torch import nn
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestSigmoid:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self, model, input_shape, mocker, request, use_qat=False, atol=None
    ):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Sigmoid: 1},
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        kwargs = {"atol": atol} if atol is not None else {}
        output_comparator = AllCloseOutputComparator(**kwargs)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
            output_comparator,
            use_qat=use_qat,
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

        output_scale = 1.0 / 256.0
        lowering_spy = mocker.spy(NeutronPartitioner, "partition")
        self.assert_delegated(
            model, input_shape, mocker, request, atol=output_scale
        )  # Allow single bit error.

        # Verify that the `atol` is indeed equal to the output scale.
        # In the near future, we would like to add support for testing with int8 IO, where this check will be trivial.
        nodes = list(lowering_spy.spy_return.tagged_exported_program.graph.nodes)
        assert nodes[-2].target == DequantizePerTensor
        assert nodes[-2].args[1] == output_scale
