# Copyright 2025-2026 NXP
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
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier, Operator
from executorch.backends.nxp.tests.models import AddmmModule, LinearModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddMM,
    ExecutorchDelegateCall,
    MM,
    PermuteCopy,
    ViewCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class TestAddMM:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        use_qat=False,
        expected_delegated_ops: dict[Operator, int] | None = None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {AddMM: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_qat=use_qat,
        )

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AddMM, MM])

    @pytest.mark.parametrize(
        "input_shape",
        [
            # PyTorch allows only 2D inputs.
            (1, 32),
            (3, 11),
        ],
        ids=lambda s: f"input_shape = {s}",
    )
    def test__from_addmm(self, mocker, use_qat, input_shape: tuple[int, ...]):
        model = AddmmModule(input_shape[-1])
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    def test__from_addmm__unsupported_alpha(self):
        input_shape = (1, 8)
        model = AddmmModule(input_shape[-1], alpha=0.42)
        self.assert_not_delegated(model, input_shape)

    def test__from_addmm__unsupported_beta(self):
        input_shape = (1, 8)
        model = AddmmModule(input_shape[-1], beta=0.42)
        self.assert_not_delegated(model, input_shape)

    @pytest.mark.parametrize(
        "alpha",
        [1, 1.0],
        ids=lambda a: f"alpha = {a}",
    )
    def test__from_addmm__supported_alpha(self, mocker, use_qat, alpha):
        input_shape = (1, 8)
        model = AddmmModule(input_shape[-1], alpha=alpha)
        self.assert_delegated(model, input_shape, mocker, use_qat)

    @pytest.mark.parametrize(
        "beta",
        [1, 1.0],
        ids=lambda b: f"beta = {b}",
    )
    def test__from_addmm__supported_beta(self, mocker, use_qat, beta):
        input_shape = (1, 8)
        model = AddmmModule(input_shape[-1], beta=beta)
        self.assert_delegated(model, input_shape, mocker, use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 32),
            (3, 11),
        ],
        ids=lambda s: f"input_shape = {s}",
    )
    def test__from_linear_with_bias__2d(
        self, mocker, use_qat, input_shape: tuple[int, ...]
    ):
        model = LinearModule(bias=True, in_features=input_shape[-1], out_features=7)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            use_qat=use_qat,
            expected_delegated_ops={AddMM: 1, PermuteCopy: 1},
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 3, 8),
            (2, 3, 5),
            (2, 3, 3, 3),
        ],
        ids=lambda s: f"input_shape = {s}",
    )
    def test__from_linear_with_bias__higher_ranks(
        self, mocker, use_qat, input_shape: tuple[int, ...]
    ):
        # More than 2D cases get reshaped to 2D, so two extra view_copy nodes are delegated.

        model = LinearModule(bias=True, in_features=input_shape[-1], out_features=7)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            use_qat=use_qat,
            expected_delegated_ops={AddMM: 1, PermuteCopy: 1, ViewCopy: 2},
        )
