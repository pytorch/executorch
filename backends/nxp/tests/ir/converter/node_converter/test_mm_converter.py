# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier, Operator
from executorch.backends.nxp.tests.models import LinearModule, MmModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import MM, PermuteCopy, ViewCopy
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class TestMM:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        request,
        use_qat=False,
        expected_delegated_ops: dict[Operator, int] | None = None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {MM: 1}

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
            request,
            dataset_creator,
            use_qat=use_qat,
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            # PyTorch allows only 2D inputs.
            (1, 32),
            (3, 11),
        ],
        ids=lambda s: f"input_shape = {s}",
    )
    def test__from_mm(self, mocker, request, use_qat, input_shape: tuple[int, ...]):
        model = MmModule(input_shape[-1])
        self.assert_delegated(model, input_shape, mocker, request, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            (1, 32),
            (3, 11),
        ],
        ids=lambda s: f"input_shape = {s}",
    )
    def test__from_linear_without_bias(
        self, mocker, request, use_qat, input_shape: tuple[int, ...]
    ):
        model = LinearModule(bias=False, in_features=input_shape[-1], out_features=7)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            use_qat=use_qat,
            expected_delegated_ops={MM: 1, PermuteCopy: 1},
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
    def test__from_linear_without_bias__higher_ranks(
        self, mocker, request, use_qat, input_shape: tuple[int, ...]
    ):
        # More than 2D cases get reshaped to 2D, so two extra view_copy nodes are delegated.

        model = LinearModule(bias=False, in_features=input_shape[-1], out_features=7)
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            use_qat=use_qat,
            expected_delegated_ops={MM: 1, PermuteCopy: 1, ViewCopy: 2},
        )
