# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import (
    ViewCopy,
)
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import (
    BatchMatMulMaxPoolModel,
    BatchMatMulModel,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import BMM, GetItem, MaxPool2DWithIndices
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)


class TestBMM:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_1_shape,
        input_2_shape,
        mocker,
        use_qat=False,
        expected_delegated_ops=None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {BMM: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Use quantized dataset and allow a single-bit error.
        remove_quant_io_ops = True
        output_comparator = AllCloseOutputComparator(atol=1)

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            [ModelInputSpec(input_1_shape), ModelInputSpec(input_2_shape)],
            graph_verifier,
            dataset_creator,
            output_comparator=output_comparator,
            use_qat=use_qat,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    def test__qat(self, mocker, use_qat):
        input_1_shape, input_2_shape = (1, 24, 16), (1, 16, 24)
        model = BatchMatMulModel()
        self.assert_delegated(
            model, input_1_shape, input_2_shape, mocker, use_qat=use_qat
        )

    # Input shape parameters used by 2 tests in this class.
    BMM_SHAPES = [
        pytest.param((3, 8, 24), (3, 24, 8), id="more batches"),
        pytest.param((2, 24, 16), (2, 16, 8), id="more batches, x1_C != x2_W"),
        pytest.param(
            (1, 8, 7), (1, 7, 16), id="x1_W (and x2_C) not divisible by NUM_MACS"
        ),
        pytest.param((1, 7, 16), (1, 16, 8), id="x1_C not divisible by NUM_MACS"),
        pytest.param((1, 8, 16), (1, 16, 7), id="x2_W not divisible by NUM_MACS"),
        pytest.param((3, 5, 7), (3, 7, 11), id="nothing divisible by NUM_MACS"),
    ]

    @pytest.mark.parametrize(
        "input_1_shape, input_2_shape",
        BMM_SHAPES,
    )
    def test__nsys_inference(self, mocker, input_1_shape, input_2_shape):
        model = BatchMatMulModel()
        self.assert_delegated(model, input_1_shape, input_2_shape, mocker)

    @pytest.mark.parametrize(
        "input_1_shape, input_2_shape",
        BMM_SHAPES,
    )
    def test__channels_first(self, mocker, input_1_shape, input_2_shape):
        model = BatchMatMulMaxPoolModel()
        self.assert_delegated(
            model,
            input_1_shape,
            input_2_shape,
            mocker,
            expected_delegated_ops={
                BMM: 1,
                MaxPool2DWithIndices: 1,
                GetItem: 1,
                ViewCopy: 2,
            },
        )
