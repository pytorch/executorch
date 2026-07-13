# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import AddTensorModule, MaxPoolAddTensorModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class TestAddTensor:
    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 5), id="2D."),
            pytest.param((6, 82), id="2D alt."),
            pytest.param((1, 4, 7), id="3D."),
            pytest.param((1, 68, 7), id="3D alt."),
            pytest.param((2, 4, 3, 15), id="4D."),
            pytest.param((1, 4, 9, 11, 4), id="5D."),
        ],
    )
    def test__basic_nsys_inference(self, mocker, request, x_input_shape):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    def test__basic_nsys_inference_qat(self, mocker, request):
        x_input_spec = ModelInputSpec((1, 4, 7))
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            remove_quant_io_ops=remove_quant_io_ops,
            use_qat=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 6)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((69, 73)), ModelInputSpec((1, 73))],
                id="2 inputs 2D alt.",
            ),
            pytest.param(
                [ModelInputSpec((5, 3, 4)), ModelInputSpec((1, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((4,)), ModelInputSpec((4, 4))], id="2 inputs 1D + 2D."
            ),
            pytest.param(
                [ModelInputSpec((1, 4, 8, 8)), ModelInputSpec((8, 8))],
                id="2 inputs 4D + 2D.",
            ),
        ],
    )
    def test__broadcast(self, mocker, request, input_spec):
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 1)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((1, 3, 4)), ModelInputSpec((5, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((6, 4)), ModelInputSpec((6, 6, 1))],
                id="2 inputs 2D + 3D.",
            ),
        ],
    )
    def test__broadcast_unsupported(self, input_spec):
        # Broadcast where at least one of the inputs is not equal to output is not supported
        model = AddTensorModule()

        delegated_ep = to_quantized_edge_program(model, input_spec).exported_program()

        # Make sure the `add.Tensor` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AddTensor])

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                ModelInputSpec((1, 4, 5, 5)),
                id="4D, product of dims is not a multiple of 8.",
            ),
        ],
    )
    def test__channels_first_input(self, mocker, request, input_spec):
        model = MaxPoolAddTensorModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AddTensor: 1, MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [input_spec, input_spec],
            graph_verifier,
            request,
            dataset_creator,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((1, 8, 5, 5)), ModelInputSpec((1, 8, 5, 1))],
                id="2 inputs 4D + 4D.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 1, 67)), ModelInputSpec((1, 8, 5, 67))],
                id="2 inputs 4D + 4D same width.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 5, 5)), ModelInputSpec((1, 1))],
                id="2 inputs 4D + 2D ones tensor.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 8, 8)), ModelInputSpec((8, 8))],
                id="2 inputs 4D + 2D both dims 8.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 5, 5)), ModelInputSpec((1, 5))],
                id="2 inputs 4D + 2D one dim 5.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 12, 10)), ModelInputSpec((8, 1, 10))],
                id="2 inputs 4D + 3D channels dim 1.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 4, 10)), ModelInputSpec((1, 4, 1))],
                id="2 inputs 4D + 3D channels dim 4.",
            ),
            pytest.param(
                [ModelInputSpec((1, 8, 25, 18)), ModelInputSpec((4, 1, 8, 25, 18))],
                id="2 inputs 4D + 5D.",
            ),
        ],
    )
    def test__broadcast__channels_first_input(self, mocker, request, input_spec):
        model = MaxPoolAddTensorModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AddTensor: 1, MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        # Quantize the dataset and allow a single bit error.
        remove_quant_io_ops = True
        comparator = AllCloseOutputComparator(atol=1)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            request,
            dataset_creator,
            comparator,
            remove_quant_io_ops=remove_quant_io_ops,
        )
