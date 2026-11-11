# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.backend.ops_aliases import AddTensor
from executorch.backends.nxp.tests.comparators import AllCloseOutputComparator
from executorch.backends.nxp.tests.dataset_creators.synthetic_creators import (
    RandomDatasetCreator,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier

from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
    ReferenceRunner,
)
from executorch.backends.nxp.tests.simple_models import AddTensorModule


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_non_quantized_reference_model(mocker, request):
    x_input_spec = ModelInputSpec((2, 4, 3, 15))
    model = AddTensorModule()
    graph_verifier = DetailedGraphVerifier(
        mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
    )
    dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

    # The non quantized FP32 model accepts only float data
    remove_quant_io_ops = False
    comparator = AllCloseOutputComparator(atol=2e-2)

    lower_run_compare(
        model,
        [x_input_spec, x_input_spec],
        graph_verifier,
        request,
        dataset_creator,
        comparator,
        reference_runner=ReferenceRunner.NON_QUANTIZED_EXECUTORCH_CPP,
        remove_quant_io_ops=remove_quant_io_ops,
    )


@pytest.mark.xfail(
    reason="Expected ValueError, int model input is not allowed.", strict=True
)
def test_non_quantized_reference_model_int_inputs(mocker, request):
    x_input_spec = ModelInputSpec((2, 4, 3, 15))
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
        reference_runner=ReferenceRunner.NON_QUANTIZED_EXECUTORCH_CPP,
        remove_quant_io_ops=remove_quant_io_ops,
    )
