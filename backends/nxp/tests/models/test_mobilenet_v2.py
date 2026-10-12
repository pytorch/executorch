# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.dataset_creator import (
    FromCalibrationDataDatasetCreator,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare, ReferenceModel
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.examples.nxp.models.mobilenet_v2 import MobileNetV2

BOUNDS_MSE = {
    "PTQ": {"channels-last": 3.8e-04, "channels-first": 3e-03},
    "QAT": {"channels-last": 6e-04, "channels-first": 3e-03},
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [False, True])
def test_mobilenet_v2_mse_cpu_vs_npu(
    mocker,
    request,
    channels_last,
    use_qat,
):
    num_samples = 1

    mobilenet_v2 = MobileNetV2(
        num_samples=num_samples,
        use_random_dataset=True,
        balanced_dataset=False,
    )
    model = mobilenet_v2.get_eager_model()
    dataset = mobilenet_v2.dataset
    labels = mobilenet_v2.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    input_spec = ModelInputSpec(mobilenet_v2.input_shape)
    if channels_last:
        model.to(memory_format=torch.channels_last)
        input_spec.dim_order = torch.channels_last
    quant_type_key = "QAT" if use_qat else "PTQ"
    format_key = "channels-last" if channels_last else "channels-first"

    mse = BOUNDS_MSE[quant_type_key][format_key]
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = (
        partial(mobilenet_v2.train_model_fn, channels_last=channels_last)
        if use_qat
        else None
    )

    # Run the channels last and QAT reference in Python as the ExecuTorch CPU model produces invalid results
    ref_model = (
        ReferenceModel.QUANTIZED_EDGE_PYTHON
        if channels_last or not use_qat
        else ReferenceModel.QUANTIZED_EXECUTORCH_CPP
    )

    lower_run_compare(
        model,
        [input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        reference_model=ref_model,
        mocker=mocker,
        use_qat=use_qat,
        train_fn=train_fn,
    )
