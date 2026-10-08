# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.examples.nxp.models.mlperf_tiny.anomaly_detection.mlperf_tiny_anomaly_detection import (
    MLPerfTinyAnomalyDetection,
)

BOUNDS_MSE = {
    "PTQ": 1.4e-08,
    "QAT": 5.205e-06,
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_mlperf_tiny_anomaly_detection_mse_cpu_vs_npu(
    mocker,
    request,
    use_qat,
):
    num_samples = 1

    anomaly_detection = MLPerfTinyAnomalyDetection(
        num_samples=num_samples, use_random_dataset=True, balanced_dataset=False
    )
    model = anomaly_detection.get_eager_model()
    dataset = anomaly_detection.dataset
    labels = anomaly_detection.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    input_spec = ModelInputSpec(anomaly_detection.input_shape)
    quant_type_key = "QAT" if use_qat else "PTQ"

    mse = BOUNDS_MSE[quant_type_key]
    comparator = NumericalStatsOutputComparator(max_mse_error=mse)
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = anomaly_detection.train_model_fn if use_qat else None

    lower_run_compare(
        model,
        [input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        mocker=mocker,
        use_qat=use_qat,
        train_fn=train_fn,
    )
