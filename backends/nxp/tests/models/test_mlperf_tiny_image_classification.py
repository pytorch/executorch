# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import torch
from executorch.backends.nxp.tests.dataset_creator import (
    FromCalibrationDataDatasetCreator,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    ClassificationAccuracyOutputComparator,
    NumericalStatsOutputComparator,
)

from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
    lower_run_compare_ptq_qat,
    ReferenceModel,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
import pytest
from executorch.examples.nxp.models.mlperf_tiny.image_classification.mlperf_tiny_image_classification import (
    MLPerfTinyImageClassification,
)

BOUNDS_MSE = {
    "PTQ": {"channels-last": 1.859e-05, "channels-first": 7.432e-09},
    "QAT": {"channels-last": 2.751e-07, "channels-first": 5.205e-06},
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [False, True])
def test_mlperf_tiny_classification_mse_cpu_vs_npu(
    mocker, request, channels_last, use_qat
):
    # 10 samples per class
    num_samples = 60

    img_classification = MLPerfTinyImageClassification(
        num_samples=num_samples, use_random_dataset=True
    )
    model = img_classification.get_eager_model()
    dataset = img_classification.dataset
    labels = img_classification.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    input_spec = ModelInputSpec(img_classification.input_shape)
    if channels_last:
        model.to(memory_format=torch.channels_last)
        input_spec.dim_order = torch.channels_last

    quant_type_key = "QAT" if use_qat else "PTQ"
    dim_order_key = "channels-last" if channels_last else "channels-first"

    mse = BOUNDS_MSE[quant_type_key][dim_order_key]
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, use_softmax=True, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = (
        partial(img_classification.train_model_fn, channels_last=channels_last)
        if use_qat
        else None
    )

    # This model does not work in channels-last format and QAT. See more information below.
    # Github issue: https://github.com/pytorch/executorch/issues/22179
    # NXP internal issue ID: EIEX-1065
    ref_model = (
        ReferenceModel.QUANTIZED_EDGE_PYTHON
        if channels_last and use_qat
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


def test_mlperf_tiny_image_classification_ptq_qat_equivalence(request):
    # 10 samples per class
    num_samples = 60

    img_classification = MLPerfTinyImageClassification(
        num_samples=num_samples, use_random_dataset=True
    )

    model = img_classification.get_eager_model()
    dataset = img_classification.dataset
    labels = img_classification.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )
    comparator = ClassificationAccuracyOutputComparator(class_dict=labels)

    input_spec = ModelInputSpec(img_classification.input_shape)
    model_verifier = BaseGraphVerifier(1, [])

    lower_run_compare_ptq_qat(
        model,
        [input_spec],
        model_verifier,
        request,
        train_fn=img_classification.train_model_fn,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
    )
