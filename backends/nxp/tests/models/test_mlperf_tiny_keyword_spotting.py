# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import pytest
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
from executorch.examples.nxp.models.mlperf_tiny.keyword_spotting.mlperf_tiny_keyword_spotting import (
    MLPerfTinyKeywordSpotting,
)

BOUNDS_MSE = {
    "PTQ": {
        "channels-last": np.inf,
        "channels-first": 5.5e-7,
    },
    "QAT": {
        "channels-last": np.inf,
        "channels-first": 3.3e-5,
    },
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "channels_last",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="EIEX-1082, don't forget to readjust bounds when it start working",
                strict=True,
            ),
        ),
    ],
)
def test_mlperf_tiny_kws_mse_cpu_vs_npu(mocker, request, channels_last, use_qat):
    # approx. 5 samples per class
    num_samples = 60

    kws = MLPerfTinyKeywordSpotting(num_samples=num_samples, use_random_dataset=True)
    model = kws.get_eager_model()
    dataset = kws.dataset
    labels = kws.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    input_spec = ModelInputSpec(kws.input_shape)
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
        partial(kws.train_model_fn, channels_last=channels_last) if use_qat else None
    )

    # This model does not work in channels-last format when running with portable kernels.
    # See more information below.
    # Github issue: https://github.com/pytorch/executorch/issues/22520
    # NXP internal issue ID: EIEX-1074
    ref_model = (
        ReferenceModel.QUANTIZED_EDGE_PYTHON
        if channels_last
        else ReferenceModel.QUANTIZED_EXECUTORCH_CPP
    )

    lower_run_compare(
        model,
        [input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        mocker=mocker,
        reference_model=ref_model,
        use_qat=use_qat,
        train_fn=train_fn,
    )


def test_mlperf_tiny_kws_ptq_qat_equivalence(request):
    # approx. 5 samples per class
    num_samples = 60

    kws = MLPerfTinyKeywordSpotting(num_samples=num_samples, use_random_dataset=True)

    model = kws.get_eager_model()
    dataset = kws.dataset
    labels = kws.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )
    comparator = ClassificationAccuracyOutputComparator(class_dict=labels)

    input_spec = ModelInputSpec(kws.input_shape)
    model_verifier = BaseGraphVerifier(1, [])

    lower_run_compare_ptq_qat(
        model,
        [input_spec],
        model_verifier,
        request,
        train_fn=kws.train_model_fn,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
    )
