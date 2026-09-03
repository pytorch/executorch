# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import torch
from executorch.backends.nxp.tests.nsys_testing import ReferenceModel
from executorch.backends.nxp.tests.dataset_creator import (
    FromCalibrationDataDatasetCreator,
)
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)

from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
import pytest
from executorch.examples.nxp.models.mlperf_tiny.keyword_spotting.mlperf_tiny_keyword_spotting import (
    MLPerfTinyKeywordSpotting,
)

BOUNDS_MSE = {
    "PTQ": {
        "channels-last": 9.5e-3,
        "channels-first": 9.5e-3,
    },
    "QAT": {
        "channels-last": 9.5e-3,
        "channels-first": 9.5e-3,
    },
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [True])
def test_mlperf_tiny_kws_mse_cpu_vs_npu(mocker, request, channels_last):
    channels_last = True
    use_qat = False
    # 5 samples per class
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

    bounds_key_1 = "QAT" if use_qat else "PTQ"
    bounds_key_2 = "channels-last" if channels_last else "channels-first"
    mse = BOUNDS_MSE[bounds_key_1][bounds_key_2]
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = (
        partial(kws.train_model_fn, channels_last=channels_last)
        if use_qat
        else None
    )

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


# @pytest.mark.xfail(reason="EIEX-512", strict=True)
# def test_mlperf_tiny_kws_ptq_qat_equivalence(request):
#     # 5 samples per class
#     num_samples = 60

#     kws = MLPerfTinyKeywordSpotting(num_samples=num_samples, use_random_dataset=True)

#     model = kws.get_eager_model()
#     dataset = kws.dataset
#     labels = kws.labels

#     dataset_creator = FromCalibrationDataDatasetCreator(
#         dataset, num_examples=num_samples, idx_to_label=labels
#     )
#     comparator = ClassificationAccuracyOutputComparator(class_dict=labels)

#     input_spec = ModelInputSpec(kws.input_shape)
#     model_verifier = BaseGraphVerifier(1, [])

#     lower_run_compare_ptq_qat(
#         model,
#         [input_spec],
#         model_verifier,
#         request,
#         train_fn=kws.train_model_fn,
#         dataset_creator=dataset_creator,
#         output_comparator=comparator,
#     )
