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
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.examples.nxp.models.mlperf_tiny.visual_wake_words.mlperf_tiny_visual_wake_words import (
    MLPerfTinyVisualWakeWords,
)

BOUNDS_MSE = {
    "PTQ": {"channels-last": 0.0, "channels-first": 0.0},
    "QAT": {"channels-last": 1.0e-6, "channels-first": 1.0e-7},
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [False, True])
def test_mlperf_tiny_vww_mse_cpu_vs_npu(mocker, request, channels_last, use_qat):
    num_samples = 1

    visual_wake_words = MLPerfTinyVisualWakeWords(
        num_samples=num_samples, use_random_dataset=True, balanced_dataset=False
    )
    model = visual_wake_words.get_eager_model()
    dataset = visual_wake_words.dataset
    labels = visual_wake_words.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    model_input_spec = ModelInputSpec(visual_wake_words.input_shape)
    if channels_last:
        model.to(memory_format=torch.channels_last)
        model_input_spec.dim_order = torch.channels_last

    quant_type_key = "QAT" if use_qat else "PTQ"
    format_key = "channels-last" if channels_last else "channels-first"
    mse = BOUNDS_MSE[quant_type_key][format_key]
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, use_softmax=True, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = (
        partial(visual_wake_words.train_model_fn, channels_last=channels_last)
        if use_qat
        else None
    )

    ref_input_spec = ModelInputSpec(visual_wake_words.input_shape)
    ref_input_spec.dim_order = torch.contiguous_format

    lower_run_compare(
        model,
        [model_input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        reference_input_spec=[ref_input_spec],
        mocker=mocker,
        use_qat=use_qat,
        train_fn=train_fn,
    )
