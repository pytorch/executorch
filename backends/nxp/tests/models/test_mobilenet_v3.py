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
from executorch.backends.nxp.tests.graph_verifier import (
    BaseGraphVerifier,
    NonDelegatedNode,
)
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.examples.nxp.models.mobilenet_v3 import MobileNetV3

BOUNDS_MSE = {
    "PTQ": {"channels-last": 7e-03, "channels-first": 8e-03},
    "QAT": {"channels-last": 6e-03, "channels-first": 3e-03},
}


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [False, True])
def test_mobilenet_v3_mse_cpu_vs_npu(
    mocker,
    request,
    channels_last,
    use_qat,
):
    num_samples = 1

    mobilenet_v3 = MobileNetV3(
        num_samples=num_samples,
        use_random_dataset=True,
        balanced_dataset=False,
    )
    model = mobilenet_v3.get_eager_model()
    dataset = mobilenet_v3.dataset
    labels = mobilenet_v3.labels

    dataset_creator = FromCalibrationDataDatasetCreator(
        dataset, num_examples=num_samples, idx_to_label=labels
    )

    input_spec = ModelInputSpec(mobilenet_v3.input_shape)
    if channels_last:
        model.to(memory_format=torch.channels_last)
        input_spec.dim_order = torch.channels_last
    quant_type_key = "QAT" if use_qat else "PTQ"
    format_key = "channels-last" if channels_last else "channels-first"

    mse = BOUNDS_MSE[quant_type_key][format_key]
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(
        1,
        [
            NonDelegatedNode("aten_add_tensor", 13),
            NonDelegatedNode("aten_div_tensor", 9),
            NonDelegatedNode("aten_clamp_default", 18),
        ],
    )
    train_fn = (
        partial(mobilenet_v3.train_model_fn, channels_last=channels_last)
        if use_qat
        else None
    )

    ref_input_spec = ModelInputSpec(input_spec.shape)
    ref_input_spec.dim_order = torch.contiguous_format

    lower_run_compare(
        model,
        [input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        reference_input_spec=[ref_input_spec],
        mocker=mocker,
        use_qat=use_qat,
        train_fn=train_fn,
    )
