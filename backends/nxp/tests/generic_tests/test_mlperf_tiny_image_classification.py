from functools import partial

import numpy as np
import torch
from executorch.backends.nxp.tests.calibration_dataset import RandomCalibrationDataset
from executorch.backends.nxp.tests.dataset_creator import FromCalibrationDataDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    NumericalStatsOutputComparator,
    ClassificationAccuracyOutputComparator,
)

from executorch.backends.nxp.tests.nsys_testing import ReferenceModel
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare, lower_run_compare_ptq_qat
from executorch.backends.nxp.tests.use_qat import *
from executorch.examples.nxp.models.mlperf_tiny.image_classification.image_classification import \
    ImageClassification

import pytest


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("channels_last", [False, True])
def test_mlperf_tiny_classification_mse_cpu_vs_npu(mocker, request, channels_last, use_qat):
    image_classification = ImageClassification()
    model = image_classification.get_eager_model()
    dataset = RandomCalibrationDataset(120, image_classification._input_shape[1:], image_classification._num_classes)

    idx_to_label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    dataset_creator = FromCalibrationDataDatasetCreator(dataset, num_examples=60, idx_to_label=idx_to_label)

    input_spec = ModelInputSpec((1, 3, 32, 32))
    if channels_last:
        model.to(memory_format=torch.channels_last)
        input_spec.dim_order = torch.channels_last

    mse = 6.79e-3 if use_qat else 2.56e-3 # Not sure why the QAT mse is a bit higher.
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, use_softmax=True, is_classification_task=True
    )
    model_verifier = BaseGraphVerifier(1, [])
    train_fn = (
        partial(
            image_classification.train_model_fn,
            channels_last=channels_last
        )
        if use_qat
        else None
    )

    lower_run_compare(
        model, [input_spec],
        model_verifier,
        request,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
        mocker=mocker,
        # Run the channels last reference in Python as the ExecuTorch CPU model would contain an incorrectly lowered
        #  operator (mean), which causes a crash in the c++ kernel. The issue is caused by ExecuTorch (not NXP).
        #  https://github.com/pytorch/executorch/issues/16507
        reference_model=ReferenceModel.QUANTIZED_EDGE_PYTHON if channels_last else ReferenceModel.QUANTIZED_EXECUTORCH_CPP,
        use_qat=use_qat,
        train_fn=train_fn
    )


def test_mlperf_tiny_image_classification_ptq_qat_equivalence(request):
    image_classification = ImageClassification()
    
    model = image_classification.get_eager_model()
    dataset = RandomCalibrationDataset(120, image_classification._input_shape()[1:], image_classification._num_classes())

    input_spec = ModelInputSpec((1, 3, 32, 32))
    idx_to_label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    dataset_creator = FromCalibrationDataDatasetCreator(dataset, num_examples=60, idx_to_label=idx_to_label)
    comparator = ClassificationAccuracyOutputComparator(class_dict=idx_to_label)
    model_verifier = BaseGraphVerifier(1, [])

    lower_run_compare_ptq_qat(
        model, [input_spec],
        model_verifier,
        request,
        train_fn=image_classification.train_model_fn,
        dataset_creator=dataset_creator,
        output_comparator=comparator,
    )
