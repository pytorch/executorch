# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import pytest
import torch

from executorch.backends.nxp.tests_models.config_importer import test_config
from executorch.backends.nxp.tests_models.dataset_creator import CopyDatasetCreator
from executorch.backends.nxp.tests_models.executors import convert_run_compare
from executorch.backends.nxp.tests_models.graph_verifier import (
    BaseGraphVerifier,
    NonDelegatedNode,
)
from executorch.backends.nxp.tests_models.model_input_spec import ModelInputSpec
from executorch.backends.nxp.tests_models.model_output_comparator import (
    NumericalStatsOutputComparator,
)
from executorch.examples.nxp.experimental.cifar_net.cifar_net import (
    CifarNet,
    store_test_data,
)


@pytest.fixture(scope="module")
def cifar_test_files(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("cifar10_dataset")
    store_test_data(dataset_dir)
    return dataset_dir


@pytest.mark.parametrize("channels_last", [False, True])
def test_cifarnet(mocker, cifar_test_files, channels_last):
    model = (
        CifarNet(
            pth_file=os.path.join(
                test_config.PROJECT_DIR,
                "examples",
                "nxp",
                "experimental",
                "cifar_net",
                "cifar_net.pth",
            )
        )
        .get_eager_model()
        .eval()
    )

    input_spec = ModelInputSpec((1, 3, 32, 32))

    if channels_last:
        model.to(memory_format=torch.channels_last)
        input_spec.dim_order = torch.channels_last

    non_dlg_nodes = [NonDelegatedNode("aten__softmax_default", 1)]

    mse = 2.4e-3 if channels_last else 1e-3
    comparator = NumericalStatsOutputComparator(
        max_mse_error=mse, is_classification_task=True
    )
    convert_run_compare(
        model,
        [input_spec],
        dataset_creator=CopyDatasetCreator(cifar_test_files),
        output_comparator=comparator,
        dlg_model_verifier=BaseGraphVerifier(1, non_dlg_nodes),
        mocker=mocker,
        # Run the channels last reference in PyTorch as the ExecuTorch CPU model contains incorrectly
        #  lowered channels last convolution weights, which cause incorrect inference results. The issue
        #  is caused by ExecuTorch (not NXP). https://github.com/pytorch/executorch/issues/16464
        run_cpu_version_in_pytorch=channels_last,
    )


def test_cifarnet_qat(mocker, cifar_test_files):
    model = CifarNet().get_eager_model().eval()

    input_shape = (1, 3, 32, 32)
    non_dlg_nodes = [NonDelegatedNode("aten__softmax_default", 1)]

    # The higher MSE threshold is due to using weaker "MovingAbs" observers instead of "MinMax" observers.
    # The "MovingAbs" observers capture only limited number of past calibration samples compared to "MinMax",
    # which uses statistics from the whole calibration set.
    comparator = NumericalStatsOutputComparator(
        max_mse_error=8e-2, is_classification_task=True
    )
    convert_run_compare(
        model,
        input_shape,
        dataset_creator=CopyDatasetCreator(cifar_test_files),
        output_comparator=comparator,
        dlg_model_verifier=BaseGraphVerifier(1, non_dlg_nodes),
        mocker=mocker,
        use_qat=True,
    )
