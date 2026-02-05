# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import pytest

from executorch.backends.nxp.tests_models.config_importer import test_config
from executorch.backends.nxp.tests_models.dataset_creator import CopyDatasetCreator
from executorch.backends.nxp.tests_models.executors import convert_run_compare
from executorch.backends.nxp.tests_models.graph_verifier import BaseGraphVerifier, NonDelegatedNode
from executorch.backends.nxp.tests_models.model_output_comparator import NumericalStatsOutputComparator
from executorch.examples.nxp.experimental.cifar_net.cifar_net import CifarNet, store_test_data


@pytest.fixture(scope="module")
def cifar_test_files(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("cifar10_dataset")
    store_test_data(dataset_dir)
    return dataset_dir


def test_cifarnet(mocker, cifar_test_files):
    model = CifarNet(pth_file=os.path.join(test_config.PROJECT_DIR, "examples", "nxp", "experimental", "cifar_net", "cifar_net.pth" )).get_eager_model().eval()

    input_shape = (1, 3, 32, 32)
    non_dlg_nodes = [NonDelegatedNode("aten__softmax_default", 1)]

    comparator = NumericalStatsOutputComparator(max_mse_error=1e-3, is_classification_task=True)
    convert_run_compare(model, input_shape,
                        dataset_creator=CopyDatasetCreator(cifar_test_files),
                        output_comparator=comparator,
                        dlg_model_verifier=BaseGraphVerifier(1, non_dlg_nodes),
                        mocker=mocker)


def test_cifarnet_qat(mocker, cifar_test_files):
    model = CifarNet().get_eager_model().eval()

    input_shape = (1, 3, 32, 32)
    non_dlg_nodes = [NonDelegatedNode("aten__softmax_default", 1)]

    # The higher MSE threshold is due to using weaker "MovingAbs" observers instead of "MinMax" observers.
    # The "MovingAbs" observers capture only limited number of past calibration samples compared to "MinMax",
    # which uses statistics from the whole calibration set.
    comparator = NumericalStatsOutputComparator(max_mse_error=8e-2, is_classification_task=True)
    convert_run_compare(model, input_shape,
                        dataset_creator=CopyDatasetCreator(cifar_test_files),
                        output_comparator=comparator,
                        dlg_model_verifier=BaseGraphVerifier(1, non_dlg_nodes),
                        mocker=mocker,
                        use_qat=True)
