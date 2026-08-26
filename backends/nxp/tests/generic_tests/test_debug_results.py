# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.models import AddTensorModule, AvgPool2dModule
from executorch.backends.nxp.tests.nsys_testing import (
    get_test_name,
    lower_run_compare,
    OUTPUTS_DIR,
)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_nsys_test_debug_results__single_input(caplog, request):
    # Set log level to DEBUG to create debug results
    caplog.set_level(logging.DEBUG)

    input_shape = (2, 4, 6, 7)
    model = AvgPool2dModule(False, 0)

    graph_verifier = BaseGraphVerifier(1, [])

    lower_run_compare(
        model,
        input_shape,
        graph_verifier,
        request,
        remove_quant_io_ops=True,
    )

    test_name = get_test_name(request)
    # Running by CI scripts adds prefix to the name
    assert "test_nsys_test_debug_results__single_input" in test_name
    assert os.path.isdir(os.path.join(OUTPUTS_DIR, test_name, "diff_cpu_npu_results"))
    assert os.path.isfile(os.path.join(OUTPUTS_DIR, test_name, "summary.yaml"))

    # Check file contains key symbols
    with open(os.path.join(OUTPUTS_DIR, test_name, "summary.yaml")) as f:
        content = f.read()
    keys = [
        "date_time",
        "eiq_neutron_sdk_version",
        "eiq_nsys_version",
        "git_branch",
        "git_commit",
        "test_name",
    ]
    assert all(key in content for key in keys)
    assert os.path.isfile(
        os.path.join(OUTPUTS_DIR, test_name, "tag1_neutron.et.tflite")
    )
    assert os.path.isfile(os.path.join(OUTPUTS_DIR, test_name, "tag1_pure.et.tflite"))

    # Check text tensor variants
    assert os.path.isfile(
        os.path.join(OUTPUTS_DIR, test_name, "dataset", "calibration", "0000.txt")
    )
    assert os.path.isfile(
        os.path.join(OUTPUTS_DIR, test_name, "dataset_quant", "0000.txt")
    )
    assert os.path.isfile(
        os.path.join(OUTPUTS_DIR, test_name, "results_cpu", "0000.bin", "0000.txt")
    )
    assert os.path.isfile(
        os.path.join(OUTPUTS_DIR, test_name, "results_npu", "0000.bin", "0000.txt")
    )
    assert os.path.isfile(
        os.path.join(
            OUTPUTS_DIR, test_name, "diff_cpu_npu_results", "0000.bin", "0000.txt"
        )
    )
    assert os.path.isfile(os.path.join(OUTPUTS_DIR, f"{test_name}.zip"))


class TestNsysDebugResults:
    def test_nsys_test_debug_results__multiple_input(self, caplog, request):
        # Set log level to DEBUG to create debug results
        caplog.set_level(logging.DEBUG)

        input_shape = (1, 4, 7)
        x_input_spec = ModelInputSpec(input_shape)
        model = AddTensorModule()

        graph_verifier = BaseGraphVerifier(1, [])

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            request,
        )

        test_name = get_test_name(request)
        # Running by CI scripts adds prefix to the name
        assert (
            "TestNsysDebugResults__test_nsys_test_debug_results__multiple_input"
            in test_name
        )
        assert os.path.isdir(
            os.path.join(OUTPUTS_DIR, test_name, "diff_cpu_npu_results")
        )
        assert os.path.isfile(os.path.join(OUTPUTS_DIR, test_name, "summary.yaml"))

        # Check file contains key symbols
        with open(os.path.join(OUTPUTS_DIR, test_name, "summary.yaml")) as f:
            content = f.read()
        keys = [
            "date_time",
            "eiq_neutron_sdk_version",
            "eiq_nsys_version",
            "git_branch",
            "git_commit",
            "test_name",
        ]
        assert all(key in content for key in keys)
        assert os.path.isfile(
            os.path.join(OUTPUTS_DIR, test_name, "tag1_neutron.et.tflite")
        )
        assert os.path.isfile(
            os.path.join(OUTPUTS_DIR, test_name, "tag1_pure.et.tflite")
        )

        # Check text tensor variants
        assert os.path.isfile(
            os.path.join(
                OUTPUTS_DIR, test_name, "dataset", "calibration", "0000", "00.txt"
            )
        )
        assert os.path.isfile(
            os.path.join(OUTPUTS_DIR, test_name, "results_cpu", "0000", "0000.txt")
        )
        assert os.path.isfile(
            os.path.join(OUTPUTS_DIR, test_name, "results_npu", "0000", "0000.txt")
        )
        assert os.path.isfile(
            os.path.join(
                OUTPUTS_DIR, test_name, "diff_cpu_npu_results", "0000", "0000.txt"
            )
        )
        assert os.path.isfile(os.path.join(OUTPUTS_DIR, f"{test_name}.zip"))
