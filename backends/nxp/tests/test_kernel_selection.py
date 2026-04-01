# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import eiq_neutron_sdk
import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import (
    AdaptiveAvgPool2dConvModule,
    Conv2dReLUMaxPoolModule,
)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_neutron_converter_kernel_selection():
    input_shape = (1, 4, 16, 16)
    model = AdaptiveAvgPool2dConvModule((4, 4))

    # Run conversion
    _ = to_quantized_edge_program(
        model,
        input_shape,
        dump_kernel_selection_code=True,
    ).exported_program()

    # Check for the created kernel selection file
    kernel_selection_file = [
        fname for fname in os.listdir(".") if fname.endswith("_kernel_selection.c")
    ]
    assert len(kernel_selection_file) == 1
    assert os.path.isfile(kernel_selection_file[0])

    # Check file contains key symbols
    with open(kernel_selection_file[0]) as f:
        content = f.read()
    assert "typedef enum NeutronKernelKind" in content
    assert "int32_t (*neutronKernels[" in content


def test_neutron_converter_kernel_selection_multiple_partitions(mocker):
    input_shape = (1, 3, 64, 64)
    model = Conv2dReLUMaxPoolModule()

    kernel_selection_spy = mocker.spy(eiq_neutron_sdk, "merge_kernel_selection_files")

    # Run conversion
    _ = to_quantized_edge_program(
        model,
        input_shape,
        dump_kernel_selection_code=True,
        operators_not_to_delegate=["aten_relu_default"],
    ).exported_program()

    # Get the kernel selection files generated for the Neutron partitions.
    kernel_selection_files = kernel_selection_spy.call_args.args[0]
    assert len(kernel_selection_files) == 2  # Expect 2 files for 2 partitions.

    # Check that only 1 final kernel selection file was stored.
    stored_kernel_selection_files = [
        fname for fname in os.listdir(".") if fname.endswith("_kernel_selection.c")
    ]
    assert len(stored_kernel_selection_files) == 1
    assert os.path.isfile(stored_kernel_selection_files[0])

    # Check file contains key symbols.
    with open(stored_kernel_selection_files[0]) as f:
        content = f.read()
    assert "typedef enum NeutronKernelKind" in content
    assert "int32_t (*neutronKernels[" in content
