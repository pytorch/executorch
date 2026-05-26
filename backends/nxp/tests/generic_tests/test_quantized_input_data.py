# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.nxp.tests.nsys_testing as nsys_testing
import torch

from executorch.backends.nxp.tests.executorch_pipeline import ModelInputSpec
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import AvgPool2dModule, MulTensorModule
from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
    OUTPUTS_DIR,
    ReferenceModel,
)
from executorch.backends.nxp.tests.ops_aliases import AvgPool2D, MulTensor


def test__single_quantized_inputs(mocker):
    input_spec = ModelInputSpec((2, 4, 6, 7))
    model = AvgPool2dModule(False, 0)
    graph_verifier = DetailedGraphVerifier(
        mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
    )
    output_tensor_spec_spy = mocker.spy(nsys_testing, "_get_program_output_spec")

    lower_run_compare(
        model,
        [input_spec],
        graph_verifier,
        use_new_flow_neutron_c=True,
        remove_quant_io_ops=True,
    )

    assert (
        OUTPUTS_DIR / "test__single_quantized_inputs" / "dataset_quant" / "0000.bin"
    ).exists()

    # Check outputs are in quantized int8 format
    output_tensor_spec = output_tensor_spec_spy.spy_return
    assert output_tensor_spec[0].dtype == torch.int8


def test__single_quantized_inputs_edge_python_reference(mocker):
    input_spec = ModelInputSpec((2, 4, 6, 7))
    model = AvgPool2dModule(False, 0)
    graph_verifier = DetailedGraphVerifier(
        mocker, expected_delegated_ops={AvgPool2D: 1}, expected_non_delegated_ops={}
    )
    output_tensor_spec_spy = mocker.spy(nsys_testing, "_get_program_output_spec")

    lower_run_compare(
        model,
        [input_spec],
        graph_verifier,
        reference_model=ReferenceModel.QUANTIZED_EDGE_PYTHON,
        use_new_flow_neutron_c=True,
        remove_quant_io_ops=True,
    )

    assert (
        OUTPUTS_DIR
        / "test__single_quantized_inputs_edge_python_reference"
        / "dataset_quant"
        / "0000.bin"
    ).exists()

    # Check outputs are in quantized int8 format
    output_tensor_spec = output_tensor_spec_spy.spy_return
    assert output_tensor_spec[0].dtype == torch.int8


def test__multiple_quantized_inputs(mocker):
    x_input_spec = ModelInputSpec((1, 4, 8, 8))
    model = MulTensorModule()
    graph_verifier = DetailedGraphVerifier(
        mocker, expected_delegated_ops={MulTensor: 1}, expected_non_delegated_ops={}
    )
    output_tensor_spec_spy = mocker.spy(nsys_testing, "_get_program_output_spec")

    lower_run_compare(
        model,
        [x_input_spec, x_input_spec],
        graph_verifier,
        use_new_flow_neutron_c=True,
        remove_quant_io_ops=True,
    )

    assert (
        OUTPUTS_DIR
        / "test__multiple_quantized_inputs"
        / "dataset_quant"
        / "0000"
        / "00.bin"
    ).exists()

    # Check outputs are in quantized int8 format
    output_tensor_spec = output_tensor_spec_spy.spy_return
    assert output_tensor_spec[0].dtype == torch.int8


def test__multiple_quantized_inputs_edge_python_reference(mocker):
    x_input_spec = ModelInputSpec((1, 4, 8, 8))
    model = MulTensorModule()
    graph_verifier = DetailedGraphVerifier(
        mocker, expected_delegated_ops={MulTensor: 1}, expected_non_delegated_ops={}
    )
    output_tensor_spec_spy = mocker.spy(nsys_testing, "_get_program_output_spec")

    lower_run_compare(
        model,
        [x_input_spec, x_input_spec],
        graph_verifier,
        reference_model=ReferenceModel.QUANTIZED_EDGE_PYTHON,
        use_new_flow_neutron_c=True,
        remove_quant_io_ops=True,
    )

    assert (
        OUTPUTS_DIR
        / "test__multiple_quantized_inputs_edge_python_reference"
        / "dataset_quant"
        / "0000"
        / "00.bin"
    ).exists()

    # Check outputs are in quantized int8 format
    output_tensor_spec = output_tensor_spec_spy.spy_return
    assert output_tensor_spec[0].dtype == torch.int8
