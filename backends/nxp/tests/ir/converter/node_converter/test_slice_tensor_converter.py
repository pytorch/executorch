# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import (
    SliceTensorModule,
    AddTensorModule,
    AddTensorOneInputModule,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "slice_params, x_input_shape",
    [
        pytest.param({
            "dim": 0,
            "start": 0,
            "end": 32,
            "step": 8
        }, (32,), id="1D.")
    ],
)
def test_slice_tensor_quant_conversion(mocker, slice_params, x_input_shape):
    model = SliceTensorModule(
        dim=slice_params["dim"],
        start=slice_params["start"],
        end=slice_params["end"],
        step=slice_params["step"]
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, x_input_shape).exported_program()

    # Capture generated model
    x = converter_spy.spy_return
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(np.int8)
    input_data = {0: input_data, 1: input_data}

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )

