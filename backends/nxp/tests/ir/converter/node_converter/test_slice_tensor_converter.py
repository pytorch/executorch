# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import torch

from backends.nxp.tests.executors import ToNCHWPreprocess, ToNHWCPreprocess
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
    SliceTensorConvModule,
    AddTensorModule,
    AddTensorOneInputModule,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param((128, 96, 16), (0, 1, 2), (8, 8, 0), (128, 96, 16), id="4D.")
    ],
)
def test_slice_tensor_quant_conversion(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorConvModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, x_input_shape).exported_program()

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data_1 = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(np.int8)
    input_data = {0: input_data_1}

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tfl_model=tflite_flatbuffers_model
    )

