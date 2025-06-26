# Copyright 2024 NXP
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
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import Conv2dModule
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class Conv2dPermuteCopyModule(torch.nn.Module):
    def __init__(self, new_dims: tuple[int, ...]):
        super().__init__()
        self.new_dims = new_dims
        self.conv = Conv2dModule()

    def forward(self, x):
        x = self.conv(x)
        return torch.permute(x, self.new_dims)


def test_permute_copy_quant_conversion__with_bias(mocker):
    input_shape = (1, 4, 8, 8)
    new_dims = (0, 2, 3, 1)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(Conv2dPermuteCopyModule(new_dims), input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        edge_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )
