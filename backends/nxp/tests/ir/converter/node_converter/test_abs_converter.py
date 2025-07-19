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
    graph_contains_any_of_ops,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ConvBlocksWithAbs(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=3,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=10,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x).abs()
        return self.block2(x)


class Abs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.abs()


def test_conv_abs(mocker, input_shape: tuple[int] = (1, 3, 112, 112)):
    model = ConvBlocksWithAbs(conv_in_channels=input_shape[1])

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.abs.default]
    )

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )
