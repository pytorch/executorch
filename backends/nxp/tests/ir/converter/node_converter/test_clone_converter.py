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
from torch import nn
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class SingleConvBlockWithDropout(torch.nn.Module):
    def __init__(
        self, conv_in_channels: int = 3, perform_inplace_dropout: bool = False
    ):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels, out_channels=64, kernel_size=(4, 4)
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(inplace=perform_inplace_dropout),
        )

    def forward(self, x):
        return self.block(x)


class KWSFinalBlock(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        pool_size = (25, 5)
        self.block = torch.nn.Sequential(
            self.conv_sep_dw(inp=input_shape[1], oup=64),
            nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),
        )

    def conv_sep_dw(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=inp, out_channels=inp, kernel_size=3, padding=1, groups=inp
            ),
            nn.BatchNorm2d(num_features=inp, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=oup, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


@pytest.mark.parametrize("inplace_dropout", [False, True])
@pytest.mark.parametrize("input_shape", [(1, 3, 128, 128), (1, 3, 256, 256)])
def test_conv_dropout_quant(mocker, inplace_dropout: bool, input_shape: tuple[int]):
    model = SingleConvBlockWithDropout(
        conv_in_channels=input_shape[1], perform_inplace_dropout=inplace_dropout
    ).eval()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.clone.default]
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


@pytest.mark.parametrize("inplace_dropout", [False, True])
def test_clone_pool_view_copy_quant(
    mocker, inplace_dropout: bool, input_shape: tuple[int] = (1, 64, 25, 5)
):
    model = KWSFinalBlock(input_shape).eval()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    assert not graph_contains_any_of_ops(
        graph=quantized_program.graph, ops=[exir_ops.edge.aten.clone.default]
    )

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        input_data=input_data,
        atol=1.0,
    )
