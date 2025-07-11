import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.hardtanh_converter import (
    HardTanhConverter,
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


class Relu6ConvBlock(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3, inplace: bool = False):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels, out_channels=64, kernel_size=(4, 4)
            ),
            torch.nn.ReLU6(inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class CustomHardTanhBlock(torch.nn.Module):
    def __init__(
        self,
        conv_in_channels: int = 3,
        min_act_val: float = -1.0,
        max_act_val: float = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels, out_channels=64, kernel_size=(4, 4)
            ),
            torch.nn.Hardtanh(
                min_val=min_act_val, max_val=max_act_val, inplace=inplace
            ),
        )

    def forward(self, x):
        return self.block(x)


@pytest.mark.parametrize("input_shape", [(1, 3, 128, 128), (1, 3, 256, 256)])
@pytest.mark.parametrize("inplace", [True, False])
def test_relu6_quant(mocker, input_shape: tuple[int], inplace: bool):
    model = Relu6ConvBlock(conv_in_channels=input_shape[1], inplace=inplace)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    ops = [exir_ops.edge.aten.hardtanh.default, exir_ops.edge.aten.hardtanh_.default]
    assert not graph_contains_any_of_ops(graph=quantized_program.graph, ops=ops)

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )


@pytest.mark.parametrize("input_shape", [(1, 3, 128, 128), (1, 3, 256, 256)])
@pytest.mark.parametrize(
    "activation_range", list(HardTanhConverter.supported_modes_map.keys())
)
@pytest.mark.parametrize("inplace", [True, False])
def test_custom_hardtanh_quant(
    mocker, input_shape: tuple[int], activation_range: tuple[int, int], inplace: bool
):
    min_val, max_val = activation_range
    model = CustomHardTanhBlock(
        conv_in_channels=input_shape[1],
        min_act_val=min_val,
        max_act_val=max_val,
        inplace=inplace,
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    ops = [exir_ops.edge.aten.hardtanh.default, exir_ops.edge.aten.hardtanh_.default]
    assert not graph_contains_any_of_ops(graph=quantized_program.graph, ops=ops)

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )
