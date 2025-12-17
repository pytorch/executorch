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
from executorch.backends.nxp.tests.models import Conv2dWithActivation
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("input_shape", [(1, 3, 128, 128)])
@pytest.mark.parametrize("inplace", [True, False])
def test_relu6_quant(mocker, input_shape: tuple[int], inplace: bool, use_qat: bool):
    # The torch.nn.Relu6 inherits from torch.nn.Hardtanh, and hence represented as HardTanh in ATen.
    # Testing the hardtanh originated from torch.nn.Relu6 op.
    model = Conv2dWithActivation(
        activation=torch.nn.ReLU6(inplace=inplace), in_channels=input_shape[1]
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()

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
        atol=2.0,
    )


@pytest.mark.parametrize("input_shape", [(1, 3, 16, 16), (1, 3, 32, 32)])
@pytest.mark.parametrize(
    "activation_range", list(HardTanhConverter.supported_modes_map.keys())
)
@pytest.mark.parametrize("inplace", [True, False])
def test_custom_hardtanh_quant(
    mocker,
    input_shape: tuple[int],
    activation_range: tuple[int, int],
    inplace: bool,
    use_qat: bool,
):
    # TODO(13063): This test suffers from non-ideal testing random quantization, because we always use range <0,1>.
    #  We should update (decrease atol) when the Conv/Linear + Activation fuse at quantization is in place.
    min_val, max_val = activation_range
    model = Conv2dWithActivation(
        activation=torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace),
        in_channels=input_shape[1],
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()

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
        atol=2.0,
    )
