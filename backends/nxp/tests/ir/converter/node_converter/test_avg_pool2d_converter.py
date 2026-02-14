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
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_edge_program,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import AvgPool2dConvModule, AvgPool2dModule
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.exir.dialects._ops import ops as exir_ops

# noinspection PyProtectedMember
AvgPool2D = exir_ops.edge.aten.avg_pool2d.default
ExecutorchDelegateCall = torch._higher_order_ops.executorch_call_delegate
Squeeze = exir_ops.edge.aten.squeeze.default
SqueezeDim = exir_ops.edge.aten.squeeze.dim
SqueezeDims = exir_ops.edge.aten.squeeze.dims
Unsqueeze = exir_ops.edge.aten.unsqueeze.default
ViewCopy = exir_ops.edge.aten.view_copy.default


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, padding, count_include_pad",
    [
        pytest.param(
            (1, 4, 8, 8),
            (0, 0),
            True,
            id="No padding, include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (0, 0),
            False,
            id="No padding, don't include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (1, 1),
            True,
            id="Padding, keep the same output tensor size as input, include "
            "padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (1, 0),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
        pytest.param(
            (1, 4, 9, 9),
            (1, 0),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
        pytest.param(
            (1, 4, 7, 7),
            (0, 1),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
    ],
)
def test_avg_pool_2d_conversion(input_shape, padding, count_include_pad):
    model = AvgPool2dModule(padding=padding, count_include_pad=count_include_pad)
    edge_program = to_edge_program(model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        conversion_config=ConversionConfig(
            {"use_neutron_for_format_conversion": False}
        ),
    )


@pytest.mark.parametrize(
    "input_shape, padding, count_include_pad",
    [
        pytest.param(
            (1, 4, 16, 16),
            (0, 0),
            True,
            id="No padding, include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (0, 0),
            False,
            id="No padding, don't include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (1, 1),
            True,
            id="Keep the same output tensor size as input, include padding "
            "to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (1, 0),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
        pytest.param(
            (1, 4, 11, 11),
            (0, 1),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
        pytest.param(
            (1, 4, 11, 11),
            (1, 0),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
    ],
)
def test_avg_pool_2d_quant_conversion(
    mocker, input_shape, padding, count_include_pad, use_qat
):
    model = AvgPool2dConvModule(padding=padding, count_include_pad=count_include_pad)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
    )


def test_avg_pool_2d_quant_conversion__padded(mocker, use_qat):
    input_shape = (1, 8, 8, 8)
    model = AvgPool2dModule(True, 1)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

    # Run conversion
    _ = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    )

    # Capture the converter operators.
    ops = ops_spy.spy_return.sub_graphs[0].operators.vector

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
    )

    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.PADV2
    assert ops[1].builtin_options.operator_type == BuiltinOperator.AVERAGE_POOL_2D

    # Make sure the padding used the `zero-point`.
    pad_value = ops[0].tmp_inputs[2].tmp_buffer.data.item()
    assert (
        pad_value == ops[0].tmp_inputs[0].quantization.zero_point[0]
    )  # `Pad` input zp.
    assert (
        pad_value == ops[0].tmp_outputs[0].quantization.zero_point[0]
    )  # `Pad` output zp.
    assert (
        pad_value == ops[1].tmp_inputs[0].quantization.zero_point[0]
    )  # `AvgPool` input zp.


class AvgPool1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool1d(
            kernel_size=3,
        )

    def forward(self, x):
        return self.avg_pool(x)


def test_from_avg_pool_1d(mocker):
    model = AvgPool1DModule()
    input_shape = (
        1,
        3,
        12,
    )  # Don't use multiples of `num_macs` so the `view_copy` nodes will NOT be deleagted.
    extended_shape = (1, 3, 1, 12)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `avg_pool` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [AvgPool2D])

    # Make sure both `view_copy` nodes were added, and there is no `squeeze` or `unsqueeze`.
    assert len([n for n in delegated_ep.graph.nodes if n.target == ViewCopy]) == 2
    assert not graph_contains_any_of_ops(
        delegated_ep.graph, [Unsqueeze, Squeeze, SqueezeDim, SqueezeDims]
    )

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data = (
        np.random.random(extended_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `avg_pool`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [AvgPool2D])

    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )
