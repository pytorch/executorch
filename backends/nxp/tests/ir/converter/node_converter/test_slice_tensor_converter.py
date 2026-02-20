# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import torch
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_converter_flavor,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)

from executorch.backends.nxp.tests.models import (
    SliceTensorConvModule,
    SliceTensorModule,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param((24, 32), (0, 1), (0, 16), (24, 32), id="2D, no transpose"),
        pytest.param(
            (24, 32, 64), (0, 1, 2), (0, 0, 8), (24, 32, 64), id="3D, no transpose"
        ),
        pytest.param(
            (24, 32, 64, 48),
            (0, 1, 2, 3),
            (0, 0, 0, 8),
            (24, 32, 64, 48),
            id="4D, no transpose",
        ),
        pytest.param(
            (24, 32),
            (0, 1),
            (8, 0),
            (24, 32),
            id="2D, one transpose",
            marks=pytest.mark.xfail(reason="EIEX-649", strict=True),
        ),
        pytest.param(
            (24, 32, 64),
            (0, 1, 2),
            (0, 8, 0),
            (24, 32, 64),
            id="3D, one transpose",
            marks=pytest.mark.xfail(reason="EIEX-649", strict=True),
        ),
        pytest.param(
            (24, 32, 64, 48),
            (0, 1, 2, 3),
            (0, 0, 8, 0),
            (24, 32, 64, 48),
            id="4D, one transpose",
            marks=pytest.mark.xfail(reason="EIEX-649", strict=True),
        ),
        pytest.param(
            (24, 32, 64),
            (0, 1, 2),
            (8, 8, 0),
            (24, 32, 64),
            id="3D, two transposes",
            marks=pytest.mark.xfail(reason="EIEX-649", strict=True),
        ),
        # bug in neutron-converter will not properly convert models in these test cases
        # pytest.param((24, 32, 64, 48), (0, 1, 2, 3), (16, 0, 8, 0), (24, 32, 64, 48), id="4D, two transposes"),
        # pytest.param((24, 32, 64, 48), (0, 1, 2, 3), (16, 0, 8, 0), (24, 24, 56, 48), id="4D, three transposes"),
        pytest.param(
            (24, 32),
            (0, 1),
            (0, 13),
            (24, 32),
            id="2D, start arg not divisible by num_macs",
        ),
        pytest.param(
            (24, 32),
            (0, 1),
            (0, 0),
            (24, 31),
            id="2D, end arg not divisible by num_macs",
        ),
        pytest.param((24, 32), (1, 0), (16, 0), (32, 24), id="2D, mixed dim args"),
        pytest.param((24, 32), (0, -1), (0, 16), (24, 32), id="2D, negative dim arg"),
    ],
)
def test_slice_tensor_quant_conversion(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    if neutron_converter_flavor == "SDK_25_09":
        pytest.skip("Neutron Software must be version 2.2.1 or higher.")

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, x_input_shape).exported_program()
    edge_nodes = list(edge_program.graph.nodes)

    # Check if slices were delegated
    assert not any("slice" in n.name for n in edge_nodes)

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data = {0: input_data}

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tfl_model=tflite_flatbuffers_model,
    )


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (1, 4, 34, 50),
            (0, 1, 2, 3),
            (0, 0, 8, 0),
            (1, 8, 32, 32),
            id="4D, handle channel order swap",
            marks=pytest.mark.xfail(reason="EIEX-649", strict=True),
        )
    ],
)
def test_slice_tensor_w_conv_quant_conversion(
    mocker, x_input_shape, dims, starts, ends
):
    if neutron_converter_flavor == "SDK_25_09":
        pytest.skip("Neutron Software must be version 2.2.1 or higher.")

    model = SliceTensorConvModule(dims=dims, starts=starts, ends=ends)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, x_input_shape, use_neutron_for_format_conversion=False
    ).exported_program()
    edge_nodes = list(edge_program.graph.nodes)

    # Check if slices were delegated
    assert not any("slice" in n.name for n in edge_nodes)

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data = {0: input_data}

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (24, 32), (0, 1), (0, 16), (24, 8), id="2D, start is higher than end"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 16), (24, 16), id="2D, start is equal to end"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 32), (24, 32), id="2D, start is equal to size"
        ),
        pytest.param(
            (24, 32), (0, 1), (0, 0), (24, -5), id="2D, clipped end equal to zero"
        ),
        pytest.param(
            (24, 32), (0, 1), (64, 0), (24, 32), id="2D, clipped start equal to size"
        ),
    ],
)
def test_invalid_slice(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, x_input_shape).exported_program()

    # Capture generated model, should be None because the model is invalid
    assert converter_spy.spy_return is None


@pytest.mark.parametrize(
    "x_input_shape, dims, starts, ends",
    [
        pytest.param(
            (24, 31),
            (0, 1),
            (0, 0),
            (24, 16),
            id="2D, input shape not divisible by num_macs",
        ),
        pytest.param(
            (24, 26, 64),
            (0, 1, 2),
            (0, 4, 0),
            (24, 26, 64),
            id="3D, input shape not divisible by num_macs",
        ),
    ],
)
def test_slice_not_delegated(mocker, x_input_shape, dims, starts, ends):
    model = SliceTensorModule(
        dims=dims,
        starts=starts,
        ends=ends,
    )

    edge_program = to_quantized_edge_program(model, x_input_shape).exported_program()
    nodes = list(edge_program.graph.nodes)

    num_slice_ops = 0
    for i in range(len(x_input_shape)):
        if starts[i] != 0 or ends[i] != x_input_shape[i]:
            num_slice_ops += 1

    for i in range(0, num_slice_ops):
        slice_idx = (i + 1) * 3
        assert nodes[slice_idx].target == exir_ops.edge.aten.slice_copy.Tensor
