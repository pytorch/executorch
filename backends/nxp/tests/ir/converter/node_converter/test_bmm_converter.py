# Copyright 2026 NXP
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
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.models import BatchMatMulConvModel, BatchMatMulModel
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)


# noinspection PyProtectedMember
ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
Bmm = exir_ops.edge.aten.bmm.default


@pytest.mark.parametrize(
    "input_shape_x1, input_shape_x2",
    [
        pytest.param((1, 8, 16), (1, 16, 24), id="3D, one batch."),
        pytest.param((4, 8, 16), (4, 16, 24), id="3D, more batches."),
    ],
)
def test_convert_bmm__supported(mocker, input_shape_x1, input_shape_x2):
    model = BatchMatMulModel()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model,
        [input_shape_x1, input_shape_x2],
    ).exported_program()

    # Make sure the `bmm` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Bmm])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, _ = converter_spy.spy_return

    input_data_1 = (
        np.random.random(input_shape_x1).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)
    input_data_2 = (
        np.random.random(input_shape_x2).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `bmm`.
    assert graph_contains_any_of_ops(intermediate_ep.graph, [Bmm])

    # Verify that the delegated `bmm` node produces correct results
    # The delegated `bmm` runs with a numerical tolerance of atol = 1
    convert_run_compare(
        intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data={
            0: input_data_1,
            1: input_data_2,
        },
        atol=1,
    )


@pytest.mark.parametrize(
    "input_shape_x1, input_shape_x2",
    [
        pytest.param((1, 7, 16), (1, 16, 24), id="3D, x1_W not divisible by NUM_MACS."),
        pytest.param(
            (1, 8, 7), (1, 7, 24), id="3D, x1_C (and x2_W) not divisible by NUM_MACS."
        ),
        pytest.param((1, 8, 16), (1, 16, 7), id="3D, x2_C not divisible by NUM_MACS."),
    ],
)
def test_convert_bmm__unsupported(input_shape_x1, input_shape_x2):
    model = BatchMatMulModel()

    delegated_ep = to_quantized_edge_program(
        model,
        [input_shape_x1, input_shape_x2],
    ).exported_program()

    # Make sure the `bmm` was NOT delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [Bmm])


def test_convert_bmm__conv_quant(mocker, use_qat):
    # These must match:
    # - `h1 = h2`
    # - `w1 = c2`
    # Otherwise it violates `bmm` constraints.
    h1 = h2 = 5
    w1 = c2 = 16
    
    # Must be divisible by `num_macs`, otherwise can be arbitrary.
    c1 = 8
    w2 = 24

    x_input_shape = (h1, c1, w1)
    y_input_shape = (h2, c2, w2)
    model = BatchMatMulConvModel(in_channels=c1, out_channels=c1)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model,
        [x_input_shape, y_input_shape],
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure the `bmm` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Bmm])

    # Verify correct behavior of the converted NeutronIR model.
    bmm_intermediate_ep = converter_spy.call_args.args[1]

    input_data_1 = (
        np.random.random(x_input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)
    input_data_2 = (
        np.random.random(y_input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `bmm`.
    assert graph_contains_any_of_ops(bmm_intermediate_ep.graph, [Bmm])

    # Verify that the delegated `bmm` node produces correct results
    # The delegated `bmm` runs with a numerical tolerance of atol = 1.
    # The `intermediate_ep` has input positions swapped.
    convert_run_compare(
        bmm_intermediate_ep,
        input_data={
            0: input_data_2,
            1: input_data_1,
        },
        atol=1,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )
