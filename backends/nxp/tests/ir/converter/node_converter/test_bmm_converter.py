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
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.neutron_operator_support import (
    transposition_is_supported_on_neutron,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import BatchMatMulConvModel, BatchMatMulModel
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
Bmm = exir_ops.edge.aten.bmm.default


@pytest.mark.parametrize(
    "input_shape_x1, input_shape_x2",
    [
        pytest.param((1, 24, 16), (1, 16, 24), id="3D, one batch."),
        pytest.param((3, 8, 24), (3, 24, 8), id="3D, more batches."),
        pytest.param((2, 24, 16), (2, 16, 8), id="3D, more batches, x1_C != x2_W"),
    ],
)
def test_convert_bmm__supported(mocker, input_shape_x1, input_shape_x2, use_qat):
    model = BatchMatMulModel()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model, [input_shape_x1, input_shape_x2], use_qat=use_qat
    ).exported_program()

    # Make sure the `bmm` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Bmm])

    # Verify correct behavior of the converted NeutronIR model.
    intermediate_ep = converter_spy.call_args.args[1]
    neutron_ir_model, *_ = converter_spy.spy_return

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
        pytest.param(
            (1, 8, 7), (1, 7, 16), id="3D, x1_W (and x2_C) not divisible by NUM_MACS."
        ),
        pytest.param((1, 7, 16), (1, 16, 8), id="3D, x1_C not divisible by NUM_MACS."),
        pytest.param((1, 8, 16), (1, 16, 7), id="3D, x2_W not divisible by NUM_MACS."),
    ],
)
def test_convert_bmm__unsupported_shape(input_shape_x1, input_shape_x2, use_qat):
    model = BatchMatMulModel()

    delegated_ep = to_quantized_edge_program(
        model, [input_shape_x1, input_shape_x2], use_qat=use_qat
    ).exported_program()

    # Make sure the `bmm` was NOT delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [Bmm])


def test_convert_bmm__unsupported_dim_order(mocker, use_qat):
    n1 = n2 = 5
    w1 = c2 = 16
    c1 = 8
    w2 = 24

    x_input_shape = (n1, c1, w1)
    y_input_shape = (n2, c2, w2)

    model = BatchMatMulConvModel(in_channels=c1, out_channels=c1)

    delegated_ep = to_quantized_edge_program(
        model,
        [x_input_shape, y_input_shape],
        use_neutron_for_format_conversion=False,
        use_qat=use_qat,
    ).exported_program()

    # Make sure the `bmm` was NOT delegated.
    # For `bmm` to work in channels-first order, support for 3D `transpose` is needed,
    # which is not implemented in NXP Executorch backend yet.
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [Bmm])


def test_convert_bmm__channels_first(mocker, use_qat):
    # These must match:
    # - `n1 = n2`
    # - `w1 = c2`
    # Otherwise it violates `bmm` constraints per mathematical definition.
    n1 = n2 = 5
    w1 = c2 = 16

    # `c1`, `w1`, `c2`, `w2` also need to be divisible by `num_macs`.
    c1 = 8
    w2 = 24

    x_input_shape = (n1, c1, w1)
    y_input_shape = (n2, c2, w2)

    # Channels-last shape of the output before the newly-inserted `transpose`
    # converts it to channels-first
    output_shape = (n1, w2, c1)

    perm = translator.create_channels_first_to_channels_last_permutation(
        len(output_shape), return_list=True
    )
    transp_not_supported = not transposition_is_supported_on_neutron(
        output_shape, perm, neutron_target_spec
    )
    if transp_not_supported:
        pytest.skip("3D dim order swap not implemented.")

    model = BatchMatMulConvModel(in_channels=c1, out_channels=c1)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model,
        [x_input_shape, y_input_shape],
        use_neutron_for_format_conversion=False,
        use_qat=use_qat,
    ).exported_program()

    # Make sure the `bmm` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Bmm])

    # Verify correct behavior of the converted NeutronIR model.
    neutron_ir_model = converter_spy.spy_return[0]
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
    input_data = {
        0: input_data_2,
        1: input_data_1,
    }
    convert_run_compare(
        bmm_intermediate_ep,
        tfl_model=neutron_ir_model,
        input_data=input_data,
        atol=1,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )
