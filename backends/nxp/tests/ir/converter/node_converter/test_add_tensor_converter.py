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
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import (
    AddTensorConvModule,
    AddTensorModule,
    AddTensorOneInputModule,
)
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    AddTensor,
    Convolution,
    ExecutorchDelegateCall,
)
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((4,), id="1D."),
        pytest.param((6, 6), id="2D."),
        pytest.param((1, 4, 8), id="3D."),
        pytest.param((1, 4, 8, 8), id="4D."),
    ],
)
def test_add_tensor_quant_conversion(mocker, input_shape, use_qat):
    model = AddTensorModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, [input_shape, input_shape], use_qat=use_qat)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    input_data = {0: input_data, 1: input_data}

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((4,), id="1D."),
        pytest.param((6, 6), id="2D."),
        pytest.param((1, 4, 8), id="3D."),
        pytest.param((1, 4, 8, 8), id="4D."),
    ],
)
def test_add_tensor_one_input_quant_conversion(mocker, input_shape, use_qat):
    model = AddTensorOneInputModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape, use_qat=use_qat)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "x_input_shape",
    [
        pytest.param((1, 4, 8, 8), id="4D."),
        pytest.param((1, 4, 5, 5), id="4D, product of dims is not a multiple of 8."),
    ],
)
def test_add_tensor_w_conv_quant_conversion(mocker, x_input_shape, use_qat):
    model = AddTensorConvModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    n, c, h, w = x_input_shape
    y_input_shape = (n, 8, h, w)

    # Run conversion
    _ = to_quantized_edge_program(
        model,
        [x_input_shape, y_input_shape],
        use_qat=use_qat,
        use_neutron_for_format_conversion=False,
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data_1 = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data_2 = (np.random.random(y_input_shape).astype(np.float32) * 50).astype(
        np.int8
    )
    input_data = {0: input_data_1, 1: input_data_2}

    convert_run_compare(
        exported_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


@pytest.mark.parametrize(
    "x_input_shape, y_input_shape",
    [
        pytest.param((1, 4, 7), (4, 7), id="3D -> 2D."),
        pytest.param((1, 4, 8), (1, 4, 4, 8), id="3D -> 4D."),
        pytest.param((1, 1, 4, 4, 8), (1, 4, 4, 8), id="5D -> 4D."),
        pytest.param((4,), (4, 4), id="1D -> 2D."),
        pytest.param((4,), (4, 4, 4), id="1D -> 3D."),
        pytest.param((6, 6), (1, 8, 6, 6), id="2D -> 4D."),
        pytest.param((6, 6), (6,), id="2D -> 1D."),
    ],
)
def test_add_tensor_broadcasting_unsupported_quant_conversion(
    x_input_shape, y_input_shape, use_qat
):
    model = AddTensorModule()

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, [x_input_shape, y_input_shape], use_qat=use_qat
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Broadcast is not supported, node is not converted
    assert nodes[6].target == AddTensor  # Add Tensor is not delegated.

    # Capture converted program
    # exported_program: ExportedProgram = converter_spy.call_args.args[1]
    #
    # x_input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(np.int8)
    # y_input_data = (np.random.random(y_input_shape).astype(np.float32) * 50).astype(np.int8)
    # input_data = {0: x_input_data, 1: y_input_data}
    #
    # convert_run_compare(exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data)


class TestAddTensorNewNeutronFlow:
    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 5), id="2D."),
            pytest.param((1, 4, 7), id="3D."),
            pytest.param((2, 4, 3, 15), id="4D."),
            pytest.param(
                (6, 82),
                id="2D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (1, 68, 7),
                id="3D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
            pytest.param(
                (1, 4, 9, 11, 4),
                id="5D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__basic_nsys_inference(self, x_input_shape, mocker):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param((1,), id="1D."),
            pytest.param((6, 5), id="2D."),
            pytest.param((1, 4, 7), id="3D."),
            pytest.param((2, 4, 3, 15), id="4D."),
            pytest.param(
                (1, 4, 9, 11, 4),
                id="5D.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__basic_nsys_inference_qat(self, x_input_shape, mocker):
        x_input_spec = ModelInputSpec(x_input_shape)
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, x_input_spec],
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
            use_qat=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 6)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((5, 3, 4)), ModelInputSpec((1, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((4,)), ModelInputSpec((4, 4))], id="2 inputs 1D + 2D."
            ),
            pytest.param(
                [ModelInputSpec((69, 73)), ModelInputSpec((1, 73))],
                id="2 inputs 2D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__broadcast(self, input_spec, mocker):
        model = AddTensorModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={AddTensor: 1}, expected_non_delegated_ops={}
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((4, 1)), ModelInputSpec((1, 6))], id="2 inputs 2D."
            ),
            pytest.param(
                [ModelInputSpec((1, 3, 4)), ModelInputSpec((5, 3, 1))],
                id="2 inputs 3D.",
            ),
            pytest.param(
                [ModelInputSpec((6, 4)), ModelInputSpec((6, 6, 1))],
                id="2 inputs 2D + 3D.",
            ),
        ],
    )
    def test__broadcast_unsupported(self, input_spec):
        # Broadcast where at least one of the inputs is not equal to output is not supported
        model = AddTensorModule()

        delegated_ep = to_quantized_edge_program(
            model, input_spec, use_new_flow_neutron_c=True
        ).exported_program()

        # Make sure the `add.Tensor` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AddTensor])

    @pytest.mark.parametrize(
        "x_input_shape",
        [
            pytest.param(
                (1, 4, 5, 5), id="4D, product of dims is not a multiple of 8."
            ),
        ],
    )
    def test__w_conv(self, x_input_shape, mocker):
        model = AddTensorConvModule()

        n, c, h, w = x_input_shape
        y_input_spec = ModelInputSpec((n, 8, h, w))
        x_input_spec = ModelInputSpec(x_input_shape)

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AddTensor: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            [x_input_spec, y_input_spec],
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((1, 4, 5, 5)), ModelInputSpec((1, 8, 5, 1))],
                id="2 inputs 4D + 4D.",
            ),
            pytest.param(
                [ModelInputSpec((1, 4, 5, 67)), ModelInputSpec((1, 8, 5, 1))],
                id="2 inputs 4D + 4D incorrect.",
                marks=pytest.mark.xfail(reason="AIR-14602: incorrect results"),
            ),
        ],
    )
    def test__w_conv_broadcast(self, input_spec, mocker):
        model = AddTensorConvModule()

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AddTensor: 1, Convolution: 1},
            expected_non_delegated_ops={},
        )
        dataset_creator = RandomDatasetCreator(low=-1.0, high=1.0)

        lower_run_compare(
            model,
            input_spec,
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    @pytest.mark.parametrize(
        "input_spec",
        [
            pytest.param(
                [ModelInputSpec((1, 4, 5, 5)), ModelInputSpec((1, 5))],
                id="2 inputs 4D + 2D.",
            ),
            pytest.param(
                [ModelInputSpec((1, 4, 4, 10)), ModelInputSpec((1, 4, 1))],
                id="2 inputs 4D + 3D.",
            ),
        ],
    )
    def test__w_conv_unsupported(self, input_spec):
        model = AddTensorConvModule()

        delegated_ep = to_quantized_edge_program(
            model, input_spec, use_new_flow_neutron_c=True
        ).exported_program()

        # Make sure the `add.Tensor` was NOT delegated.
        assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
        assert graph_contains_any_of_ops(delegated_ep.graph, [AddTensor])
