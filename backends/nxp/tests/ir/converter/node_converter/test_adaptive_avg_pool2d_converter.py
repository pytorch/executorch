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
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import (
    AdaptiveAvgPool2dConvMeanDimModule,
    AdaptiveAvgPool2dConvModule,
    AdaptiveAvgPool2dModule,
)

from executorch.backends.nxp.tests.nsys_testing import lower_run_compare

from executorch.backends.nxp.tests.ops_aliases import (
    AdaptiveAvgPool2D,
    ExecutorchDelegateCall,
)
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, output_size",
    [
        pytest.param(
            (1, 4, 16, 16), (4, 4), id="Pooling with equal height and width kernel."
        ),
        pytest.param(
            (1, 4, 16, 16), (8, 8), id="Pooling with equal height and width kernel."
        ),
        pytest.param((1, 4, 16, 16), (4, 8), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 16, 22), (4, 11), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 32, 32), (16, 4), id="Pooling with height < width kernel."),
        pytest.param((1, 4, 32, 16), (16, 4), id="Pooling with height < width kernel."),
    ],
)
def test_adaptive_avg_pool_2d_delegated_quant_conversion(
    mocker, input_shape, output_size, use_qat
):
    model = AdaptiveAvgPool2dConvModule(output_size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
    nodes = [str(node) for node in edge_program.graph.nodes]

    # Input size is a multiple of output size, can be converted to AveragePool, node is delegated
    assert "aten__adaptive_avg_pool2d_default" not in nodes

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1,
    )


@pytest.mark.parametrize(
    "input_shape, output_size",
    [
        pytest.param(
            (1, 4, 16, 16), (6, 6), id="Pooling with equal height and width kernel."
        ),
        pytest.param((1, 4, 16, 16), (4, 7), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 16, 22), (4, 10), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 32, 32), (14, 7), id="Pooling with height < width kernel."),
        pytest.param((1, 4, 32, 16), (15, 5), id="Pooling with height < width kernel."),
    ],
)
def test_adaptive_avg_pool_2d_non_delegated_quant_conversion(
    mocker, input_shape, output_size, use_qat
):
    model = AdaptiveAvgPool2dConvModule(output_size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Input size is not a multiple of output size, cannot be converted to AveragePool, node is not delegated
    assert str(nodes[6]) == "aten__adaptive_avg_pool2d_default"

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1,
    )


def test_adaptive_avg_pool_2d_mean_dim_quant_conversion(mocker, use_qat):
    input_shape = (1, 4, 16, 16)
    model = AdaptiveAvgPool2dConvMeanDimModule()

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
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
    )


class TestAdaptiveAvgPool2DNewNeutronFlow:
    @pytest.mark.parametrize(
        "input_shape, output_size",
        [
            pytest.param((1, 3, 16, 16), (8, 8), id="H == W."),
            pytest.param((1, 3, 16, 8), (8, 2), id="H != W."),
            pytest.param(
                (2, 3, 4, 6),
                (2, 3),
                id="H != W, non multiples of num_macs, batch != 1.",
            ),
        ],
    )
    def test__basic_nsys_inference(self, mocker, use_qat, input_shape, output_size):
        model = AdaptiveAvgPool2dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1},
            expected_non_delegated_ops={},
        )

        output_comparator = AllCloseOutputComparator(
            7.84e-3
        )  # Accept small error due to Neutron bug (AIR-14585).

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            use_qat=use_qat,
            use_new_flow_neutron_c=True,
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Known Neutron bad compute issue. Will be fixed in Neutron SW 3.1.2.",
    )
    def test__know_neutron_issue(self, mocker):
        input_shape = (2, 3, 10, 15)
        output_size = (5, 5)
        model = AdaptiveAvgPool2dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1},
            expected_non_delegated_ops={},
        )

        # Use high tolerance so we notice when the issue is fixed.
        output_comparator = AllCloseOutputComparator(7.8e-3)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            use_new_flow_neutron_c=True,
        )

    def test__kernel_size_and_stride_limit(self, mocker):
        input_shape = (1, 3, 4, 4096)  # input_size = (1, 4096)
        output_size = (
            2,
            1,
        )  # If we reduced both dims to 1, ExecuTorch would replace the op with mean.
        # stride = input_size // output_size = 4096 / 1 = 4096
        # kernel_size = input_size - (output_size - 1) * stride = 4096 - 0 * 4096 = 4096

        model = AdaptiveAvgPool2dModule(output_size)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={AdaptiveAvgPool2D: 1},
            expected_non_delegated_ops={},
        )

        output_comparator = AllCloseOutputComparator(
            7.9e-3
        )  # Accept small error due to Neutron bug (AIR-14585).

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            RandomDatasetCreator(low=-1, high=1),
            output_comparator=output_comparator,
            use_new_flow_neutron_c=True,
        )

    def test__kernel_size_and_stride_limit_exceeded(self):
        input_shape = (1, 3, 4, 4097)  # input_size = (1, 4097)
        output_size = (
            2,
            1,
        )  # If we reduced both dims to 1, ExecuTorch would replace the op with mean.
        # stride = input_size // output_size = 4097 / 1 = 4097
        # kernel_size = input_size - (output_size - 1) * stride = 4097 - 0 * 4097 = 4097

        model = AdaptiveAvgPool2dModule(output_size)
        delegated_ep = to_quantized_edge_program(
            model, input_shape, use_new_flow_neutron_c=True
        ).exported_program()

        # Make sure the `adaptive_avg_pool2d` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [AdaptiveAvgPool2D])
