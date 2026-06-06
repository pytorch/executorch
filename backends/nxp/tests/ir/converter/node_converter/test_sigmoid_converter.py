# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.model_output_comparator import (
    AllCloseOutputComparator,
)
from executorch.backends.nxp.tests.models import ConvWithSigmoid
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import DequantizePerTensor, Sigmoid
from torch import nn
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_conv_sigmoid(mocker, use_qat, input_shape: tuple[int] = (1, 3, 112, 112)):
    model = ConvWithSigmoid(conv_in_channels=input_shape[1])

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((10,), id="Scalar"),
        pytest.param((10, 25), id="1D"),
        pytest.param((10, 25, 25), id="2D"),
        pytest.param((10, 3, 25, 25), id="3D"),
        pytest.param((10, 3, 25, 25, 25), id="4D"),
    ],
)
def test_sigmoid_only(mocker, use_qat, input_shape):
    model = nn.Sigmoid()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    to_quantized_edge_program(model, input_shape, use_qat=use_qat).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


class TestSigmoidNewNeutronFlow:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(self, model, input_shape, mocker, use_qat=False, atol=None):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Sigmoid: 1},
            expected_non_delegated_ops={},
        )

        # Create a RandomDatasetCreator that covers also negative numbers to properly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        kwargs = {"atol": atol} if atol is not None else {}
        output_comparator = AllCloseOutputComparator(**kwargs)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            output_comparator,
            use_qat=use_qat,
            use_new_flow_neutron_c=True,  # Use the new flow.
        )

    def test__basic_nsys_inference__qat(self, mocker, use_qat):
        input_shape = (23,)
        model = nn.Sigmoid()
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "input_shape",
        [
            (2,),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
        ],
        ids=lambda shape: f"{len(shape)}D",
    )
    def test__input_shapes(self, mocker, input_shape):
        model = nn.Sigmoid()

        output_scale = 1.0 / 256.0
        lowering_spy = mocker.spy(NeutronPartitioner, "partition")
        self.assert_delegated(
            model, input_shape, mocker, atol=output_scale
        )  # Allow single bit error.

        # Verify that the `atol` is indeed equal to the output scale.
        # In the near future, we would like to add support for testing with int8 IO, where this check will be trivial.
        nodes = list(lowering_spy.spy_return.tagged_exported_program.graph.nodes)
        assert nodes[-2].target == DequantizePerTensor
        assert nodes[-2].args[1] == output_scale
