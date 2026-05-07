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
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier

from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
    RandomDatasetCreator,
)
from executorch.backends.nxp.tests.ops_aliases import Abs, Convolution, Relu

from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ConvBlocksWithAbsModule(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=3,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=10,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x).abs()
        return self.block2(x)


class AbsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.abs()


class TestAbsLegacyNeutronFlow:
    def test_conv_abs(
        self, mocker, use_qat, input_shape: tuple[int, ...] = (1, 3, 112, 112)
    ):
        model = ConvBlocksWithAbsModule(conv_in_channels=input_shape[1])

        converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

        quantized_program = to_quantized_edge_program(
            model,
            input_shape,
            use_qat=use_qat,
            use_neutron_for_format_conversion=False,
            use_new_flow_neutron_c=False,
        ).exported_program()

        tflite_flatbuffers_model, io_formats = converter_spy.spy_return
        exported_program: ExportedProgram = converter_spy.call_args.args[1]

        assert not graph_contains_any_of_ops(
            graph=quantized_program.graph, ops=[exir_ops.edge.aten.abs.default]
        )

        input_data = (np.random.random(input_shape) * 50).astype(np.int8)
        convert_run_compare(
            exported_program,
            tfl_model=tflite_flatbuffers_model,
            tflite_input_preprocess=ToChannelLastPreprocess(),
            tflite_output_preprocess=ToChannelFirstPreprocess(),
            input_data=input_data,
            atol=1.0,
        )


class TestAbsNewNeutronFlow:
    @staticmethod
    def _get_dataset_creator():
        # to test `abs` reliably, we need to include negative values
        low = -255.0
        high = 255.0

        dataset = RandomDatasetCreator(low=low, high=high)
        return dataset

    def test__basic_nsys_inference(self, mocker):
        input_shape = (2, 3, 6, 7)
        model = AbsModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Abs: 1}, expected_non_delegated_ops={}
        )

        dataset_creator = self._get_dataset_creator()
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    def test__basic_nsys_inference__big(self, mocker):
        # some operators have delegation requirement that size must be < 4096
        input_shape = (4097, 1)
        model = AbsModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Abs: 1}, expected_non_delegated_ops={}
        )

        dataset_creator = self._get_dataset_creator()
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )

    def test_basic_nsys_inference__with_conv(self, mocker):
        input_shape = (2, 3, 6, 7)
        in_channels = input_shape[1]
        model = ConvBlocksWithAbsModule(conv_in_channels=in_channels)

        # one `relu` ends up in the same delegated partition as `abs`
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Abs: 1, Relu: 1},
            expected_non_delegated_ops={Relu: 1, Convolution: 2},
        )

        dataset_creator = self._get_dataset_creator()
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_new_flow_neutron_c=True,
        )
