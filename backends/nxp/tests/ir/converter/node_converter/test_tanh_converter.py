# Copyright 2025-2026 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.nxp_backend import EdgeProgramToIRConverter
from executorch.backends.nxp.tests.dataset_creator import RandomDatasetCreator
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dWithActivation
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Convolution, Tanh, Tanh_
from parameterized import parameterized
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


class TestTanhConverter(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(23)

    @parameterized.expand(
        input=[
            ("QAT inplace", True, True),
            ("PTQ inplace", True, False),
            ("QAT not-inplace", False, True),
            ("PTQ not-inplace", False, False),
        ]
    )
    def test_conv_tanh(
        self,
        _: str,
        inplace: bool,
        use_qat: bool,
        input_shape: tuple[int] = (1, 3, 112, 112),
    ):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            if inplace:
                model = Conv2dWithActivation(
                    activation=torch.tanh_, in_channels=input_shape[1]
                )
            else:
                model = Conv2dWithActivation(
                    activation=torch.tanh, in_channels=input_shape[1]
                )

            quantized_program = to_quantized_edge_program(
                model,
                input_shape,
                use_qat=use_qat,
                use_neutron_for_format_conversion=False,
            ).exported_program()
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            lowered_module_graph = (
                quantized_program.graph_module.lowered_module_0.original_module.graph
            )
            tanh_ops = [Tanh, Tanh_]
            assert graph_contains_any_of_ops(graph=lowered_module_graph, ops=tanh_ops)

            input_data = (np.random.random(input_shape) * 50).astype(np.int8)
            convert_run_compare(
                exported_program,
                tfl_model=tflite_flatbuffers_model,
                tflite_input_preprocess=ToChannelLastPreprocess(),
                tflite_output_preprocess=ToChannelFirstPreprocess(),
                input_data=input_data,
                atol=2.0,
            )


class TanhModule(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return torch.tanh_(x)
        else:
            return torch.tanh(x)


class TestTanhNewNeutronFlow:

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self,
        model,
        input_shape,
        mocker,
        use_qat=False,
        expected_delegated_ops=None,
    ):
        if expected_delegated_ops is None:
            expected_delegated_ops = {Tanh: 1}

        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )

        # Cover also negative values to thoroughly test the operator.
        dataset_creator = RandomDatasetCreator(low=-2, high=2)

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator,
            use_qat=use_qat,
            use_new_flow_neutron_c=True,  # Use the new flow.
        )

    @pytest.fixture(params=[True, False], ids=lambda inplace: f"inplace = {inplace}")
    def inplace(self, request):
        return request.param

    def test__qat__inplace(self, mocker, use_qat, inplace):
        shape = (23,)
        model = TanhModule(inplace)
        self.assert_delegated(model, shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "shape",
        [
            (16,),
            (3, 5),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 2, 3, 2),
        ],
        ids=lambda shape: f"{len(shape)}D",
    )
    def test__shapes(self, mocker, shape):
        model = TanhModule()
        self.assert_delegated(model, shape, mocker)

    def test__with_convolution(self, mocker):
        input_shape = (1, 3, 12, 16)
        channels = input_shape[1]
        model = Conv2dWithActivation(
            activation=torch.tanh, in_channels=channels, out_channels=channels
        )
        self.assert_delegated(
            model, input_shape, mocker, expected_delegated_ops={Tanh: 1, Convolution: 1}
        )
