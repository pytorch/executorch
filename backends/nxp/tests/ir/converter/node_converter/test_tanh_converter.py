# Copyright 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.nxp_backend import EdgeProgramToIRConverter
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import Conv2dWithActivation
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram


class TestTanhConverter(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(23)

    @parameterized.expand(
        input=[
            (
                "inplace",
                True,
            ),
            (
                "not_inplace",
                False,
            ),
        ]
    )
    def test_conv_tanh(
        self, _: str, inplace: bool, input_shape: tuple[int] = (1, 3, 112, 112)
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
                model, input_shape, use_neutron_for_format_conversion=False
            ).exported_program()
            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            lowered_module_graph = (
                quantized_program.graph_module.lowered_module_0.original_module.graph
            )
            tanh_ops = [
                exir_ops.edge.aten.tanh.default,
                exir_ops.edge.aten.tanh_.default,
            ]
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
