# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import kgb
import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
)
from executorch.backends.nxp.tests.models import AddmmModule, LinearModule
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram


class TestAddmmConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    @parameterized.expand([("QAT", True), ("PTQ", False)])
    def test_addmm_conversion(self, _, use_qat: bool):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (1, 32)
            model = AddmmModule(input_shape[1])

            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.addmm.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )
            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
            )

    @parameterized.expand([("QAT", True), ("PTQ", False)])
    def test_linear_conversion__with_bias(self, _, use_qat: bool):
        with kgb.spy_on(
            EdgeProgramToIRConverter.convert_program,
            call_original=True,
            owner=EdgeProgramToIRConverter,
        ) as converter_spy:
            input_shape = (10, 32)
            model = LinearModule(bias=True)

            edge_program = to_quantized_edge_program(
                model, input_shape, use_qat=use_qat
            ).exported_program()

            # Make sure that all nodes were delegated.
            assert not graph_contains_any_of_ops(
                graph=edge_program.graph, ops=[exir_ops.edge.aten.addmm.default]
            )
            assert any(
                "lowered_module" in node.name for node in edge_program.graph.nodes
            )

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]
            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(
                np.int8
            )
            convert_run_compare(
                exported_program,
                input_data,
                tfl_model=tflite_flatbuffers_model,
            )
