# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Dict, Tuple, Union

import torch

from executorch.devtools.inspector._inspector_utils import (
    DebugHandle,
    propagate_back_debug_handle,
)
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)
from executorch.devtools.inspector.tests.inspector_test_utils import (
    check_if_intermediate_outputs_match,
    model_registry,
)

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import export, ExportedProgram


class TestIntermediateOutputCapturer(unittest.TestCase):
    def _capture_intermediate_outputs_and_check(
        self,
        inputs: Tuple[torch.Tensor],
        ep: ExportedProgram,
        expected_intermediate_outputs: Dict[
            DebugHandle, Union[torch.Tensor, Tuple[torch.Tensor]]
        ],
    ):
        captured_intermediate_outputs = IntermediateOutputCapturer(
            ep.module()
        ).run_and_capture(inputs)

        # Test keying with debug handle tuple
        for key in captured_intermediate_outputs.keys():
            self.assertIsInstance(key, tuple)

        # Test tensor cloning and detaching
        for output in captured_intermediate_outputs.values():
            if isinstance(output, torch.Tensor):
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)

        # Test placeholder nodes are skipped
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertNotIn(node.meta.get("debug_handle"), node.meta)

        # Test multiple outputs capture
        for inter_output in captured_intermediate_outputs.values():
            if isinstance(inter_output, tuple):
                for part in output:
                    self.assertIsInstance(part, torch.Tensor)

        # Test capture correct outputs
        self.assertTrue(
            check_if_intermediate_outputs_match(
                captured_intermediate_outputs, expected_intermediate_outputs
            )
        )

    def test_models(self):
        available_models = list(model_registry.keys())
        for model_name in available_models:
            with self.subTest(model=model_name):
                model = model_registry[model_name]()
                input_tensor = model.get_input()
                aten_model: ExportedProgram = export(model, (input_tensor,))
                aten_model_graph_id = id(aten_model.graph)

                edge_program_manager: EdgeProgramManager = to_edge(
                    aten_model,
                    compile_config=EdgeCompileConfig(_check_ir_validity=True),
                )

                ret = propagate_back_debug_handle(
                    aten_model,
                    aten_model_graph_id,
                    edge_program_manager.exported_program(),
                )
                assert ret is True

                self._capture_intermediate_outputs_and_check(
                    input_tensor,
                    aten_model,
                    model.get_exported_program_expected_intermediate_outputs(),
                )
                self._capture_intermediate_outputs_and_check(
                    input_tensor,
                    edge_program_manager.exported_program(),
                    model.get_edge_dialect_expected_intermediate_outputs(),
                )
