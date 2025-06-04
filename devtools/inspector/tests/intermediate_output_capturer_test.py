# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)
from executorch.devtools.inspector.tests.inspector_test_utils import (
    check_if_final_outputs_match,
    model_registry,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import export, ExportedProgram
from torch.fx import GraphModule


class TestIntermediateOutputCapturer(unittest.TestCase):
    def _set_up_model(self, model_name):
        model = model_registry[model_name]()
        input_tensor = model.get_input()
        aten_model: ExportedProgram = export(model, (input_tensor,), strict=True)
        edge_program_manager: EdgeProgramManager = to_edge(
            aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
        )
        graph_module: GraphModule = edge_program_manager._edge_programs[
            "forward"
        ].module()
        capturer = IntermediateOutputCapturer(graph_module)
        intermediate_outputs = capturer.run_and_capture(input_tensor)
        return input_tensor, graph_module, capturer, intermediate_outputs

    def test_models(self):
        available_models = list(model_registry.keys())
        for model_name in available_models:
            with self.subTest(model=model_name):
                input_tensor, graph_module, capturer, intermediate_outputs = (
                    self._set_up_model(model_name)
                )

                # Test keying with debug handle tuple
                for key in intermediate_outputs.keys():
                    self.assertIsInstance(key, tuple)

                # Test tensor cloning and detaching
                for output in intermediate_outputs.values():
                    if isinstance(output, torch.Tensor):
                        self.assertFalse(output.requires_grad)
                        self.assertTrue(output.is_leaf)

                # Test placeholder nodes are skipped
                for node in graph_module.graph.nodes:
                    if node.op == "placeholder":
                        self.assertNotIn(node.meta.get("debug_handle"), node.meta)

                # Test multiple outputs capture
                outputs = capturer.run_and_capture(input_tensor)
                for output in outputs.values():
                    if isinstance(output, tuple):
                        self.assertEqual(len(output), 2)
                        for part in output:
                            self.assertIsInstance(part, torch.Tensor)

                # Test capture correct outputs
                self.assertTrue(
                    check_if_final_outputs_match(model_name, intermediate_outputs)
                )
