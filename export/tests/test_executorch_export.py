# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from executorch.export import ExportRecipe, export

class TestExecutorchExport(unittest.TestCase):
    def test_basic_recipe(self) -> None:
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        example_inputs = [(torch.rand(1, 10),)]
        export_recipe = ExportRecipe()
        
        # Use the export API instead of creating ExportSession directly
        export_session = export(
            model=model, 
            example_inputs=example_inputs, 
            export_recipe=export_recipe
        )
        
        # The export function doesn't automatically call export() on the session
        export_session.export()
        self.assertTrue(len(export_session.get_pte_buffer())!= 0)
