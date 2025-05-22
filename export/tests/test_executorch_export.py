# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.export import export, ExportRecipe


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
            model=model, example_inputs=example_inputs, export_recipe=export_recipe
        )

        self.assertTrue(len(export_session.get_pte_buffer()) != 0)
