# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.recipes import (  # noqa: F401 (registers provider)
    CoreAIRecipeType,
)
from executorch.backends.apple.coreai.recipes.recipe_provider import (
    CoreAIRecipeProvider,
)
from executorch.exir.lowered_backend_module import executorch_call_delegate
from executorch.export import export, ExportRecipe


def _model():
    return nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).eval()


def _delegate_count(edge_manager):
    gm = edge_manager.exported_program().graph_module
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is executorch_call_delegate
    )


def _input_dtypes(edge_manager):
    gm = edge_manager.exported_program().graph_module
    return [
        n.meta["val"].dtype
        for n in gm.graph.nodes
        if n.op == "placeholder" and hasattr(n.meta.get("val"), "dtype")
    ]


class CoreAIRecipeProviderTest(unittest.TestCase):
    def test_fp32_recipe_has_no_cast_pass(self):
        recipe = CoreAIRecipeProvider().create_recipe(CoreAIRecipeType.FP32)
        self.assertIsNotNone(recipe)
        self.assertIsNone(recipe.aten_transform_passes)
        self.assertEqual(len(recipe.lowering_recipe.partitioners), 1)

    def test_fp16_recipe_has_cast_pass(self):
        recipe = CoreAIRecipeProvider().create_recipe(CoreAIRecipeType.FP16)
        self.assertIsNotNone(recipe)
        self.assertEqual(len(recipe.aten_transform_passes), 1)
        self.assertEqual(len(recipe.lowering_recipe.partitioners), 1)

    def test_extra_kwargs_are_ignored(self):
        # Unexpected kwargs are warned about, not fatal.
        recipe = CoreAIRecipeProvider().create_recipe(CoreAIRecipeType.FP32, foo=1)
        self.assertIsNotNone(recipe)


class CoreAIRecipeLoweringTest(unittest.TestCase):
    def setUp(self):
        self.example_inputs = [(torch.randn(2, 32),)]

    def test_fp32_recipe_lowers_to_coreai(self):
        recipe = ExportRecipe.get_recipe(CoreAIRecipeType.FP32)
        session = export(_model(), self.example_inputs, recipe)
        edge = session.get_edge_program_manager()
        self.assertGreaterEqual(_delegate_count(edge), 1)
        self.assertTrue(all(d == torch.float32 for d in _input_dtypes(edge)))
        self.assertGreater(len(session.get_pte_buffer()), 0)

    def test_fp16_recipe_casts_and_lowers(self):
        recipe = ExportRecipe.get_recipe(CoreAIRecipeType.FP16)
        session = export(_model(), self.example_inputs, recipe)
        edge = session.get_edge_program_manager()
        self.assertGreaterEqual(_delegate_count(edge), 1)
        # FP16 recipe casts the whole model -> edge inputs are fp16.
        self.assertTrue(all(d == torch.float16 for d in _input_dtypes(edge)))
        self.assertGreater(len(session.get_pte_buffer()), 0)


if __name__ == "__main__":
    unittest.main()
