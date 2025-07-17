# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Optional, Sequence

from executorch.export import BackendRecipeProvider, ExportRecipe, RecipeType


class TestRecipeType(RecipeType):
    FP32 = "fp32"
    INT8 = "int8"
    UNSUPPORTED = "unsupported"

    @classmethod
    def get_backend_name(cls) -> str:
        return "test_backend"


class ConcreteBackendProvider(BackendRecipeProvider):
    """Mock backend provider for testing"""

    def __init__(
        self, backend_name: str, supported_recipes: Sequence[RecipeType]
    ) -> None:
        self._backend_name = backend_name
        self._supported_recipes = supported_recipes

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return self._supported_recipes

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        _ = kwargs
        if recipe_type in self._supported_recipes:
            return ExportRecipe(name=f"{self._backend_name}_{recipe_type.value}")
        return None


class TestBackendRecipeProvider(unittest.TestCase):

    def setUp(self) -> None:
        self.supported_recipes = [TestRecipeType.FP32, TestRecipeType.INT8]
        self.provider = ConcreteBackendProvider("test_backend", self.supported_recipes)

    def test_get_supported_recipes(self) -> None:
        recipes = self.provider.get_supported_recipes()
        self.assertIn(TestRecipeType.FP32, recipes)
        self.assertIn(TestRecipeType.INT8, recipes)

    def test_create_recipe_supported(self) -> None:
        recipe = self.provider.create_recipe(TestRecipeType.FP32)
        self.assertIsNotNone(recipe)
        self.assertIsInstance(recipe, ExportRecipe)
        self.assertEqual(recipe.name, "test_backend_fp32")

    def test_supports_recipe_true(self) -> None:
        self.assertTrue(self.provider.supports_recipe(TestRecipeType.FP32))
        self.assertTrue(self.provider.supports_recipe(TestRecipeType.INT8))

    def test_supports_recipe_false(self) -> None:
        self.assertFalse(self.provider.supports_recipe(TestRecipeType.UNSUPPORTED))

    def test_empty_supported_recipes(self) -> None:
        empty_provider = ConcreteBackendProvider("empty_backend", [])

        self.assertEqual(empty_provider.get_supported_recipes(), [])
        self.assertFalse(empty_provider.supports_recipe(TestRecipeType.FP32))
        self.assertIsNone(empty_provider.create_recipe(TestRecipeType.FP32))

    def test_create_recipe_consistency(self) -> None:
        for recipe_type in [
            TestRecipeType.FP32,
            TestRecipeType.INT8,
            TestRecipeType.UNSUPPORTED,
        ]:
            supports = self.provider.supports_recipe(recipe_type)
            recipe = self.provider.create_recipe(recipe_type)

            if supports:
                self.assertIsNotNone(
                    recipe, f"Recipe should be created for supported type {recipe_type}"
                )
            else:
                self.assertIsNone(
                    recipe,
                    f"Recipe should not be created for unsupported type {recipe_type}",
                )
