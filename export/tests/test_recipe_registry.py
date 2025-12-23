# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Optional, Sequence

from executorch.export.recipe import ExportRecipe, RecipeType
from executorch.export.recipe_provider import BackendRecipeProvider
from executorch.export.recipe_registry import recipe_registry, RecipeRegistry


class TestRecipeType(RecipeType):
    FP32 = "fp32"
    INT8 = "int8"

    @classmethod
    def get_backend_name(cls) -> str:
        return "test_backend"


class MockBackendProvider(BackendRecipeProvider):
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


class TestRecipeRegistry(unittest.TestCase):

    def setUp(self) -> None:
        # Create a fresh registry for each test
        RecipeRegistry._instance = None
        RecipeRegistry._initialized = False
        self.registry = RecipeRegistry()

    def test_get_supported_recipes_type(self) -> None:
        provider = MockBackendProvider("test_backend", [TestRecipeType.FP32])
        self.registry.register_backend_recipe_provider(provider)

        self.assertIsInstance(self.registry.get_supported_recipes("test_backend"), list)
        for recipe in self.registry.get_supported_recipes("test_backend"):
            self.assertIsInstance(recipe, RecipeType)

    def test_singleton_pattern(self) -> None:
        registry1 = RecipeRegistry()
        registry2 = RecipeRegistry()
        self.assertIs(registry1, registry2)

    def test_register_backend_recipe_provider(self) -> None:
        provider = MockBackendProvider("test_backend", [TestRecipeType.FP32])
        self.registry.register_backend_recipe_provider(provider)

        backends = self.registry.list_backends()
        self.assertIn("test_backend", backends)

    def test_create_recipe_success(self) -> None:
        provider = MockBackendProvider(
            "test_backend", [TestRecipeType.FP32, TestRecipeType.INT8]
        )
        self.registry.register_backend_recipe_provider(provider)

        recipe = self.registry.create_recipe(TestRecipeType.FP32, "test_backend")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "test_backend_fp32")

    def test_create_recipe_unsupported_backend(self) -> None:
        with self.assertRaises(ValueError) as context:
            self.registry.create_recipe(TestRecipeType.FP32, "nonexistent_backend")
        self.assertIn(
            "Backend 'nonexistent_backend' not available", str(context.exception)
        )

    def test_create_recipe_unsupported_recipe_type(self) -> None:
        provider = MockBackendProvider("test_backend", [TestRecipeType.FP32])
        self.registry.register_backend_recipe_provider(provider)
        recipe = self.registry.create_recipe(TestRecipeType.INT8, "test_backend")
        self.assertIsNone(recipe)

    def test_get_supported_recipes(self) -> None:
        supported_recipes = [TestRecipeType.FP32, TestRecipeType.INT8]
        provider = MockBackendProvider("test_backend", supported_recipes)
        self.registry.register_backend_recipe_provider(provider)

        recipes = self.registry.get_supported_recipes("test_backend")
        self.assertEqual(recipes, supported_recipes)

    def test_get_supported_recipes_unknown_backend(self) -> None:
        with self.assertRaises(ValueError) as context:
            self.registry.get_supported_recipes("unknown_backend")

        self.assertIn("Backend 'unknown_backend' not available", str(context.exception))

    def test_list_backends(self) -> None:
        provider1 = MockBackendProvider("backend1", [TestRecipeType.FP32])
        provider2 = MockBackendProvider("backend2", [TestRecipeType.INT8])

        self.registry.register_backend_recipe_provider(provider1)
        self.registry.register_backend_recipe_provider(provider2)

        backends = self.registry.list_backends()
        self.assertIn("backend1", backends)
        self.assertIn("backend2", backends)
        self.assertEqual(len(backends), 2)

    def test_list_backends_empty(self) -> None:
        backends = self.registry.list_backends()
        self.assertEqual(backends, [])

    def test_global_registry_instance(self) -> None:
        provider = MockBackendProvider("global_test", [TestRecipeType.FP32])
        recipe_registry.register_backend_recipe_provider(provider)

        backends = recipe_registry.list_backends()
        self.assertIn("global_test", backends)

        recipe = recipe_registry.create_recipe(TestRecipeType.FP32, "global_test")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "global_test_fp32")
