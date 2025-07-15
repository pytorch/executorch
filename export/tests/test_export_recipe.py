# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Optional, Sequence

from executorch.export.recipe import ExportRecipe, RecipeType
from executorch.export.recipe_provider import BackendRecipeProvider
from executorch.export.recipe_registry import recipe_registry


class TestRecipeType(RecipeType):
    FP32 = "fp32"
    INT8 = "int8"
    UNSUPPORTED = "unsupported"

    @classmethod
    def get_backend_name(cls) -> str:
        return "test_backend"


class AnotherTestRecipeType(RecipeType):
    DYNAMIC = "dynamic"

    @classmethod
    def get_backend_name(cls) -> str:
        return "another_backend"


class ConcreteBackendProvider(BackendRecipeProvider):
    def __init__(
        self, backend_name: str, supported_recipes: Sequence[RecipeType]
    ) -> None:
        self._backend_name = backend_name
        self._supported_recipes = supported_recipes
        self.last_kwargs: Optional[Dict[str, Any]] = None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return self._supported_recipes

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        self.last_kwargs = kwargs
        if recipe_type in self._supported_recipes:
            return ExportRecipe(name=f"{self._backend_name}_{recipe_type.value}")
        return None


class TestExportRecipeGetRecipe(unittest.TestCase):

    def setUp(self) -> None:
        self.provider = ConcreteBackendProvider(
            "test_backend", [TestRecipeType.FP32, TestRecipeType.INT8]
        )
        recipe_registry.register_backend_recipe_provider(self.provider)

        self.another_provider = ConcreteBackendProvider(
            "another_backend", [AnotherTestRecipeType.DYNAMIC]
        )
        recipe_registry.register_backend_recipe_provider(self.another_provider)

    def tearDown(self) -> None:
        if recipe_registry._initialized:
            recipe_registry._providers.clear()

    def test_get_recipe_success(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

    def test_get_recipe_unsupported_recipe_raises_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            ExportRecipe.get_recipe(TestRecipeType.UNSUPPORTED)

        error_message = str(context.exception)
        self.assertIn(
            "Recipe 'unsupported' not supported by 'test_backend'", error_message
        )
        self.assertIn("Supported: ['fp32', 'int8']", error_message)

    def test_get_recipe_unsupported_recipe_type_raises_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            # pyre-ignore[6]
            ExportRecipe.get_recipe("abc")

        error_message = str(context.exception)
        self.assertIn("Invalid recipe type:", error_message)

    def test_get_recipe_backend_name_extraction(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

        result2 = ExportRecipe.get_recipe(AnotherTestRecipeType.DYNAMIC)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.name, "another_backend_dynamic")

    def test_get_recipe_empty_kwargs(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32, **{})

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

    def test_get_recipe_returns_correct_type(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)

        self.assertIsInstance(result, ExportRecipe)

    def test_get_recipe_with_kwargs_verification(self) -> None:
        """Test that kwargs are properly passed to recipe_registry.create_recipe"""
        kwargs = {"group_size": 32, "custom_kwarg": "val"}

        result = ExportRecipe.get_recipe(TestRecipeType.INT8, **kwargs)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_int8")

        # Verify that the kwargs were passed to the backend provider's create_recipe method
        self.assertIsNotNone(self.provider.last_kwargs)
        self.assertEqual(self.provider.last_kwargs, kwargs)
