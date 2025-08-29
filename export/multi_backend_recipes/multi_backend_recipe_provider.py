# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-backend recipe provider for target-specific deployment.

This module provides the recipe provider that combines multiple backends
for optimized deployment on specific targets.
"""

import logging
from typing import Any, Optional, Sequence

from executorch.export import BackendRecipeProvider, ExportRecipe, RecipeType

from .target_recipe_types import IOSTargetRecipeType, TargetRecipeType


class MultiBackendRecipeProvider(BackendRecipeProvider):
    """
    Recipe provider that combines multiple backends for target-specific deployment.
    """

    @property
    def backend_name(self) -> str:
        return "multi_backend"

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        """
        Returns all available target recipe types.
        """
        # Collect all target recipe types from all target classes
        recipes = []
        for target_class in [IOSTargetRecipeType]:
            recipes.extend(list(target_class.__members__.values()))

        return recipes

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """
        Create a multi-backend recipe for the given target recipe type.

        Args:
            recipe_type: Target recipe type (e.g., IOSTargetRecipeType.IOS_ARM64_COREML_FP32)
            **kwargs: Additional parameters to pass to individual backend recipes

        Returns:
            ExportRecipe configured for multi-backend deployment
        """
        if not isinstance(recipe_type, TargetRecipeType):
            return None

        # Get the backend combination for this target
        backend_recipe_types = recipe_type.get_backend_combination()
        if not backend_recipe_types:
            logging.warning(
                f"No backend recipes available for target: {recipe_type.value}"
            )
            return None

        # Create individual backend recipes using the existing recipe registry
        backend_recipes = []
        for backend_recipe_type in backend_recipe_types:
            # Extract backend-specific kwargs
            backend_recipe_kwargs = self._extract_backend_recipe_kwargs(
                backend_recipe_type.get_backend_name(), kwargs
            )

            backend_recipe = ExportRecipe.get_recipe(
                backend_recipe_type, **backend_recipe_kwargs
            )
            backend_recipes.append(backend_recipe)

        if not backend_recipes:
            raise ValueError(
                f"No available backend recipes for target '{recipe_type.value}'. "
                f"Ensure required backend dependencies are installed."
            )

        # Combine the recipes into a single multi-backend recipe
        return ExportRecipe.combine(backend_recipes, recipe_type.value)

    def _extract_backend_recipe_kwargs(self, backend_name: str, kwargs: dict) -> dict:
        """
        Extract backend-specific kwargs using nested dictionary structure.

        Args:
            backend_name: Name of the backend (e.g., "xnnpack", "coreml")
            kwargs: All kwargs passed to the target recipe

        Returns:
            Dict of kwargs specific to this backend
        """
        backend_kwargs = {}

        # Extract from nested backend_configs structure
        backend_configs = kwargs.get("backend_configs", {})
        if backend_name in backend_configs:
            backend_kwargs.update(backend_configs[backend_name])

        # Include common kwargs (parameters passed directly without backend_configs)
        common_kwargs = self._get_common_kwargs(kwargs)
        backend_kwargs.update(common_kwargs)

        return backend_kwargs

    def _get_common_kwargs(self, kwargs: dict) -> dict:
        excluded_keys = {
            "backend_configs",  # Skip, represents nested backend recipe kwargs
        }
        common_kwargs = {}
        for key, value in kwargs.items():
            if key not in excluded_keys:
                common_kwargs[key] = value
        return common_kwargs
