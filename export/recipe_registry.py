# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Recipe registry for managing backend recipe providers.

This module provides the registry system for backend recipe providers and
the abstract interface that all backends must implement.
"""

from typing import Any, Dict, Optional, Sequence

from .recipe import ExportRecipe, RecipeType
from .recipe_provider import BackendRecipeProvider


class RecipeRegistry:
    """Global registry for all backend recipe providers"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once to avoid resetting state on subsequent calls
        if not RecipeRegistry._initialized:
            self._providers: Dict[str, BackendRecipeProvider] = {}
            RecipeRegistry._initialized = True

    def register_backend_recipe_provider(self, provider: BackendRecipeProvider) -> None:
        """
        Register a backend recipe provider
        """
        self._providers[provider.backend_name] = provider

    def create_recipe(
        self, recipe_type: RecipeType, backend: str, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """
        Create a recipe for a specific backend.

        Args:
            recipe_type: The type of recipe to create
            backend: Backend name
            **kwargs: Recipe-specific parameters

        Returns:
            ExportRecipe if supported, None if not supported
        """
        if backend not in self._providers:
            raise ValueError(
                f"Backend '{backend}' not available. Available: {list(self._providers.keys())}"
            )

        return self._providers[backend].create_recipe(recipe_type, **kwargs)

    def get_supported_recipes(self, backend: str) -> Sequence[RecipeType]:
        """
        Get list of recipes supported by a backend.

        Args:
            backend: Backend name

        Returns:
            List of supported recipe types
        """
        if backend not in self._providers:
            raise ValueError(f"Backend '{backend}' not available")
        return self._providers[backend].get_supported_recipes()

    def list_backends(self) -> Sequence[str]:
        """
        Get list of all registered backends
        """
        return list(self._providers.keys())


# initialize recipe registry
recipe_registry = RecipeRegistry()
