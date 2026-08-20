# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Recipe registry for managing backend recipe providers.

This module provides the registry system for backend recipe providers and
the abstract interface that all backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .recipe import ExportRecipe, RecipeType


class BackendRecipeProvider(ABC):
    """
    Abstract recipe provider that all backends must implement
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Name of the backend (ex: 'xnnpack', 'qnn' etc)
        """
        pass

    @abstractmethod
    def get_supported_recipes(self) -> Sequence[RecipeType]:
        """
        Get list of supported recipes.
        """
        pass

    @abstractmethod
    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """
        Create a recipe for the given type.
        Returns None if the recipe is not supported by this backend.

        Args:
            recipe_type: The type of recipe to create
            **kwargs: Recipe-specific parameters (ex: group_size)

        Returns:
            ExportRecipe if supported, None otherwise
        """
        pass

    def supports_recipe(self, recipe_type: RecipeType) -> bool:
        return recipe_type in self.get_supported_recipes()
