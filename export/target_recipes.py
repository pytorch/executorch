# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Target-specific recipe functions for simplified multi-backend deployment.

This module provides platform-specific functions that abstract away backend 
selection and combine multiple backends optimally for target hardware.
"""

from typing import Dict, List

import coremltools as ct

# pyre-ignore
from executorch.backends.apple.coreml.recipes import CoreMLRecipeType
from executorch.backends.xnnpack.recipes import XNNPackRecipeType
from executorch.export.recipe import ExportRecipe, RecipeType


## Target configs
IOS_CONFIGS: Dict[str, List[RecipeType]] = {
    # pyre-ignore
    "ios-arm64-coreml-fp32": [CoreMLRecipeType.FP32, XNNPackRecipeType.FP32],
    # pyre-ignore
    "ios-arm64-coreml-int8": [CoreMLRecipeType.PT2E_INT8_STATIC],
}


def _create_target_recipe(
    target_config: str, recipes: List[RecipeType], **kwargs
) -> ExportRecipe:
    """
    Create a combined recipe for a target.

    Args:
        target: Human-readable hardware configuration name
        recipes: List of backend recipe types to combine
        **kwargs: Additional parameters - each backend will use what it needs

    Returns:
        Combined ExportRecipe for the hardware configuration
    """
    if not recipes:
        raise ValueError(f"No backends configured for: {target_config}")

    # Create individual backend recipes
    backend_recipes = []
    for recipe_type in recipes:
        try:
            backend_recipe = ExportRecipe.get_recipe(recipe_type, **kwargs)
            backend_recipes.append(backend_recipe)
        except Exception as e:
            raise ValueError(
                f"Failed to create {recipe_type.value} recipe for {target_config}: {e}"
            ) from e

    # Combine into single recipe
    if len(backend_recipes) == 1:
        return backend_recipes[0]

    return ExportRecipe.combine(backend_recipes, recipe_name=target_config)


# IOS Recipe
def get_ios_recipe(
    target_config: str = "ios-arm64-coreml-fp32", **kwargs
) -> ExportRecipe:
    """
    Get iOS-optimized recipe for specified hardware configuration.

    Supported configurations:
    - 'ios-arm64-coreml-fp32': CoreML + XNNPACK fallback (FP32)
    - 'ios-arm64-coreml-int8': CoreML INT8 quantization recipe

    Args:
        target_config: iOS configuration string
        **kwargs: Additional parameters for backend recipes

    Returns:
        ExportRecipe configured for iOS deployment

    Raises:
        ValueError: If target configuration is not supported

    Example:
        recipe = get_ios_recipe('ios-arm64-coreml-int8')
        session = export(model, recipe, example_inputs)
    """
    if target_config not in IOS_CONFIGS:
        supported = list(IOS_CONFIGS.keys())
        raise ValueError(
            f"Unsupported iOS configuration: '{target_config}'. "
            f"Supported: {supported}"
        )

    kwargs = kwargs or {}

    if target_config == "ios-arm64-coreml-int8":
        if "minimum_deployment_target" not in kwargs:
            kwargs["minimum_deployment_target"] = ct.target.iOS17

    backend_recipes = IOS_CONFIGS[target_config]
    return _create_target_recipe(target_config, backend_recipes, **kwargs)
