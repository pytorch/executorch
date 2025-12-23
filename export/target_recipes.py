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

import os
from typing import Dict, List

from executorch.backends.xnnpack.recipes import XNNPackRecipeType
from executorch.export.recipe import ExportRecipe, RecipeType
from executorch.export.utils import (
    is_supported_platform_for_coreml_lowering,
    is_supported_platform_for_qnn_lowering,
)


def _create_target_recipe(
    target_config: str, recipes: List[RecipeType], **kwargs
) -> ExportRecipe:
    """
    Create a combined recipe for a target.

    Args:
        target_config: Human-readable hardware configuration name
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

    if len(backend_recipes) == 1:
        return backend_recipes[0]

    return ExportRecipe.combine(backend_recipes, recipe_name=target_config)


# IOS Recipe
def get_ios_recipe(
    target_config: str = "ios-arm64-coreml-fp16", **kwargs
) -> ExportRecipe:
    """
    Get iOS-optimized recipe for specified hardware configuration.

    Supported configurations:
    - 'ios-arm64-coreml-fp32': CoreML + XNNPACK fallback (FP32)
    - 'ios-arm64-coreml-fp16': CoreML fp16 recipe
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

    if not is_supported_platform_for_coreml_lowering():
        raise ValueError("CoreML is not supported on this platform")

    import coremltools as ct
    from executorch.backends.apple.coreml.recipes import CoreMLRecipeType

    ios_configs: Dict[str, List[RecipeType]] = {
        # pyre-ignore
        "ios-arm64-coreml-fp32": [CoreMLRecipeType.FP32, XNNPackRecipeType.FP32],
        # pyre-ignore
        "ios-arm64-coreml-fp16": [CoreMLRecipeType.FP16, XNNPackRecipeType.FP32],
        # pyre-ignore
        "ios-arm64-coreml-int8": [CoreMLRecipeType.PT2E_INT8_STATIC],
    }

    if target_config not in ios_configs:
        supported = list(ios_configs.keys())
        raise ValueError(
            f"Unsupported iOS configuration: '{target_config}'. "
            f"Supported: {supported}"
        )

    kwargs = kwargs or {}

    if target_config == "ios-arm64-coreml-int8":
        if "minimum_deployment_target" not in kwargs:
            kwargs["minimum_deployment_target"] = ct.target.iOS17

    backend_recipes = ios_configs[target_config]
    return _create_target_recipe(target_config, backend_recipes, **kwargs)


# Android Recipe
def get_android_recipe(
    target_config: str = "android-arm64-snapdragon-fp16", **kwargs
) -> ExportRecipe:
    """
    Get Android-optimized recipe for specified hardware configuration.

    Supported configurations:
    - 'android-arm64-snapdragon-fp16': QNN fp16 recipe

    Args:
        target_config: Android configuration string
        **kwargs: Additional parameters for backend recipes

    Returns:
        ExportRecipe configured for Android deployment

    Raises:
        ValueError: If target configuration is not supported

    Example:
        recipe = get_android_recipe('android-arm64-snapdragon-fp16')
        session = export(model, recipe, example_inputs)
    """

    if not is_supported_platform_for_qnn_lowering():
        raise ValueError(
            "QNN is not supported or not properly configured on this platform"
        )

    try:
        # Qualcomm QNN backend runs QNN sdk download on first use
        # with a pip install, so wrap it in a try/except
        # pyre-ignore
        from executorch.backends.qualcomm.recipes import QNNRecipeType

        # (1) if this is called from a pip install, the QNN SDK will be available
        # (2) if this is called from a source build, check if qnn is available otherwise, had to run build.sh
        if os.getenv("QNN_SDK_ROOT", None) is None:
            raise ValueError(
                "QNN SDK not found, cannot use QNN recipes. First run `./backends/qualcomm/scripts/build.sh`, if building from source"
            )
    except Exception as e:
        raise ValueError(
            "QNN backend is not available. Please ensure the Qualcomm backend "
            "is properly installed and configured, "
        ) from e

    android_configs: Dict[str, List[RecipeType]] = {
        # pyre-ignore
        "android-arm64-snapdragon-fp16": [QNNRecipeType.FP16, XNNPackRecipeType.FP32],
    }

    if target_config not in android_configs:
        supported = list(android_configs.keys())
        raise ValueError(
            f"Unsupported Android configuration: '{target_config}'. "
            f"Supported: {supported}"
        )

    kwargs = kwargs or {}

    if target_config == "android-arm64-snapdragon-fp16":
        if "soc_model" not in kwargs:
            kwargs["soc_model"] = "SM8650"

    backend_recipes = android_configs[target_config]
    return _create_target_recipe(target_config, backend_recipes, **kwargs)
