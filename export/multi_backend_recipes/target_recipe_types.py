# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Target-specific recipe types for multi-backend deployment.

This module defines target-specific recipe types that combine multiple backends
for optimized deployment on specific platforms and hardware configurations.
"""

from abc import abstractmethod
from typing import List

# pyre-ignore
from executorch.backends.apple.coreml.recipes.coreml_recipe_types import (
    CoreMLRecipeType,
)

from executorch.backends.xnnpack import XNNPackRecipeType

from executorch.export.recipe import RecipeType, RecipeTypeMeta


class TargetRecipeTypeMeta(RecipeTypeMeta):
    """Metaclass that extends RecipeTypeMeta for target recipe types"""

    pass


class TargetRecipeType(RecipeType, metaclass=TargetRecipeTypeMeta):
    """
    Base class for target-specific recipe types that combine multiple backends.
    Target recipes define optimal backend combinations for specific deployment scenarios.
    """

    @abstractmethod
    def get_backend_combination(self) -> List[RecipeType]:
        """
        Return the backend combination for this target recipe type.

        Returns:
            List of RecipeType enums in order of precedence.
            Earlier backends get first chance at partitioning operations.
        """
        pass

    @classmethod
    @abstractmethod
    def get_target_platform(cls) -> str:
        """
        Return the target platform for this recipe type.

        Returns:
            str: The target platform (e.g., "android", "ios", "linux")
        """
        pass

    @classmethod
    def get_backend_name(cls) -> str:
        return "multi_backend"


class IOSTargetRecipeType(TargetRecipeType):
    """iOS-specific target recipe types, refer individual backend recipes for customization"""

    # FP32 - iOS with CoreML backend as primary, fallback XNNPACK
    IOS_ARM64_COREML_FP32 = "ios-arm64-coreml-fp32"

    # FP16 - iOS with CoreML backend as primary, fallback XNNPACK
    IOS_ARM64_COREML_FP16 = "ios-arm64-coreml-fp16"

    # INT8 Static Quantization (weights + activations)
    # No xnnpack fallback for quantization as coreml uses torch.ao quantizer vs xnnpack uses torchao quantizer
    IOS_ARM64_COREML_INT8_STATIC = "ios-arm64-coreml-int8-static"

    @classmethod
    def get_target_platform(cls) -> str:
        return "ios"

    def get_backend_combination(self) -> List[RecipeType]:
        """
        Get backend combinations for iOS targets.

        Returns combinations optimized for iOS deployment scenarios.
        """
        if self == IOSTargetRecipeType.IOS_ARM64_COREML_FP32:
            return [CoreMLRecipeType.FP32, XNNPackRecipeType.FP32]
        elif self == IOSTargetRecipeType.IOS_ARM64_COREML_FP16:
            return [CoreMLRecipeType.FP16, XNNPackRecipeType.FP32]
        elif self == IOSTargetRecipeType.IOS_ARM64_COREML_INT8_STATIC:
            return [CoreMLRecipeType.PT2E_INT8_STATIC]
        return []
