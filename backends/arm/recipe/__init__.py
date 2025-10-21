# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry  # type: ignore[import-untyped]

from .recipe import ArmExportRecipe, TargetRecipe  # noqa  # usort: skip
from .arm_recipe_types import ArmRecipeType  # noqa  # usort: skip
from .arm_recipe_provider import ArmRecipeProvider  # noqa  # usort: skip

# Auto-register Arm recipe provider
recipe_registry.register_backend_recipe_provider(ArmRecipeProvider())

__all__ = ["ArmRecipeType", "ArmExportRecipe", "TargetRecipe"]
