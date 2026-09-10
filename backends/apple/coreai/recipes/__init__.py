# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .recipe_provider import CoreAIRecipeProvider
from .recipe_types import CoreAIRecipeType

# Auto-register the Core AI backend recipe provider on import.
recipe_registry.register_backend_recipe_provider(CoreAIRecipeProvider())

__all__ = [
    "CoreAIRecipeProvider",
    "CoreAIRecipeType",
]
