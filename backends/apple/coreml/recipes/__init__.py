# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


from executorch.export import recipe_registry

from .coreml_recipe_provider import CoreMLRecipeProvider
from .coreml_recipe_types import CoreMLRecipeType

# Auto-register CoreML backend recipe provider
recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())

__all__ = [
    "CoreMLRecipeProvider",
    "CoreMLRecipeType",
]
