# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .nxp_recipe_provider import NeutronRecipeConfig, NXPRecipeProvider
from .nxp_recipe_types import NXPRecipeType

# Auto-register NXP recipe provider
recipe_registry.register_backend_recipe_provider(NXPRecipeProvider())

__all__ = [
    "NeutronRecipeConfig",
    "NXPRecipeProvider",
    "NXPRecipeType",
]
