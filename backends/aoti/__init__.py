from executorch.export import recipe_registry

from .recipes.aoti_recipe_provider import AOTIRecipeProvider
from .recipes.aoti_recipe_types import AOTIRecipeType

# Auto-register AOTI recipe provider
recipe_registry.register_backend_recipe_provider(AOTIRecipeProvider())

__all__ = [
    "AOTIRecipeType",
]
