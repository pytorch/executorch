# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .xnnpack_recipe_provider import XNNPACKRecipeProvider
from .xnnpack_recipe_types import XNNPackRecipeType

# Auto-register XNNPACK recipe provider
recipe_registry.register_backend_recipe_provider(XNNPACKRecipeProvider())


__all__ = [
    "XNNPACKRecipeProvider",
    "XNNPackRecipeType",
]
