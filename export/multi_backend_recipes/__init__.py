# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import recipe_registry

from .multi_backend_recipe_provider import MultiBackendRecipeProvider
from .target_recipe_types import TargetRecipeType

# Auto-register MultiBackendRecipeProvider
recipe_registry.register_backend_recipe_provider(MultiBackendRecipeProvider())

__all__ = [
    "MultiBackendRecipeProvider",
    "TargetRecipeType",
]
