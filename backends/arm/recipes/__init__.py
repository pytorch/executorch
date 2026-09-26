# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .arm_recipe_provider import ArmRecipeProvider
from .arm_recipe_types import ArmRecipeType

recipe_registry.register_backend_recipe_provider(ArmRecipeProvider())


__all__ = ["ArmRecipeProvider", "ArmRecipeType"]
