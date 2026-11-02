# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .cpu_recipe_provider import CPURecipeProvider
from .cpu_recipe_types import CPURecipeType

recipe_registry.register_backend_recipe_provider(CPURecipeProvider())

__all__ = ["CPURecipeProvider", "CPURecipeType"]
