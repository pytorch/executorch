# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

from .cortex_m_recipe_provider import CortexMRecipeProvider
from .cortex_m_recipe_types import CortexMRecipeType

recipe_registry.register_backend_recipe_provider(CortexMRecipeProvider())


__all__ = ["CortexMRecipeProvider", "CortexMRecipeType"]
