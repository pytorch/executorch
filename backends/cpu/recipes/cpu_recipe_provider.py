# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Sequence

from executorch.backends.cpu.partitioner import CPUPartitioner
from executorch.backends.cpu.recipes.cpu_recipe_types import CPURecipeType
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
)


class CPURecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return "cpu"

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return [CPURecipeType.FP32]

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if recipe_type not in self.get_supported_recipes():
            return None
        if kwargs:
            raise ValueError(f"Unexpected CPU recipe options: {sorted(kwargs)}")
        return ExportRecipe(
            name=recipe_type.value,
            lowering_recipe=LoweringRecipe(partitioners=[CPUPartitioner()]),
        )
