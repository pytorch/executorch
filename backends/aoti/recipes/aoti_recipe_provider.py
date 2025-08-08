# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional, Sequence

from executorch.backends.aoti.recipes.aoti_recipe_types import AOTIRecipeType

from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
    StageType,
)


class AOTIRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return "aoti"

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return [AOTIRecipeType.FP32]

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create AOTI recipe"""

        if recipe_type not in self.get_supported_recipes():
            return None

        if recipe_type == AOTIRecipeType.FP32:
            return self._build_fp32_recipe(recipe_type)

    def _get_aoti_lowering_recipe(self) -> LoweringRecipe:
        return LoweringRecipe(
            partitioners=None,
            edge_transform_passes=None,
            edge_compile_config=None,
        )

    def _build_fp32_recipe(self, recipe_type: RecipeType) -> ExportRecipe:
        return ExportRecipe(
            name=recipe_type.value,
            lowering_recipe=self._get_aoti_lowering_recipe(),
            pipeline_stages=[StageType.TORCH_EXPORT, StageType.AOTI_LOWERING],
        )
