# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Sequence

from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.recipes.recipe_types import (
    COREAI_BACKEND,
    CoreAIRecipeType,
)
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
)

logger = logging.getLogger(__name__)


def _cast_fp32_to_fp16_pass(_method_name, exported_program):
    """aten_transform pass: cast the program to FP16 before edge lowering.

    coreai-torch does not auto-downcast (unlike coremltools), so we run
    coreai_opt's ``cast_fp32_to_fp16`` on the ExportedProgram pre-``to_edge``.
    Running it on the whole model before partitioning keeps the cast off the
    delegate boundary: the model's own I/O becomes FP16 rather than the
    partitions disagreeing about where conversions belong. Values FP16 cannot
    represent stay FP32, so the result is mixed, not uniformly FP16.

    Note that ``cast_fp32_to_fp16`` mutates and returns the program it is
    given.
    """
    from coreai_opt.casting import cast_fp32_to_fp16

    return cast_fp32_to_fp16(exported_program)


def _default_edge_transform_passes(_method_name, _exported_program):
    """Factory the recipe framework calls as ``(method_name, ep) -> [passes]``.

    LoweringRecipe.edge_transform_passes entries are pass *factories* (see
    executorch/export/stages.py), unlike ``to_edge_transform_and_lower``'s
    ``transform_passes`` which take the passes directly.  Both ultimately apply
    the same GraphModule passes from ``get_default_passes``.
    """
    from executorch.backends.apple.coreai import get_default_passes

    return get_default_passes()


class CoreAIRecipeProvider(BackendRecipeProvider):
    """Provides Core AI export recipes (FP32 / FP16)."""

    @property
    def backend_name(self) -> str:
        return COREAI_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(CoreAIRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if recipe_type not in self.get_supported_recipes():
            return None

        if kwargs:
            logger.warning(
                "Core AI recipe '%s' ignoring unexpected parameters: %s",
                recipe_type.value,
                list(kwargs.keys()),
            )

        if recipe_type == CoreAIRecipeType.FP32:
            return self._build_recipe(recipe_type, fp16=False)
        if recipe_type == CoreAIRecipeType.FP16:
            return self._build_recipe(recipe_type, fp16=True)
        return None

    def _build_recipe(self, recipe_type: RecipeType, fp16: bool) -> ExportRecipe:
        from executorch.backends.apple.coreai import get_default_compile_config

        lowering_recipe = LoweringRecipe(
            partitioners=[CoreAIPartitioner()],
            edge_transform_passes=[_default_edge_transform_passes],
            edge_compile_config=get_default_compile_config(),
        )
        return ExportRecipe(
            name=recipe_type.value,
            aten_transform_passes=[_cast_fp32_to_fp16_pass] if fp16 else None,
            lowering_recipe=lowering_recipe,
        )
