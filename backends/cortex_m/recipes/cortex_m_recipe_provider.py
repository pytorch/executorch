# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Optional, Sequence

from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)

from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.op_backend import CortexMOpBackend
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.recipes.cortex_m_recipe_types import (
    CORTEX_M_BACKEND,
    CortexMRecipeType,
)
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    RecipeType,
)


logger: logging.Logger = logging.getLogger(__name__)

_RECIPE_KWARGS: frozenset[str] = frozenset({"target", "isa"})

# CortexMPassManager's own default, restated so the recipe picks it explicitly.
_DEFAULT_TARGET: str = "cortex-m55"


class CortexMRecipeProvider(BackendRecipeProvider):
    """Builds ExportRecipes for the Cortex-M CMSIS-NN backend.

    A recipe produces the same quantizer, to_edge configuration and pass
    pipeline as the ``--target=cortex-m<variant>`` path of
    ``backends/arm/scripts/aot_arm_compiler.py``. The one thing it does not do
    is that path's move of the model and its inputs to channels_last; see
    ``CortexMRecipeType``.
    """

    @property
    def backend_name(self) -> str:
        return CORTEX_M_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        # Only what create_recipe dispatches, so a new enum member cannot be
        # advertised as supported and then refused.
        return [CortexMRecipeType.INT8]

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        if recipe_type is not CortexMRecipeType.INT8:
            return None

        # Warn, as the other providers do: a combined recipe hands every
        # backend the same kwargs.
        unexpected = set(kwargs) - _RECIPE_KWARGS
        if unexpected:
            logger.warning(
                "Cortex-M recipe '%s' ignoring unexpected parameters: %s. Allowed: %s",
                recipe_type.value,
                sorted(unexpected),
                sorted(_RECIPE_KWARGS),
            )

        # Not `or`: a falsy target would silently become the default instead of
        # reaching the check below.
        target = kwargs.get("target", _DEFAULT_TARGET)
        if not isinstance(target, str):
            raise ValueError(
                f"Cortex-M recipe 'target' must be a 'cortex-m<variant>' string, "
                f"got {target!r}"
            )
        isa = kwargs.get("isa")
        if isa is not None and not isinstance(isa, cmsis_nn.Backend):
            raise ValueError(
                "Cortex-M recipe 'isa' must be a "
                "executorch.backends.cortex_m.library.cmsis_nn.Backend, got "
                f"{isa!r}"
            )

        cpu = CortexMTargetConfig.from_target_string(target).cpu
        target_config = CortexMTargetConfig(cpu=cpu, isa=isa)

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=QuantizationRecipe(quantizers=[CortexMQuantizer()]),
            lowering_recipe=LoweringRecipe(
                edge_compile_config=cortex_m_edge_compile_config(),
                op_backends=[CortexMOpBackend(target_config)],
            ),
        )
