# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


from typing import Any, Optional, Sequence

import coremltools as ct

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import (
    CoreMLPartitioner,
)
from executorch.backends.apple.coreml.recipes.coreml_recipe_types import (
    COREML_BACKEND,
    CoreMLRecipeType,
)

from executorch.exir import EdgeCompileConfig
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
)


class CoreMLRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return COREML_BACKEND

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(CoreMLRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create CoreML recipe with precision and compute unit combinations"""

        if recipe_type not in self.get_supported_recipes():
            return None

        if ct is None:
            raise ImportError(
                "coremltools is required for CoreML recipes. "
                "Install it with: pip install coremltools"
            )

        # Validate kwargs
        self._validate_recipe_kwargs(recipe_type, **kwargs)

        # Parse recipe type to get precision and compute unit
        precision = None
        if recipe_type == CoreMLRecipeType.FP32:
            precision = ct.precision.FLOAT32
        elif recipe_type == CoreMLRecipeType.FP16:
            precision = ct.precision.FLOAT16

        if precision is None:
            raise ValueError(f"Unknown precision for recipe: {recipe_type.value}")

        return self._build_recipe(recipe_type, precision, **kwargs)

    def _validate_recipe_kwargs(self, recipe_type: RecipeType, **kwargs: Any) -> None:
        if not kwargs:
            return
        expected_keys = {"minimum_deployment_target", "compute_unit"}
        unexpected = set(kwargs.keys()) - expected_keys
        if unexpected:
            raise ValueError(
                f"CoreML Recipes only accept 'minimum_deployment_target' or 'compute_unit' as parameter. "
                f"Unexpected parameters: {list(unexpected)}"
            )
        if "minimum_deployment_target" in kwargs:
            minimum_deployment_target = kwargs["minimum_deployment_target"]
            if not isinstance(minimum_deployment_target, ct.target):
                raise ValueError(
                    f"Parameter 'minimum_deployment_target' must be an enum of type ct.target, got {type(minimum_deployment_target)}"
                )
        if "compute_unit" in kwargs:
            compute_unit = kwargs["compute_unit"]
            if not isinstance(compute_unit, ct.ComputeUnit):
                raise ValueError(
                    f"Parameter 'compute_unit' must be an enum of type ct.ComputeUnit, got {type(compute_unit)}"
                )

    def _build_recipe(
        self,
        recipe_type: RecipeType,
        precision: ct.precision,
        **kwargs: Any,
    ) -> ExportRecipe:
        lowering_recipe = self._get_coreml_lowering_recipe(
            compute_precision=precision,
            **kwargs,
        )

        return ExportRecipe(
            name=recipe_type.value,
            quantization_recipe=None,  # TODO - add quantization recipe
            lowering_recipe=lowering_recipe,
        )

    def _get_coreml_lowering_recipe(
        self,
        compute_precision: ct.precision,
        **kwargs: Any,
    ) -> LoweringRecipe:
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=compute_precision,
            **kwargs,
        )

        minimum_deployment_target = kwargs.get("minimum_deployment_target", None)
        take_over_mutable_buffer = True
        if minimum_deployment_target and minimum_deployment_target < ct.target.iOS18:
            take_over_mutable_buffer = False

        partitioner = CoreMLPartitioner(
            compile_specs=compile_specs,
            take_over_mutable_buffer=take_over_mutable_buffer,
        )

        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=False,
        )

        return LoweringRecipe(
            partitioners=[partitioner], edge_compile_config=edge_compile_config
        )
