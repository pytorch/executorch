# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Optional, Sequence

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.recipe import ArmExportRecipe, ArmRecipeType, TargetRecipe
from executorch.backends.arm.test import common
from executorch.backends.arm.util._factory import create_quantizer
from executorch.export import (  # type: ignore[import-untyped]
    BackendRecipeProvider,
    ExportRecipe,
    QuantizationRecipe,
    RecipeType,
)

QuantizerConfigurator = Callable[[TOSAQuantizer], None]


def global_int8_per_channel(quantizer: TOSAQuantizer):
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))


def global_int8_per_tensor(quantizer: TOSAQuantizer):
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))


class ArmRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return ArmRecipeType.get_backend_name()

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(ArmRecipeType)

    @classmethod
    def build_export_recipe(
        cls,
        recipe_type: RecipeType,
        target_recipe: TargetRecipe,
        quantization_configurators: Optional[list[QuantizerConfigurator]] = None,
    ) -> ArmExportRecipe:

        if quantization_configurators is not None:
            quantizer = create_quantizer(target_recipe.compile_spec)
            for configure in quantization_configurators:
                configure(quantizer)
            quantization_recipe = QuantizationRecipe([quantizer])
        else:
            quantization_recipe = None

        return ArmExportRecipe(
            name=str(recipe_type),
            target_recipe=target_recipe,
            quantization_recipe=quantization_recipe,
        )

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create arm recipe"""
        return create_recipe(recipe_type, **kwargs)


def create_recipe(recipe_type: RecipeType, **kwargs: Any) -> ArmExportRecipe:
    """Create an ArmExportRecipe depending on the ArmRecipeType enum, with some kwargs. See documentation for
    the ArmRecipeType for the available kwargs."""

    match recipe_type:
        case ArmRecipeType.TOSA_FP:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_tosa_compile_spec("TOSA-1.0+FP", **kwargs)),
            )
        case ArmRecipeType.TOSA_INT8_STATIC_PER_TENSOR:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_tosa_compile_spec("TOSA-1.0+INT", **kwargs)),
                [global_int8_per_tensor],
            )
        case ArmRecipeType.TOSA_INT8_STATIC_PER_CHANNEL:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_tosa_compile_spec("TOSA-1.0+INT", **kwargs)),
                [global_int8_per_channel],
            )
        case ArmRecipeType.ETHOSU_U55_INT8_STATIC_PER_CHANNEL:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_u55_compile_spec(**kwargs)),
                [global_int8_per_channel],
            )
        case ArmRecipeType.ETHOSU_U55_INT8_STATIC_PER_TENSOR:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_u55_compile_spec(**kwargs)),
                [global_int8_per_tensor],
            )
        case ArmRecipeType.ETHOSU_U85_INT8_STATIC_PER_TENSOR:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_u85_compile_spec(**kwargs)),
                [global_int8_per_tensor],
            )
        case ArmRecipeType.ETHOSU_U85_INT8_STATIC_PER_CHANNEL:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_u85_compile_spec(**kwargs)),
                [global_int8_per_channel],
            )

        case ArmRecipeType.VGF_FP:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_vgf_compile_spec("TOSA-1.0+FP", **kwargs)),
            )
        case ArmRecipeType.VGF_INT8_STATIC_PER_TENSOR:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_vgf_compile_spec("TOSA-1.0+INT", **kwargs)),
                [global_int8_per_tensor],
            )
        case ArmRecipeType.VGF_INT8_STATIC_PER_CHANNEL:
            return ArmRecipeProvider.build_export_recipe(
                recipe_type,
                TargetRecipe(common.get_vgf_compile_spec("TOSA-1.0+INT", **kwargs)),
                [global_int8_per_channel],
            )
        case ArmRecipeType.CUSTOM:
            if "recipe" not in kwargs or not isinstance(
                kwargs["recipe"], ArmExportRecipe
            ):
                raise ValueError(
                    "ArmRecipeType.CUSTOM requires a kwarg 'recipe' that provides the ArmExportRecipe"
                )
            return kwargs["recipe"]
        case _:
            raise ValueError(f"Unsupported recipe type {recipe_type}")
