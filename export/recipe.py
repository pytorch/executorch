# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Callable, List, Optional

import torch
from executorch.exir import ExportedProgram

from executorch.exir._warnings import experimental

from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.pass_manager import PassType
from torchao.core.config import AOBaseConfig
from torchao.quantization.pt2e.quantizer import Quantizer

from .types import StageType


"""
Export recipe definitions for ExecuTorch.

This module provides the data structures needed to configure the export process
for ExecuTorch models, including export configurations and quantization recipes.
"""


class RecipeTypeMeta(EnumMeta, ABCMeta):
    """Metaclass that combines EnumMeta and ABCMeta"""

    pass


class RecipeType(Enum, metaclass=RecipeTypeMeta):
    """
    Base recipe type class that backends can extend to define their own recipe types.
    Backends should create their own enum classes that inherit from RecipeType:
    """

    @classmethod
    @abstractmethod
    def get_backend_name(cls) -> str:
        """
        Return the backend name for this recipe type.

        Returns:
            str: The backend name (e.g., "xnnpack", "qnn", etc.)
        """
        pass


class Mode(str, Enum):
    """
    Export mode enumeration.

    Attributes:
        DEBUG: Debug mode with additional checks and information
        RELEASE: Release mode optimized for performance
    """

    DEBUG = "debug"
    RELEASE = "release"


@dataclass
class AOQuantizationConfig:
    """
    Configuration for torchao quantization with optional filter function.

    Attributes:
        ao_base_config: The AOBaseConfig for quantization
        filter_fn: Optional filter function to selectively apply quantization
    """

    ao_base_config: AOBaseConfig
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None


@dataclass
class QuantizationRecipe:
    """
    Configuration recipe for quantization.

    This class holds the configuration parameters for quantizing a model.

    Attributes:
        quantizers: Optional list of quantizers for model quantization
        ao_quantization_configs: Optional list of AOQuantizationConfig objects that pair
                                 AOBaseConfig with optional filter functions
    """

    quantizers: Optional[List[Quantizer]] = None
    ao_quantization_configs: Optional[List[AOQuantizationConfig]] = None

    def get_quantizers(self) -> Optional[List[Quantizer]]:
        """
        Get the quantizers associated with this recipe.

        Returns:
            The quantizers if any are set, otherwise None
        """
        return self.quantizers


@dataclass
class LoweringRecipe:
    """
    Configuration recipe for lowering and partitioning.

    This class holds the configuration parameters for lowering a model
    to backend-specific representations.

    Attributes:
        partitioners: Optional list of partitioners for model partitioning
        edge_transform_passes: Optional list of callables that take (method_name: str, exported_program: ExportedProgram) as arguments
                               and return a list of passes (PassType) to be executed during lowering stages.
        edge_compile_config: Optional edge compilation configuration
    """

    partitioners: Optional[List[Partitioner]] = None
    edge_transform_passes: (
        None | List[Callable[[str, ExportedProgram], List[PassType]]]
    ) = None
    # pyre-ignore[11]: Type not defined
    edge_compile_config: Optional[EdgeCompileConfig] = None


@experimental(
    "This API and all of its related functionality such as ExportSession and ExportRecipe are experimental."
)
@dataclass
class ExportRecipe:
    """
    Configuration recipe for the export process.

    This class holds the configuration parameters for exporting a model,
    including compilation and transformation options.

    Attributes:
        name: Optional name for the recipe
        quantization_recipe: Optional quantization recipe for model quantization
        aten_transform_passes: Optional list of functions to apply transformation passes to the program before edge lowering.
                               These callables are invoked to modify and return the transformed program.
        lowering_recipe: Optional lowering recipe for model lowering and partitioning
        executorch_backend_config: Optional backend configuration for ExecuTorch
        pipeline_stages: Optional list of stages to execute, defaults to a standard pipeline.
        mode: Export mode (debug or release)
        strict: Set the strict flag in the torch export call.
    """

    name: Optional[str] = None
    quantization_recipe: Optional[QuantizationRecipe] = None
    aten_transform_passes: Optional[
        List[Callable[[str, ExportedProgram], ExportedProgram]]
    ] = None
    lowering_recipe: Optional[LoweringRecipe] = None
    # pyre-ignore[11]: Type not defined
    executorch_backend_config: Optional[ExecutorchBackendConfig] = None
    pipeline_stages: Optional[List[StageType]] = None
    mode: Mode = Mode.RELEASE
    strict: bool = True

    @classmethod
    def get_recipe(cls, recipe: "RecipeType", **kwargs) -> "ExportRecipe":
        """
        Get an export recipe from backend. Backend is automatically determined based on the
        passed recipe type.

        Args:
            recipe: The type of recipe to create
            **kwargs: Recipe-specific parameters

        Returns:
            ExportRecipe configured for the specified recipe type
        """
        from .recipe_registry import recipe_registry

        if not isinstance(recipe, RecipeType):
            raise ValueError(f"Invalid recipe type: {recipe}")

        backend = recipe.get_backend_name()
        export_recipe = recipe_registry.create_recipe(recipe, backend, **kwargs)
        if export_recipe is None:
            supported = recipe_registry.get_supported_recipes(backend)
            raise ValueError(
                f"Recipe '{recipe.value}' not supported by '{backend}'. "
                f"Supported: {[r.value for r in supported]}"
            )
        return export_recipe

    @classmethod
    def combine(
        cls, recipes: List["ExportRecipe"], recipe_name: Optional[str] = None
    ) -> "ExportRecipe":
        """
        Combine multiple ExportRecipe objects into a single recipe.

        Args:
            recipes: List of ExportRecipe objects to combine
            recipe_name: Optional name for the combined recipe

        Returns:
            A new ExportRecipe that combines all input recipes

        Example:
            recipe1 = ExportRecipe.get_recipe(CoreMLRecipeType.FP32)
            recipe2 = ExportRecipe.get_recipe(XNNPackRecipeType.FP32)
            combined_recipe = ExportRecipe.combine(
                [recipe1, recipe2],
                recipe_name="multi_backend_coreml_xnnpack_fp32"
            )
        """
        if not recipes:
            raise ValueError("Recipes cannot be empty")

        if len(recipes) == 1:
            return recipes[0]

        return cls._combine_recipes(recipes, recipe_name)

    @classmethod
    def _combine_recipes(  # noqa: C901
        cls, backend_recipes: List["ExportRecipe"], recipe_name: Optional[str] = None
    ) -> "ExportRecipe":
        """
        Util to combine multiple backend recipes into a single multi-backend recipe.

        Args:
            backend_recipes: List of ExportRecipe objects to combine
            recipe_name: Optional name for the combined recipe

        Returns:
            Combined ExportRecipe for multi-backend deployment
        """
        # Extract components from individual recipes
        all_partitioners = []
        all_quantizers = []
        all_ao_quantization_configs = []
        all_pre_edge_passes = []
        all_transform_passes = []
        combined_backend_config = None

        for recipe in backend_recipes:
            # Collect pre-edge transform passes
            if recipe.aten_transform_passes:
                all_pre_edge_passes.extend(recipe.aten_transform_passes)

            # Collect partitioners from lowering recipes
            if recipe.lowering_recipe and recipe.lowering_recipe.partitioners:
                all_partitioners.extend(recipe.lowering_recipe.partitioners)

            # Collect transform passes from lowering recipes
            if recipe.lowering_recipe and recipe.lowering_recipe.edge_transform_passes:
                all_transform_passes.extend(
                    recipe.lowering_recipe.edge_transform_passes
                )

            # Collect for quantize stage
            if quantization_recipe := recipe.quantization_recipe:
                # Collect PT2E quantizers
                if quantization_recipe.quantizers:
                    all_quantizers.extend(quantization_recipe.quantizers)

                # Collect source transform configs
                if quantization_recipe.ao_quantization_configs:
                    all_ao_quantization_configs.extend(
                        quantization_recipe.ao_quantization_configs
                    )

            # Use the first backend config as base
            if combined_backend_config is None and recipe.executorch_backend_config:
                combined_backend_config = copy.deepcopy(
                    recipe.executorch_backend_config
                )

        # Create combined quantization recipe
        combined_quantization_recipe = None
        if all_quantizers or all_ao_quantization_configs:
            combined_quantization_recipe = QuantizationRecipe(
                quantizers=all_quantizers if all_quantizers else None,
                ao_quantization_configs=(
                    all_ao_quantization_configs if all_ao_quantization_configs else None
                ),
            )

        # Create combined lowering recipe
        combined_lowering_recipe = None
        if all_partitioners or all_transform_passes:
            edge_compile_config = None
            for recipe in backend_recipes:
                if (
                    recipe.lowering_recipe
                    and recipe.lowering_recipe.edge_compile_config
                ):
                    edge_compile_config = recipe.lowering_recipe.edge_compile_config
                    break

            combined_lowering_recipe = LoweringRecipe(
                partitioners=all_partitioners if all_partitioners else None,
                edge_transform_passes=(
                    all_transform_passes if all_transform_passes else None
                ),
                edge_compile_config=edge_compile_config or EdgeCompileConfig(),
            )

        recipe_name = recipe_name or "_".join(
            [r.name for r in backend_recipes if r.name is not None]
        )
        return cls(
            name=recipe_name,
            quantization_recipe=combined_quantization_recipe,
            aten_transform_passes=all_pre_edge_passes,
            lowering_recipe=combined_lowering_recipe,
            executorch_backend_config=combined_backend_config,
        )
