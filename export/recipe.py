# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Callable, List, Optional, Sequence

import torch

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
        edge_transform_passes: Optional sequence of transformation passes to apply
        edge_compile_config: Optional edge compilation configuration
    """

    partitioners: Optional[List[Partitioner]] = None
    edge_transform_passes: Optional[Sequence[PassType]] = None
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
        pre_edge_transform_passes: Optional function to apply transformation passes
                                  before edge lowering
        lowering_recipe: Optional lowering recipe for model lowering and partitioning
        executorch_backend_config: Optional backend configuration for ExecuTorch
        pipeline_stages: Optional list of stages to execute, defaults to a standard pipeline.
        mode: Export mode (debug or release)
    """

    name: Optional[str] = None
    quantization_recipe: Optional[QuantizationRecipe] = None
    pre_edge_transform_passes: Optional[Sequence[PassType]] = None
    lowering_recipe: Optional[LoweringRecipe] = None
    # pyre-ignore[11]: Type not defined
    executorch_backend_config: Optional[ExecutorchBackendConfig] = None
    pipeline_stages: Optional[List[StageType]] = None
    mode: Mode = Mode.RELEASE

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
