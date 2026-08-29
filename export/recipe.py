# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import dataclasses
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Callable, Dict, List, Optional, Union

import torch
from executorch.exir import EdgeProgramManager, ExportedProgram

from executorch.exir._warnings import experimental

from executorch.exir.backend.op_backend import OpBackend
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.pass_manager import PassManager, PassType
from torchao.core.config import AOBaseConfig
from torchao.quantization.pt2e.quantizer import Quantizer

from .types import StageType


"""
Export recipe definitions for ExecuTorch.

This module provides the data structures needed to configure the export process
for ExecuTorch models, including export configurations and quantization recipes.
"""


# Operator lists say which ops to keep, not in what order.
_UNORDERED_EDGE_CONFIG_FIELDS = ("preserve_ops", "_core_aten_ops_exception_list")


def _edge_config_key(config: EdgeCompileConfig) -> tuple:
    return tuple(
        (
            frozenset(getattr(config, f.name) or [])
            if f.name in _UNORDERED_EDGE_CONFIG_FIELDS
            else getattr(config, f.name)
        )
        for f in dataclasses.fields(config)
    )


def _edge_compile_configs_agree(
    left: EdgeCompileConfig, right: EdgeCompileConfig
) -> bool:
    return left is right or _edge_config_key(left) == _edge_config_key(right)


def _edge_compile_config_conflict(configs: List[tuple[str, EdgeCompileConfig]]) -> str:
    """Report only the fields the configs actually disagree on."""

    def render(config: EdgeCompileConfig, name: str) -> str:
        value = getattr(config, name)
        if name in _UNORDERED_EDGE_CONFIG_FIELDS:
            return str(sorted(str(op) for op in value or []))
        return str(value)

    keys = [_edge_config_key(config) for _, config in configs]
    return "; ".join(
        f"{field.name} ("
        + ", ".join(f"{name}={render(config, field.name)}" for name, config in configs)
        + ")"
        for i, field in enumerate(dataclasses.fields(configs[0][1]))
        if any(key[i] != keys[0][i] for key in keys[1:])
    )


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
        partitioners: Optional partitioners for model partitioning. Either a list
                      applied to every method, or a dict mapping method names to
                      per-method partitioner lists. Use the dict form when
                      backends need per-method compile specs.
        edge_transform_passes: Optional list of callables that take (method_name: str, exported_program: ExportedProgram)
                               and return either List[PassType] or PassManager to be applied during edge lowering.
        op_backends: Optional list of OpBackends, which lower by rewriting
                               operators rather than by delegating a subgraph. Run in
                               their own stage after partitioning, so they see what
                               the delegates left behind.
        edge_manager_transform_passes: Optional list of callables handed the whole
                               EdgeProgramManager, which return passes to apply to it.
        edge_compile_config: Optional edge compilation configuration
    """

    partitioners: Optional[Union[List[Partitioner], Dict[str, List[Partitioner]]]] = (
        None
    )
    edge_transform_passes: (
        None | List[Callable[[str, ExportedProgram], List[PassType] | PassManager]]
    ) = None
    # pyre-ignore[11]: Type not defined
    edge_manager_transform_passes: (
        None | List[Callable[[EdgeProgramManager], List[PassType] | PassManager]]
    ) = None
    # pyre-ignore[11]: Type not defined
    edge_compile_config: Optional[EdgeCompileConfig] = None
    # pyre-ignore[11]: Type not defined
    op_backends: Optional[List[OpBackend]] = None


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
        source_transform_in_place: Skip the defensive deepcopy in the SOURCE_TRANSFORM
                               stage and mutate the caller's model. Necessary for models
                               large enough that a second copy will not fit in memory.
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
    source_transform_in_place: bool = False
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

        How the edge_compile_configs are resolved::

            collect each recipe's edge_compile_config, deduped by value
               |
               |- two recipes partition, configs disagree ------> refuse
               |     no precedence between two delegates
               |
               |- configs all agree (0 or 1 distinct) ---------> use it
               |
               `- configs disagree --|- none partitions -------> refuse
                                     |     nothing to appeal to
                                     |
                                     `- one partitions --------> the delegate's
                                           config wins, with a warning naming
                                           the fields it overrode -- except
                                           preserve_ops, which it only takes
                                           when it filled the field itself

            finally, if any op_backends: _check_ir_validity = False
                                         (not a decomposition decision)

        A delegate outranks an operator backend because it partitions first and
        the backend lowers what it declined. `preserve_ops` is the exception:
        a delegate states what it needs kept whole through its partitioner's
        `ops_to_not_decompose`, so an empty field here is no opinion rather
        than a request to decompose.

        Operator backends are ordered so a delegate's own run last: theirs tidy
        up the boundary the partitioner left, which the others' passes still
        need intact.

        Args:
            backend_recipes: List of ExportRecipe objects to combine
            recipe_name: Optional name for the combined recipe

        Returns:
            Combined ExportRecipe for multi-backend deployment
        """
        overriding = [
            r.name or f"recipes[{i}]"
            for i, r in enumerate(backend_recipes)
            if r.pipeline_stages
        ]
        if overriding:
            raise ValueError(
                "Cannot combine recipes that override pipeline_stages, there is no "
                f"correct way to merge the orderings: {overriding}"
            )

        # Extract components from individual recipes
        all_partitioners = []
        all_partitioners_by_method = {}
        all_quantizers = []
        all_ao_quantization_configs = []
        all_pre_edge_passes = []
        all_transform_passes = []
        all_op_backends = []
        delegate_op_backends = []
        combined_backend_config = None

        for recipe in backend_recipes:
            # Collect pre-edge transform passes
            if recipe.aten_transform_passes:
                all_pre_edge_passes.extend(recipe.aten_transform_passes)

            # Collect partitioners from lowering recipes
            if recipe.lowering_recipe and recipe.lowering_recipe.partitioners:
                partitioners = recipe.lowering_recipe.partitioners
                if isinstance(partitioners, dict):
                    for method_name, method_partitioners in partitioners.items():
                        all_partitioners_by_method.setdefault(method_name, []).extend(
                            method_partitioners
                        )
                else:
                    all_partitioners.extend(partitioners)

            # Collect transform passes from lowering recipes
            if recipe.lowering_recipe and recipe.lowering_recipe.edge_transform_passes:
                all_transform_passes.extend(
                    recipe.lowering_recipe.edge_transform_passes
                )

            if recipe.lowering_recipe and recipe.lowering_recipe.op_backends:
                # Partitioning is the proxy for "tidies up its own boundary",
                # which has to happen after the main lowering.
                target = (
                    delegate_op_backends
                    if recipe.lowering_recipe.partitioners
                    else all_op_backends
                )
                target.extend(recipe.lowering_recipe.op_backends)

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

        if all_partitioners and all_partitioners_by_method:
            raise ValueError(
                "Cannot combine recipes that mix list-valued and dict-valued "
                "partitioners; convert the list-valued recipe to per-method form."
            )
        combined_partitioners = all_partitioners_by_method or all_partitioners

        all_op_backends.extend(delegate_op_backends)

        # By value, not identity: every provider builds a fresh config object,
        # so asking for the same thing twice is not a conflict.
        distinct: List[tuple[str, EdgeCompileConfig]] = []
        delegating: Optional[tuple[str, EdgeCompileConfig]] = None
        for i, recipe in enumerate(backend_recipes):
            lowering = recipe.lowering_recipe
            config = lowering.edge_compile_config if lowering else None
            if config is None:
                continue
            named = (recipe.name or f"recipes[{i}]", config)
            if lowering and lowering.partitioners:
                if delegating is not None and not _edge_compile_configs_agree(
                    config, delegating[1]
                ):
                    raise ValueError(
                        "Cannot combine recipes: "
                        f"'{delegating[0]}' and '{named[0]}' both partition and "
                        "their edge_compile_configs disagree on "
                        + _edge_compile_config_conflict([delegating, named])
                        + ". Precedence is only defined between a delegate and "
                        "an operator backend, not between two delegates."
                    )
                delegating = named
            if not any(
                _edge_compile_configs_agree(config, seen) for _, seen in distinct
            ):
                distinct.append(named)

        if len(distinct) > 1 and delegating is None:
            raise ValueError(
                "Cannot combine recipes whose edge_compile_configs disagree on "
                + _edge_compile_config_conflict(distinct)
            )

        edge_compile_config = None
        if delegating is not None and len(distinct) > 1:
            name, config = delegating
            others = [cfg for other, cfg in distinct if other != name]
            # Empty means no opinion, not "decompose everything": a delegate
            # asks through its partitioner's ops_to_not_decompose instead.
            if not config.preserve_ops:
                inherited = list(
                    dict.fromkeys(op for cfg in others for op in cfg.preserve_ops or [])
                )
                config = dataclasses.replace(config, preserve_ops=inherited)

            overridden = sorted(
                {
                    field.name
                    for cfg in others
                    for field in dataclasses.fields(cfg)
                    if getattr(cfg, field.name) != getattr(config, field.name)
                }
            )
            if overridden:
                logging.warning(
                    "Combining with '%s', which partitions: where the recipes "
                    "disagree its edge_compile_config wins, overriding %s "
                    "requested by the others.",
                    name,
                    overridden,
                )
            edge_compile_config = copy.deepcopy(config)
        elif distinct:
            edge_compile_config = copy.deepcopy(distinct[0][1])

        # Not redundant with EdgeProgramManager.to_op_backend, which clears the
        # flag on the manager it rebuilds: the verifier to_edge gave the program
        # fires earlier, inside the backend's own `_transform`.
        if all_op_backends:
            edge_compile_config = edge_compile_config or EdgeCompileConfig()
            edge_compile_config._check_ir_validity = False

        combined_lowering_recipe = None
        if (
            combined_partitioners
            or all_transform_passes
            or all_op_backends
            or edge_compile_config
        ):
            combined_lowering_recipe = LoweringRecipe(
                partitioners=combined_partitioners if combined_partitioners else None,
                edge_transform_passes=(
                    all_transform_passes if all_transform_passes else None
                ),
                op_backends=(all_op_backends if all_op_backends else None),
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
