# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import dataclasses
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
from executorch.exir import EdgeProgramManager, ExportedProgram

from executorch.exir._warnings import experimental

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

    This class holds the configuration parameters for quantizing a model, supporting
    both post-training quantization (PTQ) and quantization-aware training (QAT).

    Attributes:
        quantizers: Optional list of quantizers for model quantization.
        ao_quantization_configs: Optional list of AOQuantizationConfig objects that pair
                                 AOBaseConfig with optional filter functions.
        is_qat: If True, use the QAT flow (prepare_qat_pt2e -> train_fn -> convert_pt2e).
                If False (default), use the PTQ flow (prepare_pt2e -> calibrate -> convert_pt2e).
        calibration_inputs_fn: Optional callable returning an iterable of input tuples used for
                               PTQ calibration. When None (default), the example inputs are used.
                               Ignored when is_qat=True.
        train_fn: Callable that receives the prepared GraphModule and trains it.
                  Required when is_qat=True; ignored otherwise.
        pre_prepare_passes: Optional list of callables applied to the captured GraphModule
                            before prepare_pt2e / prepare_qat_pt2e.
                            Each callable receives a GraphModule and must return a GraphModule.
        post_prepare_passes: Optional list of callables applied to the prepared GraphModule
                             after prepare_pt2e / prepare_qat_pt2e and before calibration / training.
                             Each callable receives a GraphModule and must return a GraphModule.
        pre_convert_passes: Optional list of callables applied to the GraphModule after
                            calibration (PTQ) or training (QAT) and before convert_pt2e.
                            Each callable receives a GraphModule and must return a GraphModule.
        post_convert_passes: Optional list of callables applied to the GraphModule after convert_pt2e.
                             Each callable receives a GraphModule and must return a GraphModule.
    """

    quantizers: Optional[List[Quantizer]] = None
    ao_quantization_configs: Optional[List[AOQuantizationConfig]] = None
    is_qat: bool = False
    calibration_inputs_fn: Optional[Callable[[], Iterable[tuple]]] = None
    train_fn: Optional[Callable[["torch.fx.GraphModule"], None]] = None
    pre_prepare_passes: Optional[
        List[Callable[["torch.fx.GraphModule"], "torch.fx.GraphModule"]]
    ] = None
    post_prepare_passes: Optional[
        List[Callable[["torch.fx.GraphModule"], "torch.fx.GraphModule"]]
    ] = None
    pre_convert_passes: Optional[
        List[Callable[["torch.fx.GraphModule"], "torch.fx.GraphModule"]]
    ] = None
    post_convert_passes: Optional[
        List[Callable[["torch.fx.GraphModule"], "torch.fx.GraphModule"]]
    ] = None

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
        edge_manager_transform_passes: Optional list of callables that take EdgeProgramManager as argument
                                        and return passes to be applied. Applied sequentially after TO_EDGE stage.
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

        # Scalar fields that must be identical across all recipes.
        def _assert_agree(field_name: str, values: list) -> None:
            unique = set(values)
            if len(unique) > 1:
                raise ValueError(
                    f"Cannot combine recipes with conflicting '{field_name}' values: {unique}"
                )

        # Collect all components.
        all_partitioners: list = []
        all_partitioners_by_method = {}
        all_quantizers: list = []
        all_ao_quantization_configs: list = []
        all_pre_edge_passes: list = []
        all_edge_transform_passes: list = []
        all_edge_manager_transform_passes: list = []
        all_pre_prepare_passes: list = []
        all_post_prepare_passes: list = []
        all_pre_convert_passes: list = []
        all_post_convert_passes: list = []
        combined_backend_config = None

        is_qat_values: list = []
        train_fn_values: list = []
        calibration_inputs_fn_values: list = []
        strict_values: list = []
        mode_values: list = []
        pipeline_stages_values: list = []
        source_transform_in_place_values: list = []

        for recipe in backend_recipes:
            if recipe.aten_transform_passes:
                all_pre_edge_passes.extend(recipe.aten_transform_passes)

            if lr := recipe.lowering_recipe:
                if lr.partitioners:
                    if isinstance(lr.partitioners, dict):
                        for method_name, method_partitioners in lr.partitioners.items():
                            all_partitioners_by_method.setdefault(method_name, []).extend(
                                method_partitioners
                            )
                    else:
                        all_partitioners.extend(lr.partitioners)
                if lr.edge_transform_passes:
                    all_edge_transform_passes.extend(lr.edge_transform_passes)
                if lr.edge_manager_transform_passes:
                    all_edge_manager_transform_passes.extend(
                        lr.edge_manager_transform_passes
                    )

            if qr := recipe.quantization_recipe:
                if qr.quantizers:
                    all_quantizers.extend(qr.quantizers)
                if qr.ao_quantization_configs:
                    all_ao_quantization_configs.extend(qr.ao_quantization_configs)
                is_qat_values.append(qr.is_qat)
                train_fn_values.append(qr.train_fn)
                calibration_inputs_fn_values.append(qr.calibration_inputs_fn)
                if qr.pre_prepare_passes:
                    all_pre_prepare_passes.extend(qr.pre_prepare_passes)
                if qr.post_prepare_passes:
                    all_post_prepare_passes.extend(qr.post_prepare_passes)
                if qr.pre_convert_passes:
                    all_pre_convert_passes.extend(qr.pre_convert_passes)
                if qr.post_convert_passes:
                    all_post_convert_passes.extend(qr.post_convert_passes)

            strict_values.append(recipe.strict)
            mode_values.append(recipe.mode)
            pipeline_stages_values.append(
                tuple(recipe.pipeline_stages) if recipe.pipeline_stages else None
            )
            source_transform_in_place_values.append(recipe.source_transform_in_place)

            # Use the first backend config as base
            if combined_backend_config is None and recipe.executorch_backend_config:
                combined_backend_config = copy.deepcopy(
                    recipe.executorch_backend_config
                )

        # Validate fields that must agree across all recipes.
        _assert_agree("strict", strict_values)
        _assert_agree("mode", mode_values)
        _assert_agree("pipeline_stages", pipeline_stages_values)
        _assert_agree("source_transform_in_place", source_transform_in_place_values)

        # is_qat must agree across all recipes that carry a QuantizationRecipe.
        _assert_agree("is_qat", is_qat_values)
        # train_fn must have at most one non-None value across all recipes.
        non_none_train_fns = [f for f in train_fn_values if f is not None]
        if len(non_none_train_fns) > 1:
            raise ValueError(
                "Cannot combine recipes: more than one recipe provides a train_fn."
            )
        # Multiple calibration_inputs_fn values are chained into a single factory.
        non_none_calib_fns = [f for f in calibration_inputs_fn_values if f is not None]
        if len(non_none_calib_fns) > 1:
            _fns = non_none_calib_fns

            def _combined_calib_fn():
                for _fn in _fns:
                    yield from _fn()

            combined_calib_fn = _combined_calib_fn
        else:
            combined_calib_fn = non_none_calib_fns[0] if non_none_calib_fns else None

        # Build combined QuantizationRecipe.
        combined_quantization_recipe = None
        if (
            all_quantizers
            or all_ao_quantization_configs
            or all_pre_prepare_passes
            or all_post_prepare_passes
            or all_pre_convert_passes
            or all_post_convert_passes
        ):
            combined_quantization_recipe = QuantizationRecipe(
                quantizers=all_quantizers or None,
                ao_quantization_configs=all_ao_quantization_configs or None,
                is_qat=is_qat_values[0] if is_qat_values else False,
                train_fn=non_none_train_fns[0] if non_none_train_fns else None,
                calibration_inputs_fn=combined_calib_fn,
                pre_prepare_passes=all_pre_prepare_passes or None,
                post_prepare_passes=all_post_prepare_passes or None,
                pre_convert_passes=all_pre_convert_passes or None,
                post_convert_passes=all_post_convert_passes or None,
            )

        if all_partitioners and all_partitioners_by_method:
            raise ValueError(
                "Cannot combine recipes that mix list-valued and dict-valued "
                "partitioners; convert the list-valued recipe to per-method form."
            )
        combined_partitioners = all_partitioners_by_method or all_partitioners

        # By value, not identity: every provider builds a fresh config object,
        # so asking for the same thing twice is not a conflict.
        distinct: List[tuple[str, EdgeCompileConfig]] = []
        for i, recipe in enumerate(backend_recipes):
            config = (
                recipe.lowering_recipe.edge_compile_config
                if recipe.lowering_recipe
                else None
            )
            if config is None or any(
                _edge_compile_configs_agree(config, seen) for _, seen in distinct
            ):
                continue
            distinct.append((recipe.name or f"recipes[{i}]", config))

        if len(distinct) > 1:
            raise ValueError(
                "Cannot combine recipes whose edge_compile_configs disagree on "
                + _edge_compile_config_conflict(distinct)
            )
        edge_compile_config = copy.deepcopy(distinct[0][1]) if distinct else None

        combined_lowering_recipe = None
        if (
            combined_partitioners
            or all_edge_transform_passes
            or all_edge_manager_transform_passes
            or edge_compile_config
        ):
            edge_compile_config = None
            for recipe in backend_recipes:
                if (
                    recipe.lowering_recipe
                    and recipe.lowering_recipe.edge_compile_config
                ):
                    edge_compile_config = recipe.lowering_recipe.edge_compile_config
                    break

            combined_lowering_recipe = LoweringRecipe(
                partitioners=combined_partitioners or None,
                edge_transform_passes=all_edge_transform_passes or None,
                edge_manager_transform_passes=all_edge_manager_transform_passes or None,
                edge_compile_config=edge_compile_config or EdgeCompileConfig(),
            )

        recipe_name = recipe_name or "_".join(
            [r.name for r in backend_recipes if r.name is not None]
        )
        return cls(
            name=recipe_name,
            quantization_recipe=combined_quantization_recipe,
            aten_transform_passes=all_pre_edge_passes or None,
            lowering_recipe=combined_lowering_recipe,
            executorch_backend_config=combined_backend_config,
            strict=strict_values[0] if strict_values else True,
            mode=mode_values[0] if mode_values else Mode.RELEASE,
            pipeline_stages=(
                list(pipeline_stages_values[0])
                if pipeline_stages_values and pipeline_stages_values[0] is not None
                else None
            ),
            source_transform_in_place=(
                source_transform_in_place_values[0]
                if source_transform_in_place_values
                else False
            ),
        )
