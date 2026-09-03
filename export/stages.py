# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExportedProgram
from executorch.exir.backend.backend_api import validation_disabled
from executorch.exir.program import to_edge, to_edge_transform_and_lower
from executorch.export.recipe import LoweringRecipe, QuantizationRecipe
from executorch.export.types import StageType
from torch import nn
from torch._export.pass_base import PassType
from torch.fx.passes.infra.pass_manager import PassManager as GraphModulePassManager
from torchao.quantization import quantize_
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    Quantizer as TorchAOPT2EQuantizer,
)
from torchao.utils import unwrap_tensor_subclass


def _drop_empty(
    passes_by_method: Dict[str, List[PassType]]
) -> Dict[str, List[PassType]]:
    return {method: p for method, p in passes_by_method.items() if p}


class PipelineArtifact:
    def __init__(
        self,
        data: Any,
        context: Dict[str, Any],
    ) -> None:
        self.data = data
        self.context = context

    def add_context(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def copy_with_new_data(self, new_data: Any) -> "PipelineArtifact":
        return PipelineArtifact(data=new_data, context=self.context.copy())


class Stage(ABC):
    """
    Interface for a Stage in the ExecuTorch export pipeline.

    Each stage can be connected to other stages to form a pipeline.
    Each stage implements its own run method with specific parameter names.
    """

    def __init__(self) -> None:
        """
        Initialize the stage.
        """
        self._artifact = None

    @property
    @abstractmethod
    def stage_type(self) -> "StageType":
        """
        Returns the type of this stage.
        """
        pass

    @property
    @abstractmethod
    def valid_predecessor_stages(self) -> List["StageType"]:
        """
        Returns the list of stage types that can come before this stage.
        """
        pass

    @property
    @abstractmethod
    def can_start_pipeline(self) -> bool:
        """
        Returns whether this stage can be the first stage in a pipeline.
        """
        pass

    @abstractmethod
    def run(self, artifact: PipelineArtifact) -> None:
        """
        Executes this stage with the given inputs.

        Each concrete stage class implements this method with specific parameter names.
        """
        pass

    def get_artifacts(self) -> "PipelineArtifact":
        if self._artifact is None:
            raise RuntimeError(f"Stage: {self.__class__.__name__} not executed")
        return self._artifact


class TorchExportStage(Stage):
    """
    Purpose: Export PyTorch model to ExportedProgram.
    """

    def __init__(
        self,
        aten_transform_passes: Optional[
            List[Callable[[str, ExportedProgram], ExportedProgram]]
        ] = None,
        strict=True,
    ) -> None:
        super().__init__()
        self._aten_transform_passes = aten_transform_passes
        self.strict = strict

    @property
    def stage_type(self) -> str:
        return StageType.TORCH_EXPORT

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [StageType.SOURCE_TRANSFORM, StageType.QUANTIZE]

    @property
    def can_start_pipeline(self) -> bool:
        return True

    def run(self, artifact: PipelineArtifact) -> None:
        models = artifact.data
        example_inputs = artifact.get_context("example_inputs")
        dynamic_shapes = artifact.get_context("dynamic_shapes", {})

        exported_programs = {}

        with torch.no_grad():
            for method_name, model in models.items():
                if method_name not in example_inputs:
                    raise ValueError(
                        f"Example inputs for method {method_name} not found."
                    )

                method_dynamic_shapes = dynamic_shapes.get(method_name)

                # Export the model
                exported_programs[method_name] = torch.export.export(
                    model,
                    example_inputs[method_name][0],
                    dynamic_shapes=method_dynamic_shapes,
                    strict=self.strict,
                )

                # Apply pre-edge transform passes if available
                for pass_ in self._aten_transform_passes or []:
                    if not callable(pass_):
                        raise ValueError(
                            "Aten transform passes must be a callable that can transform and return an exported program"
                        )
                    exported_programs[method_name] = pass_(
                        method_name, exported_programs[method_name]
                    )

        self._artifact = artifact.copy_with_new_data(exported_programs)


def _collect_delegation_info(
    edge_program_manager: EdgeProgramManager,
) -> Dict[str, Any]:
    """
    Delegation info for every method, keyed by method name.

    `EdgeProgramManager.exported_program()` defaults to `forward`, which raises
    KeyError for a multi-method program that has no method by that name.
    """
    return {
        name: get_delegation_info(
            edge_program_manager.exported_program(name).graph_module
        )
        for name in sorted(edge_program_manager.methods)
    }


def _add_delegation_info_context(
    artifact: PipelineArtifact, edge_program_manager: EdgeProgramManager
) -> None:
    by_method = _collect_delegation_info(edge_program_manager)
    artifact.add_context("delegation_info_by_method", by_method)
    # `forward` when present, else the first method by name, so that
    # single-method callers keep seeing the value they always have.
    artifact.add_context(
        "delegation_info",
        by_method.get("forward", next(iter(by_method.values()), None)),
    )


class EdgeTransformAndLowerStage(Stage):
    """
    Second stage: Transform and lower to EdgeProgramManager.
    """

    def __init__(
        self,
        partitioners: Optional[Union[List[Any], Dict[str, List[Any]]]] = None,
        transform_passes: (
            None
            | List[
                Callable[
                    [str, ExportedProgram], List[PassType] | GraphModulePassManager
                ]
            ]
        ) = None,
        compile_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._partitioners = partitioners
        self._transform_passes = transform_passes
        self._compile_config = compile_config

    @classmethod
    def from_recipe(
        cls, lowering_recipe: Optional["LoweringRecipe"]
    ) -> "EdgeTransformAndLowerStage":
        if lowering_recipe is None:
            return cls()

        return cls(
            partitioners=lowering_recipe.partitioners,
            transform_passes=lowering_recipe.edge_transform_passes,
            compile_config=lowering_recipe.edge_compile_config,
        )

    @property
    def stage_type(self) -> str:
        return StageType.TO_EDGE_TRANSFORM_AND_LOWER

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [StageType.TORCH_EXPORT]

    @property
    def can_start_pipeline(self) -> bool:
        return True

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Transform and lower to EdgeProgramManager.
        """
        exported_programs = artifact.data
        constant_methods = artifact.get_context("constant_methods")
        generate_etrecord = artifact.get_context("generate_etrecord", False)

        # Detect if any callable returns PassManager
        pass_manager = None
        transform_passes = defaultdict(list)
        for method_name, ep in exported_programs.items():
            # Resolve transform passes from callable
            for pass_callable in self._transform_passes or []:
                if not callable(pass_callable):
                    raise ValueError(
                        "Transform passes must be a callable that resolves to passes"
                    )
                passes = pass_callable(method_name, ep)
                if isinstance(passes, GraphModulePassManager):
                    pass_manager = passes
                    break
                else:
                    transform_passes[method_name].extend(passes)
            if pass_manager:
                break

        # An empty dict is not no passes: EdgeProgramManager deep-copies every
        # method the dict does not name, so it would copy to apply nothing.
        final_passes = pass_manager or _drop_empty(transform_passes) or None

        with validation_disabled():
            edge_program_manager = to_edge_transform_and_lower(
                exported_programs,
                partitioner=self._partitioners,
                transform_passes=final_passes,
                constant_methods=constant_methods,
                compile_config=self._compile_config,
                generate_etrecord=generate_etrecord,
            )

        self._artifact = artifact.copy_with_new_data(edge_program_manager)
        _add_delegation_info_context(self._artifact, edge_program_manager)

    @property
    def delegation_info(self) -> Any:
        """
        Returns the delegation info.
        """
        return self._artifact.get_context("delegation_info")

    @property
    def delegation_info_by_method(self) -> Dict[str, Any]:
        """
        Returns the delegation info for every method, keyed by method name.
        """
        return self._artifact.get_context("delegation_info_by_method")


class ExecutorchStage(Stage):
    """
    Convert to ExecutorchProgramManager.
    """

    def __init__(self, backend_config: Any) -> None:
        self._backend_config = backend_config

    @property
    def stage_type(self) -> str:
        return StageType.TO_EXECUTORCH

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_BACKEND,
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,  # Added for server model generation (skipping TO_BACKEND)
        ]

    @property
    def can_start_pipeline(self) -> bool:
        return False

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Convert to ExecutorchProgramManager.
        """
        edge_program_manager = artifact.data

        # Process inputs
        if edge_program_manager is None:
            raise RuntimeError("Edge program manager is not set.")

        # Convert to ExecutorchProgramManager
        executorch_program_manager = edge_program_manager.to_executorch(
            self._backend_config
        )
        self._artifact = artifact.copy_with_new_data(executorch_program_manager)


class SourceTransformStage(Stage):
    """
    Optional stage: Source transform stage: Apply source transformations to the model.
    """

    def __init__(
        self,
        quantization_recipe: Optional[QuantizationRecipe],
        in_place: bool = False,
    ) -> None:
        self._quantization_recipe = quantization_recipe
        self._in_place = in_place
        self._transformed_models: Dict[str, nn.Module] = {}

    @property
    def stage_type(self) -> str:
        return StageType.SOURCE_TRANSFORM

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return []

    @property
    def can_start_pipeline(self) -> bool:
        return True

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Apply source transformations to the model.
        """
        if (
            not self._quantization_recipe
            or not self._quantization_recipe.ao_quantization_configs
        ):
            logging.info(
                "Quantization recipe is invalid to run SourceTransform, returning original artifact"
            )
            self._artifact = artifact
            return

        assert isinstance(artifact.data, dict)

        # A second copy of the model is not affordable for every caller, so
        # large models can opt out and have their own model mutated instead.
        self._transformed_models = (
            artifact.data if self._in_place else copy.deepcopy(artifact.data)
        )

        # Apply torchao quantize_ to each model
        for _, model in self._transformed_models.items():
            # pyre-ignore
            if len(self._quantization_recipe.ao_quantization_configs) > 1:
                raise ValueError(
                    "AO quantization configs cannot be reliably composed together, multiple quantization configs are disallowed for source transform at this point"
                )

            ao_config = self._quantization_recipe.ao_quantization_configs[0]
            quantize_(model, ao_config.ao_base_config, ao_config.filter_fn)
            unwrap_tensor_subclass(model)

        self._artifact = artifact.copy_with_new_data(self._transformed_models)


class QuantizeStage(Stage):
    """
    Optional stage: Perform post-training quantization on the model.
    """

    def __init__(self, quantization_recipe: Optional[QuantizationRecipe]) -> None:
        self._quantization_recipe = quantization_recipe

    @property
    def stage_type(self) -> str:
        return StageType.QUANTIZE

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [StageType.SOURCE_TRANSFORM]

    @property
    def can_start_pipeline(self) -> bool:
        return True

    def _get_quantizer_for_prepare_pt2e(self, quantizers: List[Any]):
        torch_ao_quantizers = []
        torchao_pt2e_quantizers = []

        for quantizer in quantizers:
            if isinstance(quantizer, TorchAOPT2EQuantizer):
                torchao_pt2e_quantizers.append(quantizer)
            else:
                # torch.ao quantizer support will soon be deprecated, remove this once CoreML moves to torchao quantizer
                logging.warning(
                    f"torch.ao quantizer {quantizer} is deprecated, consider moving to torchao quantizer"
                )
                torch_ao_quantizers.append(quantizer)

        if torch_ao_quantizers and torchao_pt2e_quantizers:
            raise ValueError("Mixed quantizer types are not supported")
        if len(torch_ao_quantizers) > 1:
            raise ValueError(
                "Multiple quantizers of torch.ao.quantization.quantizer not supported"
            )

        if torch_ao_quantizers:
            # prepare_pt2e has backward compat with torch.ao quantizer
            return torch_ao_quantizers[0]
        elif torchao_pt2e_quantizers:
            # Multiple torchao quantizers - use ComposableQuantizer
            return ComposableQuantizer(torchao_pt2e_quantizers)
        else:
            raise ValueError("No quantizers detected")

    @staticmethod
    def _apply_passes(
        model: "torch.fx.GraphModule",
        passes: Optional[List[Callable]],
    ) -> "torch.fx.GraphModule":
        for pass_fn in passes or []:
            try:
                model = pass_fn(model)
            except Exception as exc:
                raise RuntimeError(
                    f"QuantizeStage: Pass '{pass_fn!r}' raised an error: {exc}"
                ) from exc
        return model

    def run(self, artifact: PipelineArtifact) -> None:
        if not self._quantization_recipe or not self._quantization_recipe.quantizers:
            logging.info(
                "Quantization recipe is invalid to run QuantizeStage, returning original model"
            )
            self._artifact = artifact
            return

        assert isinstance(artifact.data, dict)

        recipe = self._quantization_recipe
        models = artifact.data
        example_inputs = artifact.get_context("example_inputs")

        quantized_models = {}

        for method_name, model in models.items():
            if method_name not in example_inputs or not example_inputs[method_name]:
                raise ValueError(
                    f"Example inputs for method {method_name} not found or empty."
                )

            inputs = example_inputs[method_name][0]

            # When dynamic_batch_size is requested, mark dimension 0 of every
            # tensor input as dynamic so that a QAT training loop can feed
            # mini-batches of arbitrary size through the prepared graph.
            export_dynamic_shapes = None
            if recipe.dynamic_batch_size:
                from torch.export import Dim

                batch = Dim("batch", min=1)
                export_dynamic_shapes = tuple(
                    {0: batch} if isinstance(t, torch.Tensor) else None for t in inputs
                )

            # QAT requires the model to be in training mode at capture time so
            # that batch_norm and dropout decompose with training-mode semantics.
            if recipe.is_qat:
                model.train()

            captured_graph = torch.export.export(
                model, inputs, dynamic_shapes=export_dynamic_shapes, strict=True
            ).module()

            # Pass 1: pre-prepare passes.
            captured_graph = self._apply_passes(
                captured_graph, recipe.pre_prepare_passes
            )

            quantizer = self._get_quantizer_for_prepare_pt2e(recipe.quantizers)

            if recipe.is_qat:
                if recipe.train_fn is None:
                    raise ValueError("train_fn must be provided when is_qat=True")
                prepared_model = prepare_qat_pt2e(captured_graph, quantizer)

                # Pass 2: post-prepare passes.
                prepared_model = self._apply_passes(
                    prepared_model, recipe.post_prepare_passes
                )

                allow_exported_model_train_eval(prepared_model)
                move_exported_model_to_train(prepared_model)
                recipe.train_fn(prepared_model)
                move_exported_model_to_eval(prepared_model)
            else:
                prepared_model = prepare_pt2e(captured_graph, quantizer)

                # Pass 2: post-prepare passes.
                prepared_model = self._apply_passes(
                    prepared_model, recipe.post_prepare_passes
                )

                # Use custom calibration inputs when provided; fall back to example inputs.
                if recipe.calibration_inputs_fn is not None:
                    calibration_inputs = recipe.calibration_inputs_fn()
                else:
                    calibration_inputs = example_inputs[method_name]

                for calibration_input in calibration_inputs:
                    prepared_model(*calibration_input)

            # Pass 3: pre-convert passes.
            prepared_model = self._apply_passes(
                prepared_model, recipe.pre_convert_passes
            )

            quantized_model = convert_pt2e(prepared_model)

            # Pass 4: post-convert passes.
            quantized_model = self._apply_passes(
                quantized_model, recipe.post_convert_passes
            )

            quantized_models[method_name] = quantized_model

        self._artifact = artifact.copy_with_new_data(quantized_models)


class ToEdgeStage(Stage):
    """
    Stage: Convert ExportedProgram to EdgeProgramManager.
    """

    def __init__(
        self,
        edge_compile_config: Optional[EdgeCompileConfig] = None,  # pyre-ignore
    ) -> None:
        super().__init__()
        self._edge_compile_config = edge_compile_config

    @classmethod
    def from_recipe(cls, lowering_recipe: Optional["LoweringRecipe"]) -> "ToEdgeStage":
        if lowering_recipe is None:
            return cls()

        return cls(
            edge_compile_config=lowering_recipe.edge_compile_config,
        )

    @property
    def stage_type(self) -> str:
        return StageType.TO_EDGE

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [StageType.TORCH_EXPORT]

    @property
    def can_start_pipeline(self) -> bool:
        return True

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Convert ExportedProgram to EdgeProgramManager.

        Args:
            artifact: Contains exported programs and context
        """
        exported_programs = artifact.data
        constant_methods = artifact.get_context("constant_methods")

        # Convert to edge program manager
        edge_program_manager = to_edge(
            exported_programs,
            constant_methods=constant_methods,
            compile_config=self._edge_compile_config,
            generate_etrecord=artifact.get_context("generate_etrecord", False),
        )

        self._artifact = artifact.copy_with_new_data(edge_program_manager)


class EdgeProgramManagerTransformStage(Stage):
    """
    Stage: Apply transformation passes that require EdgeProgramManager.

    This stage enables dynamic pass generation where passes need access to the
    EdgeProgramManager instance. Passes are applied sequentially, allowing
    to control order and dependencies between pass groups.
    """

    def __init__(
        self,
        edge_transform_passes: (
            None
            | List[
                Callable[
                    [str, ExportedProgram], List[PassType] | GraphModulePassManager
                ]
            ]
        ) = None,
        edge_manager_transform_passes: (
            None
            | List[
                Callable[[EdgeProgramManager], List[PassType] | GraphModulePassManager]
            ]
        ) = None,
    ) -> None:
        """
        Initialize the EdgeProgramManagerTransformStage.

        Args:
            edge_manager_transform_passes: List of callables that take EdgeProgramManager
                                           and return either List[PassType] or PassManager.
                                           Each callable is applied sequentially, allowing
                                           backends to control pass ordering and dependencies.
        """
        super().__init__()
        self._edge_transform_passes = edge_transform_passes or []
        self._edge_manager_transform_passes = edge_manager_transform_passes or []

    @classmethod
    def from_recipe(
        cls, lowering_recipe: Optional[LoweringRecipe]
    ) -> "EdgeProgramManagerTransformStage":
        if lowering_recipe is None:
            return cls()

        return cls(
            edge_transform_passes=lowering_recipe.edge_transform_passes,
            edge_manager_transform_passes=lowering_recipe.edge_manager_transform_passes,
        )

    @property
    def stage_type(self) -> str:
        return StageType.EDGE_PROGRAM_MANAGER_TRANSFORM

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [
            StageType.TO_EDGE,
            # StageType.TO_EDGE_TRANSFORM_AND_LOWER,  # TODO
        ]

    @property
    def can_start_pipeline(self) -> bool:
        return False

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Apply transformation passes sequentially.

        Args:
            artifact: Pipeline artifact containing EdgeProgramManager
        """
        edge_program_manager = artifact.data

        if not isinstance(edge_program_manager, EdgeProgramManager):
            raise TypeError(
                f"Expected EdgeProgramManager but got {type(edge_program_manager)}"
            )

        if not self._edge_transform_passes and not self._edge_manager_transform_passes:
            self._artifact = artifact
            return

        # Detect if any callable returns PassManager
        pass_manager = None
        transform_passes = defaultdict(list)
        for method_name in edge_program_manager.methods:
            # Resolve transform passes if it's a callable
            ep = edge_program_manager.exported_program(method_name)
            for pass_callable in self._edge_transform_passes or []:
                if not callable(pass_callable):
                    raise ValueError(
                        "Transform passes must be a callable that resolves to passes"
                    )
                passes = pass_callable(method_name, ep)
                if isinstance(passes, GraphModulePassManager):
                    pass_manager = passes
                    break
                else:
                    transform_passes[method_name].extend(passes)
            if pass_manager:
                break

        # See EdgeTransformAndLowerStage.run.
        final_passes = pass_manager or _drop_empty(transform_passes) or None
        if final_passes is not None:
            edge_program_manager = edge_program_manager.transform(final_passes)

        # Run edge manager transform passes
        for pass_callable in self._edge_manager_transform_passes:
            passes = pass_callable(edge_program_manager)
            if passes:
                edge_program_manager = edge_program_manager.transform(passes)

        self._artifact = artifact.copy_with_new_data(edge_program_manager)


class ToBackendStage(Stage):
    """
    Stage: Apply partitioning to EdgeProgramManager.
    """

    def __init__(
        self,
        partitioners: Optional[Union[List[Any], Dict[str, List[Any]]]] = None,
    ) -> None:
        super().__init__()
        self._partitioners = partitioners

    @classmethod
    def from_recipe(
        cls, lowering_recipe: Optional["LoweringRecipe"]
    ) -> "ToBackendStage":
        if lowering_recipe is None:
            return cls()

        return cls(
            partitioners=lowering_recipe.partitioners,
        )

    @property
    def stage_type(self) -> str:
        return StageType.TO_BACKEND

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [
            StageType.TO_EDGE,
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
        ]

    @property
    def can_start_pipeline(self) -> bool:
        return False

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Apply partitioning to EdgeProgramManager.

        Args:
            artifact: Contains edge program manager and context
        """
        edge_program_manager = artifact.data

        if edge_program_manager is None:
            raise RuntimeError("Edge program manager is not set.")

        # Apply partitioners if available
        if self._partitioners is not None and len(self._partitioners) > 0:
            with validation_disabled():
                if isinstance(self._partitioners, dict):
                    method_names = list(self._partitioners)
                    for partitioner_round in zip_longest(*self._partitioners.values()):
                        partitioners_by_method = {
                            method_name: partitioner
                            for method_name, partitioner in zip(
                                method_names, partitioner_round
                            )
                            if partitioner is not None
                        }
                        edge_program_manager = edge_program_manager.to_backend(
                            partitioners_by_method
                        )
                else:
                    # pyre-ignore
                    for partitioner in self._partitioners:
                        edge_program_manager = edge_program_manager.to_backend(
                            partitioner
                        )

        self._artifact = artifact.copy_with_new_data(edge_program_manager)
        _add_delegation_info_context(self._artifact, edge_program_manager)

    @property
    def delegation_info(self) -> Any:
        """
        Returns the delegation info.
        """
        return self._artifact.get_context("delegation_info")

    @property
    def delegation_info_by_method(self) -> Dict[str, Any]:
        """
        Returns the delegation info for every method, keyed by method name.
        """
        return self._artifact.get_context("delegation_info_by_method")
