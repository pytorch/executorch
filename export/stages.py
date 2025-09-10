# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import EdgeCompileConfig, ExportedProgram
from executorch.exir.backend.backend_api import validation_disabled
from executorch.exir.program import to_edge, to_edge_transform_and_lower
from executorch.export.recipe import LoweringRecipe, QuantizationRecipe
from executorch.export.types import StageType
from torch import nn
from torch._export.pass_base import PassType
from torchao.quantization import quantize_
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    Quantizer as TorchAOPT2EQuantizer,
)
from torchao.utils import unwrap_tensor_subclass


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
    ) -> None:
        super().__init__()
        self._aten_transform_passes = aten_transform_passes

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
                    strict=True,
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


class EdgeTransformAndLowerStage(Stage):
    """
    Second stage: Transform and lower to EdgeProgramManager.
    """

    def __init__(
        self,
        partitioners: Optional[List[Any]] = None,
        transform_passes: (
            None | List[Callable[[str, ExportedProgram], List[PassType]]]
        ) = None,
        compile_config: Optional[Any] = None,
    ) -> None:
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
        return False

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Transform and lower to EdgeProgramManager.
        """
        exported_programs = artifact.data
        constant_methods = artifact.get_context("constant_methods")
        generate_etrecord = artifact.get_context("generate_etrecord", False)

        # per method transform passes
        transform_passes = defaultdict(list)
        for method_name, ep in exported_programs.items():
            # Resolve transform passes from callable
            for pass_ in self._transform_passes or []:
                if not callable(pass_):
                    raise ValueError(
                        "Transform passes must be a callable that resolves to a list of passes"
                    )
                passes = pass_(method_name, ep)
                if isinstance(passes, list):
                    transform_passes[method_name].extend(passes)
                else:
                    raise ValueError(
                        "Transform passes must be a callable that resolves to a list of passes"
                    )

        with validation_disabled():
            edge_program_manager = to_edge_transform_and_lower(
                exported_programs,
                partitioner=self._partitioners,
                transform_passes=transform_passes,
                constant_methods=constant_methods,
                compile_config=self._compile_config,
                generate_etrecord=generate_etrecord,
            )

        delegation_info = get_delegation_info(
            edge_program_manager.exported_program().graph_module
        )
        self._artifact = artifact.copy_with_new_data(edge_program_manager)
        self._artifact.add_context("delegation_info", delegation_info)

    @property
    def delegation_info(self) -> Any:
        """
        Returns the delegation info.
        """
        return self._artifact.get_context("delegation_info")


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
        return [StageType.TO_EDGE_TRANSFORM_AND_LOWER, StageType.TO_BACKEND]

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

    def __init__(self, quantization_recipe: Optional[QuantizationRecipe]) -> None:
        self._quantization_recipe = quantization_recipe
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

        # Store the original models
        self._transformed_models = copy.deepcopy(artifact.data)

        # Apply torchao quantize_ to each model
        for _, model in artifact.data.items():
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

    def run(self, artifact: PipelineArtifact) -> None:
        if not self._quantization_recipe or not self._quantization_recipe.quantizers:
            logging.info(
                "Quantization recipe is invalid to run QunatizeStage, returning original model"
            )
            self._artifact = artifact
            return

        assert isinstance(artifact.data, dict)

        models = artifact.data
        example_inputs = artifact.get_context("example_inputs")

        quantized_models = {}

        for method_name, model in models.items():
            if method_name not in example_inputs or not example_inputs[method_name]:
                raise ValueError(
                    f"Example inputs for method {method_name} not found or empty."
                )

            inputs = example_inputs[method_name][0]
            captured_graph = torch.export.export(model, inputs, strict=True).module()

            quantizer = self._get_quantizer_for_prepare_pt2e(
                self._quantization_recipe.quantizers  # pyre-ignore
            )
            prepared_model = prepare_pt2e(captured_graph, quantizer)

            for calibration_input in example_inputs[method_name]:
                prepared_model(*calibration_input)

            quantized_model = convert_pt2e(prepared_model)
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
        return False

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


class ToBackendStage(Stage):
    """
    Stage: Apply transformations and partitioning to EdgeProgramManager.
    """

    def __init__(
        self,
        partitioners: Optional[List[Any]] = None,
        transform_passes: (
            None | List[Callable[[str, ExportedProgram], List[PassType]]]
        ) = None,
    ) -> None:
        super().__init__()
        self._partitioners = partitioners
        self._transform_passes = transform_passes

    @classmethod
    def from_recipe(
        cls, lowering_recipe: Optional["LoweringRecipe"]
    ) -> "ToBackendStage":
        if lowering_recipe is None:
            return cls()

        return cls(
            partitioners=lowering_recipe.partitioners,
            transform_passes=lowering_recipe.edge_transform_passes,
        )

    @property
    def stage_type(self) -> str:
        return StageType.TO_BACKEND

    @property
    def valid_predecessor_stages(self) -> List["StageType"]:
        return [StageType.TO_EDGE]

    @property
    def can_start_pipeline(self) -> bool:
        return False

    def run(self, artifact: PipelineArtifact) -> None:
        """
        Apply transformations and partitioning to EdgeProgramManager.

        Args:
            artifact: Contains edge program manager and context
        """
        edge_program_manager = artifact.data

        if edge_program_manager is None:
            raise RuntimeError("Edge program manager is not set.")

        # per method transform passes
        transform_passes = defaultdict(list)
        for method_name in edge_program_manager.methods:
            # Resolve transform passes if it's a callable
            ep = edge_program_manager.exported_program(method_name)
            for pass_ in self._transform_passes or []:
                if not callable(pass_):
                    raise ValueError(
                        "Transform passes must be a callable that resolves to a list of passes"
                    )
                passes = pass_(method_name, ep)
                if isinstance(passes, list):
                    transform_passes[method_name].extend(passes)
                else:
                    raise ValueError("Transform passes must return list of passes")

        # Apply transform passes
        edge_program_manager = edge_program_manager.transform(transform_passes)

        # Apply partitioners if available
        if self._partitioners is not None and len(self._partitioners) > 0:
            with validation_disabled():
                # pyre-ignore
                for partitioner in self._partitioners:
                    edge_program_manager = edge_program_manager.to_backend(partitioner)

        # Get delegation info
        delegation_info = get_delegation_info(
            edge_program_manager.exported_program().graph_module
        )

        self._artifact = artifact.copy_with_new_data(edge_program_manager)
        self._artifact.add_context("delegation_info", delegation_info)

    @property
    def delegation_info(self) -> Any:
        """
        Returns the delegation info.
        """
        return self._artifact.get_context("delegation_info")
