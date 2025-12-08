# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from executorch.exir import EdgeProgramManager
from executorch.exir._warnings import experimental
from executorch.exir.program import ExecutorchProgramManager
from executorch.exir.schema import Program
from executorch.extension.export_util.utils import save_pte_program
from tabulate import tabulate
from torch import nn
from torch.export import ExportedProgram
from torch.fx import GraphModule

from .recipe import ExportRecipe, LoweringRecipe, QuantizationRecipe
from .stages import (
    EdgeTransformAndLowerStage,
    ExecutorchStage,
    PipelineArtifact,
    QuantizeStage,
    SourceTransformStage,
    Stage,
    ToBackendStage,
    ToEdgeStage,
    TorchExportStage,
)
from .types import StageType


@experimental(
    "This API and all of its related functionality such as ExportSession and ExportRecipe are experimental."
)
def export(
    model: Union[
        nn.Module,
        Dict[str, nn.Module],
        GraphModule,
        Dict[str, GraphModule],
        ExportedProgram,
        Dict[str, ExportedProgram],
        str,
    ],
    example_inputs: Optional[
        Union[
            List[tuple[torch.Tensor, ...]],
            Dict[str, List[tuple[torch.Tensor, ...]]],
        ]
    ] = None,
    export_recipe: ExportRecipe = None,
    name: Optional[str] = None,
    dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
    constant_methods: Optional[Union[Dict[str, Callable]]] = None,
    artifact_dir: Optional[str] = None,
    generate_etrecord: bool = False,
) -> "ExportSession":
    """
    Create and configure an ExportSession with the given parameters.

    This function provides a convenient way to create an ExportSession and
    optionally run the export process in one step.

    Args:
        model: The PyTorch model(s) to export. Can be:
              - nn.Module or Dict[str, nn.Module]: Eager PyTorch model(s)
              - GraphModule or Dict[str, GraphModule]: Quantized model(s) (e.g., from prepare/convert)
              - ExportedProgram or Dict[str, ExportedProgram]: Already exported model(s)
              - str: Path to load an ExportedProgram from disk
        example_inputs: Example inputs for the model(s), either a list of input tuples
                      or a dictionary mapping method names to lists of input tuples.
                      First sample (index 0) is used for torch.export.export() to export the model.
                      All samples are used as calibration dataset in PT2E Quantize stage.
                      Optional when model is ExportedProgram (not needed).
        export_recipe: Contains the configuration for the export process
        name: Optional name for the export
        dynamic_shapes: Optional dynamic shape specifications
        constant_methods: Optional dictionary of constant methods
        artifact_dir: Optional directory to store artifacts
        generate_etrecord: Optional flag to generate an etrecord

    Returns:
        A configured ExportSession instance with the export process completed if requested
    """
    session = ExportSession(
        model=model,
        example_inputs=example_inputs,
        export_recipe=export_recipe,
        name=name,
        dynamic_shapes=dynamic_shapes,
        constant_methods=constant_methods,
        artifact_dir=artifact_dir,
        generate_etrecord=generate_etrecord,
    )
    session.export()

    return session


@experimental(
    "This API and all of its related functionality such as ExportSession and ExportRecipe are experimental."
)
class ExportSession:
    """
    Manages the export process for ExecuTorch models.

    This class handles the export process through a pipeline of stages:
    1. (Optional) Quantize - Apply post-training quantization to the model
    2. Export - Export PyTorch model to ExportedProgram
    3. EdgeTransformAndLower - Transform and lower to EdgeProgramManager
    4. Executorch - Convert to ExecutorchProgramManager for final execution
    """

    def __init__(
        self,
        model: Union[
            nn.Module,
            Dict[str, nn.Module],
            GraphModule,
            Dict[str, GraphModule],
            ExportedProgram,
            Dict[str, ExportedProgram],
            str,
        ],
        example_inputs: Optional[
            Union[
                List[tuple[torch.Tensor, ...]],
                Dict[str, List[tuple[torch.Tensor, ...]]],
            ]
        ] = None,
        export_recipe: ExportRecipe = None,
        name: Optional[str] = None,
        dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
        constant_methods: Optional[Union[Dict[str, Callable]]] = None,
        artifact_dir: Optional[str] = None,
        generate_etrecord: Optional[bool] = False,
    ) -> None:
        """
        Initialize the ExportSession with model, inputs, and recipe.

        Args:
            model: The PyTorch model(s) to export. Can be:
                  - nn.Module or Dict[str, nn.Module]: Eager PyTorch model(s)
                  - GraphModule or Dict[str, GraphModule]: Quantized model(s)
                  - ExportedProgram or Dict[str, ExportedProgram]: Already exported model(s)
                  - str: Path to load an ExportedProgram from disk
            example_inputs: Example inputs for the model(s), either a list of input tuples
                          or a dictionary mapping method names to lists of input tuples.
                          First sample (index 0) is used for torch.export.export() to export the model.
                          All samples are used as calibration dataset in PT2E Quantize stage,
                          Optional when model is ExportedProgram (not needed).
            export_recipe: Contains the configuration for the export process
            name: Optional name for the export
            dynamic_shapes: Optional dynamic shape specifications
            constant_methods: Optional dictionary of constant methods
            artifact_dir: Optional directory to store artifacts
            generate_etrecord: Optional flag to generate an etrecord
        """
        # Load model from file if string path provided
        if isinstance(model, str):
            model = torch.export.load(model)
            logging.info(f"Loaded ExportedProgram from {model}")

        # Detect input model type to determine which stages to skip
        self._input_model_type = self._detect_model_type(model)

        # Standardize model to dictionary format
        self._model = model if isinstance(model, dict) else {"forward": model}

        # Validate and standardize example_inputs to dictionary format
        # example_inputs not required for ExportedProgram input
        if self._input_model_type == "ExportedProgram":
            self._example_inputs = example_inputs or {}
            if isinstance(self._example_inputs, list):
                self._example_inputs = {"forward": self._example_inputs}
        else:
            # For nn.Module and GraphModule, example_inputs are required
            if example_inputs is None:
                raise ValueError(
                    f"example_inputs are required when model is {self._input_model_type}. "
                    f"Only ExportedProgram inputs can omit example_inputs."
                )
            self._example_inputs = (
                example_inputs
                if isinstance(example_inputs, dict)
                else {"forward": example_inputs}
            )

        # Standardize dynamic_shapes to dictionary format
        self._dynamic_shapes = {}
        if dynamic_shapes is not None:
            if isinstance(dynamic_shapes, dict):
                self._dynamic_shapes = dynamic_shapes
            else:
                self._dynamic_shapes = {"forward": dynamic_shapes}

        self._export_recipe = export_recipe

        self._quant_recipe: Optional[QuantizationRecipe] = (
            self._export_recipe.quantization_recipe
        )

        self._lowering_recipe: Optional[LoweringRecipe] = (
            self._export_recipe.lowering_recipe
        )

        # Stages to run
        self._pipeline_stages = (
            self._export_recipe.pipeline_stages or self._get_default_pipeline()
        )

        # Stage registry: map of StageType to Stage instances
        self._stage_registry: Dict[StageType, Stage] = self._build_stages(
            self._pipeline_stages
        )

        # Intialize run context
        self._run_context: Dict[str, Any] = {
            "example_inputs": self._example_inputs,
            "dynamic_shapes": self._dynamic_shapes,
            "constant_methods": constant_methods,
            "export_recipe": self._export_recipe,
            "session_name": name,
            "artifact_dir": artifact_dir,
            "generate_etrecord": generate_etrecord,
        }

        self._stage_to_artifacts: Dict[StageType, PipelineArtifact] = {}

    def _detect_model_type(
        self, model: Union[nn.Module, GraphModule, ExportedProgram, Dict]
    ) -> str:
        """
        Detect the type of input model.

        Args:
            model: Input model in various formats

        Returns:
            String indicating the model type: "nn.Module", "GraphModule", or "ExportedProgram"
        """
        # Handle dict (multi-method) - check first value
        if isinstance(model, dict):
            first_value = next(iter(model.values()))
            return self._detect_model_type(first_value)

        # Detect single model type
        if isinstance(model, ExportedProgram):
            return "ExportedProgram"
        elif isinstance(model, GraphModule):
            return "GraphModule"
        elif isinstance(model, nn.Module):
            return "nn.Module"
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    def _get_default_pipeline(self) -> List[StageType]:
        """
        Get default pipeline stages based on input model type.

        Returns:
            List of stages appropriate for the input model type
        """
        stages = []

        # Add quantization stages only for eager nn.Module
        if self._input_model_type == "nn.Module":
            stages.extend(
                [
                    StageType.SOURCE_TRANSFORM,  # Optional stage, returns original model if quant recipe is invalid
                    StageType.QUANTIZE,  # Optional stage, returns original model if quant recipe is invalid
                ]
            )

        # Add torch export stage if not already exported
        if self._input_model_type != "ExportedProgram":
            stages.append(StageType.TORCH_EXPORT)

        # Always include edge and executorch stages
        stages.extend(
            [
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        return stages

    def _build_stages(self, stages: List[StageType]) -> Dict[StageType, Stage]:
        """Build the stage registry from the given stages."""
        stage_registry: Dict[StageType, Stage] = {}

        stage = None
        for stage_type in stages or self._get_default_pipeline():
            if stage_type == StageType.SOURCE_TRANSFORM:
                stage = SourceTransformStage(self._quant_recipe)
            elif stage_type == StageType.QUANTIZE:
                stage = QuantizeStage(self._quant_recipe)
            elif stage_type == StageType.TORCH_EXPORT:
                aten_transform_passes = None
                if self._export_recipe.aten_transform_passes is not None:
                    aten_transform_passes = list(
                        self._export_recipe.aten_transform_passes
                    )
                stage = TorchExportStage(
                    aten_transform_passes, strict=self._export_recipe.strict
                )
            elif stage_type == StageType.TO_EDGE_TRANSFORM_AND_LOWER:
                stage = EdgeTransformAndLowerStage.from_recipe(self._lowering_recipe)
            elif stage_type == StageType.TO_EDGE:
                stage = ToEdgeStage.from_recipe(self._lowering_recipe)
            elif stage_type == StageType.TO_BACKEND:
                stage = ToBackendStage.from_recipe(self._lowering_recipe)
            elif stage_type == StageType.TO_EXECUTORCH:
                stage = ExecutorchStage(self._export_recipe.executorch_backend_config)
            else:
                logging.info(
                    f"{stage_type} is unknown, you have to register it before executing export()"
                )

            if stage:
                stage_registry[stage_type] = stage
        return stage_registry

    def register_stage(self, stage_type: StageType, stage: Stage) -> None:
        """
        Register a new stage or override an existing stage implementation.

        Args:
            stage_type: The type of stage to register
            stage: The stage instance to register
        """
        self._stage_registry[stage_type] = stage

    def get_registered_stage(self, stage_type: StageType) -> Optional[Stage]:
        """
        Get a registered stage by its type.

        Args:
            stage_type: The type of stage to retrieve

        Returns:
            The registered stage instance, or None if not found
        """
        return self._stage_registry.get(stage_type)

    def get_all_registered_stages(self) -> Dict[StageType, Stage]:
        """
        Get all registered stages.

        Returns:
            Dictionary mapping stage types to stage instances
        """
        return self._stage_registry

    def _validate_pipeline_sequence(
        self,
        stages: List[StageType],
    ) -> None:
        if not stages:
            raise ValueError("Pipeline stages cannot be empty")

        # Validate pipeline compatibility with input model type
        if self._input_model_type == "GraphModule":
            # GraphModule input should not run quantization stages
            incompatible_stages = {StageType.SOURCE_TRANSFORM, StageType.QUANTIZE}
            found_incompatible = set(stages) & incompatible_stages
            if found_incompatible:
                stage_names = ", ".join(s.name for s in found_incompatible)
                raise ValueError(
                    f"Cannot run {stage_names} stage(s) with GraphModule input. "
                    f"GraphModule is already quantized. "
                    f"Remove {stage_names} from pipeline_stages or use nn.Module input."
                )
        elif self._input_model_type == "ExportedProgram":
            # ExportedProgram input should not run quantization or torch export stages
            incompatible_stages = {
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
            }
            found_incompatible = set(stages) & incompatible_stages
            if found_incompatible:
                stage_names = ", ".join(s.name for s in found_incompatible)
                raise ValueError(
                    f"Cannot run {stage_names} stage(s) with ExportedProgram input. "
                    f"ExportedProgram is already exported. "
                    f"Remove {stage_names} from pipeline_stages or use nn.Module/GraphModule input."
                )

        # Validate that the first stage can start a pipeline
        first_stage = stages[0]
        first_stage_instance = self._stage_registry.get(first_stage)
        if first_stage_instance is None:
            raise ValueError(
                f"Stage {first_stage} not found in registry, register it using session.register_stage()"
            )

        if not first_stage_instance.can_start_pipeline:
            raise ValueError(f"Stage {first_stage} cannot start a pipeline. ")

        # Validate stage transitions
        for i in range(1, len(stages)):
            current_stage = stages[i]
            previous_stage = stages[i - 1]

            # Get the stage instance to check its valid predecessors
            stage_instance = self._stage_registry.get(current_stage)
            if stage_instance is None:
                raise ValueError(
                    f"Stage {current_stage} not found in registry, , register it using session.register_stage()"
                )

            valid_predecessors = stage_instance.valid_predecessor_stages

            # Check if the previous stage is valid for the current stage
            if valid_predecessors and previous_stage not in valid_predecessors:
                raise ValueError(
                    f"Invalid transition from {previous_stage} to {current_stage}. "
                    f"Valid predecessors for {current_stage}: {valid_predecessors}"
                )

    def _run_pipeline(self) -> None:
        # Validate if given stage sequence is valid
        self._validate_pipeline_sequence(
            stages=self._pipeline_stages,
        )

        current_artifact = PipelineArtifact(data=self._model, context=self._run_context)

        # Execute stages from registry in the order specified by pipeline_stages
        for stage_type in self._pipeline_stages:
            stage = self._stage_registry.get(stage_type)
            if stage is None:
                raise ValueError(f"Stage {stage_type} not found in registry")

            logging.info(f"Executing stage: {stage_type}")

            start = time.perf_counter()
            stage.run(current_artifact)
            elapsed = (time.perf_counter() - start) * 1000
            current_artifact = stage.get_artifacts()
            current_artifact.add_context("duration_ms", int(elapsed))

            logging.info(f"Stage {stage_type} execution done")

            self._stage_to_artifacts[stage_type] = current_artifact

    def export(self) -> None:
        """
        Execute the full export process.

        This method orchestrates the export process with optional quantization:
        1. (Optional) Apply quantization to the model
        2. Export the PyTorch model to ExportedProgram
        3. Transform and lower to EdgeProgramManager
        4. Convert to ExecutorchProgramManager
        """
        # Run the pipeline from the beginning
        self._run_pipeline()

    def get_stage_artifacts(self) -> Dict[StageType, PipelineArtifact]:
        return self._stage_to_artifacts

    def get_exported_program(self, method_name: str = "forward") -> ExportedProgram:
        """
        Get the ExportedProgram for a specific method after torch export.

        Args:
            method_name: Name of the method to get exported program for, defaults to "forward"

        Returns:
            The ExportedProgram for the specified method

        Raises:
            RuntimeError: If torch export stage has not been run
            KeyError: If the method name is not found in exported programs
        """
        artifact = self._stage_to_artifacts.get(StageType.TORCH_EXPORT)
        if artifact is None or artifact.data is None:
            raise RuntimeError(
                "Exported program is not available. Run Torch Export Stage first."
            )

        exported_programs = artifact.data
        if method_name not in exported_programs:
            raise KeyError(
                f"Method name '{method_name}' not found in exported programs. "
                f"Available methods: {list(exported_programs.keys())}"
            )

        return exported_programs[method_name]

    def get_edge_program_manager(self) -> "EdgeProgramManager":
        """
        Get the EdgeProgramManager after edge lowering stages.

        This method checks multiple stages in order of preference:
        1. TO_EDGE_TRANSFORM_AND_LOWER (combined stage)
        2. TO_BACKEND (separate stage with backend delegation)
        3. TO_EDGE (separate stage without backend delegation)

        Returns:
            The EdgeProgramManager

        Raises:
            RuntimeError: If no edge stage has been run
        """
        # Check stages in order of preference
        for stage_type in [
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_BACKEND,
            StageType.TO_EDGE,
        ]:
            artifact = self._stage_to_artifacts.get(stage_type)
            if artifact is not None and artifact.data is not None:
                logging.info(f"Returning edge program manager from stage {stage_type}")
                return artifact.data

        raise RuntimeError(
            "Edge program manager is not available. "
            "Run one of the edge stages first: TO_EDGE_TRANSFORM_AND_LOWER, TO_EDGE, or TO_BACKEND."
        )

    def get_executorch_program(self) -> Program:
        """
        Get the ExecutorchProgram from the ExecutorchProgramManager.

        Returns:
            The ExecutorchProgram

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        return self.get_executorch_program_manager().executorch_program

    def get_executorch_program_manager(self) -> ExecutorchProgramManager:
        """
        Get the ExecutorchProgramManager.

        Returns:
            The ExecutorchProgramManager

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        artifact = self._stage_to_artifacts.get(StageType.TO_EXECUTORCH)
        if artifact is None or artifact.data is None:
            raise RuntimeError(
                "Executorch program manager is not initialized. Run Executorch Stage first."
            )
        return artifact.data

    def get_pte_buffer(self) -> bytes:
        """
        Get the PTE buffer as bytes.

        Returns:
            The PTE buffer as bytes

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        return self.get_executorch_program_manager().buffer

    def save_to_pte(self, output_name: str) -> None:
        """
        Save the model to a .pte file.

        Args:
            output_name (Optional[str]): The name of the .pte file.
        """
        assert output_name, "Need a valid output name"
        save_pte_program(self.get_executorch_program_manager(), output_name)

    def get_example_input(
        self, method_name: str = "forward"
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get the example input for a specific method.

        Args:
            method_name: Name of the method to get example input for, defaults to "forward"

        Returns:
            Tuple of tensors representing the example input

        Raises:
            KeyError: If the method name is not found in example inputs
            ValueError: If the example inputs list is empty
        """
        if method_name not in self._example_inputs:
            raise KeyError(f"Method name '{method_name}' not found in example inputs")

        # Access the first element of the list for this method
        example_inputs_list = self._example_inputs[method_name]
        if not example_inputs_list:
            raise ValueError(f"Example inputs list for method {method_name} is empty")

        # The original code expects this to be a tuple of tensors
        return self._example_inputs[method_name][0]

    def run_method(
        self,
        method_name: str = "forward",
        example_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Sequence[Any]:
        """
        Run a specific method with the given inputs.

        Args:
            method_name: Name of the method to run, defaults to "forward"
            example_inputs: Optional inputs to use, defaults to the example inputs

        Returns:
            The outputs of the method execution

        Raises:
            RuntimeError: If the method cannot be loaded
        """
        # Lazy import to avoid forcing portable_lib dependency at module load time
        from executorch.runtime import Runtime, Verification

        et_runtime = Runtime.get()
        program = et_runtime.load_program(
            self.get_pte_buffer(), verification=Verification.Minimal
        )
        forward = program.load_method(method_name)

        if forward is None:
            raise RuntimeError(
                f"Failed to load method '{method_name}' from the program"
            )
        if example_inputs is None:
            example_inputs = self.get_example_input(method_name)

        return forward.execute(example_inputs)

    def print_delegation_info(self) -> None:
        """
        Print delegation information for the exported program.
        """
        lowering_stage = list(
            set(self._pipeline_stages)
            & {StageType.TO_EDGE_TRANSFORM_AND_LOWER, StageType.TO_BACKEND}
        )
        if not lowering_stage:
            RuntimeError(
                "No delegation info available, atleast one of the lowering stages should be present"
            )

        stage_artifact = self._stage_to_artifacts.get(lowering_stage[0])
        if stage_artifact is None:
            RuntimeError("No delegation info available, run the lowering stage first")

        # pyre-ignore
        delegation_info = stage_artifact.get_context("delegation_info", None)
        if delegation_info:
            print(delegation_info.get_summary())
            df = delegation_info.get_operator_delegation_dataframe()
            print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
        else:
            print("No delegation info available")

    # Use Any instead of ETRecord as return type to avoid static dependency on etrecord
    def get_etrecord(self) -> Any:
        """
        Get the etrecord from the ExecuTorchProgramManager.

        Returns:
            The etrecord in the ExecuTorchProgramManager

        Raises:
            RuntimeError: If the ExecuTorchManager is unavailable, or etrecord is not available in the ExecuTorchProgramManager
        """
        return self.get_executorch_program_manager().get_etrecord()
