# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from executorch.exir._warnings import experimental
from executorch.exir.program import ExecutorchProgramManager
from executorch.exir.schema import Program
from executorch.extension.export_util.utils import save_pte_program
from executorch.runtime import Runtime, Verification
from tabulate import tabulate
from torch import nn

from .recipe import ExportRecipe, QuantizationRecipe
from .stages import (
    EdgeTransformAndLowerStage,
    ExecutorchStage,
    PipelineArtifact,
    QuantizeStage,
    SourceTransformStage,
    Stage,
    StageType,
    ToBackendStage,
    ToEdgeStage,
    TorchExportStage,
)


@experimental(
    "This API and all of its related functionality such as ExportSession and ExportRecipe are experimental."
)
def export(
    model: Union[nn.Module, Dict[str, nn.Module]],
    example_inputs: Union[
        List[tuple[torch.Tensor, ...]], Dict[str, List[tuple[torch.Tensor, ...]]]
    ],
    export_recipe: ExportRecipe,
    name: Optional[str] = None,
    dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
    constant_methods: Optional[Union[Dict[str, Callable]]] = None,
    artifact_dir: Optional[str] = None,
) -> "ExportSession":
    """
    Create and configure an ExportSession with the given parameters.

    This function provides a convenient way to create an ExportSession and
    optionally run the export process in one step.

    Args:
        model: The PyTorch model(s) to export, either a single model or a dictionary
              mapping method names to models
        example_inputs: Example inputs for the model(s), either a list of input tuples
                      or a dictionary mapping method names to lists of input tuples
        export_recipe: Contains the configuration for the export process
        name: Optional name for the export
        dynamic_shapes: Optional dynamic shape specifications
        constant_methods: Optional dictionary of constant methods
        artifact_dir: Optional directory to store artifacts

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
        model: Union[nn.Module, Dict[str, nn.Module]],
        example_inputs: Union[
            List[tuple[torch.Tensor, ...]], Dict[str, List[tuple[torch.Tensor, ...]]]
        ],
        export_recipe: ExportRecipe,
        name: Optional[str] = None,
        dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
        constant_methods: Optional[Union[Dict[str, Callable]]] = None,
        artifact_dir: Optional[str] = None,
        pipeline_stages: Optional[List[StageType]] = None,
    ) -> None:
        """
        Initialize the ExportSession with model, inputs, and recipe.

        Args:
            model: The PyTorch model(s) to export, either a single model or a dictionary
                  mapping method names to models
            example_inputs: Example inputs for the model(s), either a list of input tuples
                          or a dictionary mapping method names to lists of input tuples
            export_recipe: Contains the configuration for the export process
            name: Optional name for the export
            dynamic_shapes: Optional dynamic shape specifications
            constant_methods: Optional dictionary of constant methods
            artifact_dir: Optional directory to store artifacts
            pipeline_stages: Optional list of stages to execute, defaults to a standard pipeline.
        """
        # Standardize model to dictionary format
        self._model = model if isinstance(model, dict) else {"forward": model}

        # Standardize example_inputs to dictionary format
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

        # Default pipeline
        self._pipeline_stages = pipeline_stages or self._get_default_pipeline()
        self._pipeline = self._build_pipeline_from_stages(self._pipeline_stages)

        self.valid_next_stage_transitions: Dict[StageType, List[StageType]] = {
            StageType.SOURCE_TRANSFORM: [StageType.QUANTIZE, StageType.TORCH_EXPORT],
            StageType.QUANTIZE: [StageType.TORCH_EXPORT],
            StageType.TORCH_EXPORT: [
                StageType.TO_EDGE,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            ],
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: [StageType.TO_EXECUTORCH],
            StageType.TO_EDGE: [StageType.TO_BACKEND],
            StageType.TO_BACKEND: [StageType.TO_EXECUTORCH],
            StageType.TO_EXECUTORCH: [],
        }

        # Intialize run context
        self._run_context: Dict[str, Any] = {
            "example_inputs": self._example_inputs,
            "dynamic_shapes": self._dynamic_shapes,
            "constant_methods": constant_methods,
            "export_recipe": self._export_recipe,
            "session_name": name,
            "artifact_dir": artifact_dir,
        }

        self._stage_to_artifacts: Dict[StageType, PipelineArtifact] = {}

    def _get_default_pipeline(self) -> List[StageType]:
        return [
            StageType.SOURCE_TRANSFORM,  # Optional stage, returns original model if quant recipe is invalid
            StageType.QUANTIZE,  # Optional stage, returns original model if quant recipe is invalid
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]

    def _build_pipeline_from_stages(self, stage_types: List[StageType]) -> List[Stage]:
        pipeline: List[Stage] = []

        for stage_type in stage_types:
            if stage_type == StageType.SOURCE_TRANSFORM:
                stage = SourceTransformStage(self._quant_recipe)
            elif stage_type == StageType.QUANTIZE:
                stage = QuantizeStage(self._quant_recipe)
            elif stage_type == StageType.TORCH_EXPORT:
                pre_edge_passes = None
                if self._export_recipe.pre_edge_transform_passes is not None:
                    pre_edge_passes = list(
                        self._export_recipe.pre_edge_transform_passes
                    )
                stage = TorchExportStage(pre_edge_passes)
            elif stage_type == StageType.TO_EDGE_TRANSFORM_AND_LOWER:
                stage = EdgeTransformAndLowerStage(
                    partitioners=self._export_recipe.partitioners,
                    transform_passes=self._export_recipe.edge_transform_passes,
                    compile_config=self._export_recipe.edge_compile_config,
                )
            elif stage_type == StageType.TO_EDGE:
                stage = ToEdgeStage(
                    edge_compile_config=self._export_recipe.edge_compile_config
                )
            elif stage_type == StageType.TO_BACKEND:
                stage = ToBackendStage(
                    partitioners=self._export_recipe.partitioners,
                    transform_passes=self._export_recipe.edge_transform_passes,
                )
            elif stage_type == StageType.TO_EXECUTORCH:
                stage = ExecutorchStage(self._export_recipe.executorch_backend_config)
            else:
                raise ValueError(f"Unknown stage type: {stage_type}")

            pipeline.append(stage)

        return pipeline

    @classmethod
    def _validate_pipeline_sequence(
        cls,
        valid_next_stage_transitions: Dict[StageType, List[StageType]],
        stages: List[StageType],
        mandatory_stages: Set[StageType],
    ) -> bool:
        if not stages:
            raise ValueError("Pipeline stages cannot be empty")

        # Check for mandatory stages
        if not mandatory_stages.issubset(stages):
            return False

        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            if next_stage not in valid_next_stage_transitions.get(current_stage, []):
                logging.error(
                    "Invalid transition from", current_stage, "to", next_stage
                )
                return False
        return True

    def _run_pipeline(self) -> None:
        # Validate if given stage sequence is valid
        if not self._validate_pipeline_sequence(
            self.valid_next_stage_transitions,
            stages=self._pipeline_stages,
            mandatory_stages={StageType.TORCH_EXPORT},
        ):
            raise ValueError("Invalid pipeline sequence")

        current_artifact = PipelineArtifact(data=self._model, context=self._run_context)

        # Execute stages
        for stage in self._pipeline:
            stage_type = stage.stage_type
            logging.info(f"Executing stage: {stage_type}")

            stage.run(current_artifact)
            current_artifact = stage.get_artifacts()

            self._stage_to_artifacts[stage.stage_type] = current_artifact

            logging.debug(
                f"Context after {stage_type}: {list(current_artifact.context.keys())}"
            )
            logging.info(f"Completed stage: {stage_type}")

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

    def save_pte_file(self, path: str) -> None:
        """
        Save the exported program to a PTE file.

        Args:
            path: Path where the PTE file will be saved

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        self.get_executorch_program_manager().save(path)

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
        delegation_info = self._run_context.get("delegation_info", None)
        if delegation_info:
            logging.info(delegation_info.get_summary())
            df = delegation_info.get_operator_delegation_dataframe()
            logging.info(tabulate(df, headers="keys", tablefmt="fancy_grid"))
        else:
            logging.info("No delegation info available")
