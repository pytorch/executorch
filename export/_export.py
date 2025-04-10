from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir.program import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from executorch.runtime import Runtime, Verification
from tabulate import tabulate
from torch.ao.quantization import allow_exported_model_train_eval
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import export_for_training, ExportedProgram

from ._recipe import ExportRecipe


def export(
    model: Union[nn.Module, Dict[str, nn.Module]],
    example_inputs: Union[List[tuple[torch.Tensor, ...]], Dict[str, List[tuple[torch.Tensor, ...]]]],
    export_recipe: ExportRecipe,
    name: Optional[str] = None,
    dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
    constant_methods: Optional[Union[Dict[str, Callable]]] = None,
    artifact_dir: Optional[str] = None,
    apply_quantization: bool = False,
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
        apply_quantization: Whether to apply quantization during export, defaults to False

    Returns:
        A configured ExportSession instance with the export process completed if requested
    """
    manager = ExportSession(
        model=model,
        example_inputs=example_inputs,
        export_recipe=export_recipe,
        name=name,
        dynamic_shapes=dynamic_shapes,
        constant_methods=constant_methods,
        artifact_dir=artifact_dir,
    )
    
    if apply_quantization:
        manager.export(apply_quantization=True)
    
    return manager


class ExportSession:
    """
    Manages the export process for ExecuTorch models.

    This class handles the three-stage export process:
    1. Export PyTorch model to ExportedProgram
    2. Transform and lower to EdgeProgramManager
    3. Convert to ExecutorchProgramManager for final execution
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        example_inputs: Union[List[tuple[torch.Tensor, ...]], Dict[str, List[tuple[torch.Tensor, ...]]]],
        export_recipe: ExportRecipe,
        name: Optional[str] = None,
        dynamic_shapes: Optional[Union[Any, Dict[str, Any]]] = None,
        constant_methods: Optional[Union[Dict[str, Callable]]] = None,
        artifact_dir: Optional[str] = None,
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
        """
        # Standardize model to dictionary format
        self._model = model if isinstance(model, dict) else {"forward": model}
        
        # Standardize example_inputs to dictionary format
        self._example_inputs = example_inputs if isinstance(example_inputs, dict) else {"forward": example_inputs}
        
        # Standardize dynamic_shapes to dictionary format
        self._dynamic_shapes = {}
        if dynamic_shapes is not None:
            if isinstance(dynamic_shapes, dict):
                self._dynamic_shapes = dynamic_shapes
            else:
                self._dynamic_shapes = {"forward": dynamic_shapes}
        
        self._name = name
        self._constant_methods = constant_methods
        self._artifact_dir = artifact_dir
        self._export_recipe = export_recipe
        self._exported_program: Dict[str, ExportedProgram] = {}
        self._edge_program_manager: Optional[EdgeProgramManager] = None
        self._executorch_program_manager: Optional[ExecutorchProgramManager] = None
        self._delegation_info = None
        
        # Export models for training to enable quantization
        self._exported_models: Dict[str, nn.Module] = {}
        for method_name, model in self._model.items():
            self._exported_models[method_name] = export_for_training(
                model,
                self._example_inputs[method_name][0],  # type: ignore
                dynamic_shapes=self._dynamic_shapes.get(method_name, None),
            ).module()

    def quantize(self) -> None:
        """
        Perform post-training quantization on the model.

        This method applies post-training quantization to the model using the
        quantizer specified in the export recipe and the calibration data from
        the export input. The model is modified in-place.

        Note:
            This should be called before the export process if quantization is desired.
        """
        if self._export_recipe.quantizer is None:
            raise ValueError("Quantizer not specified in the export recipe")

        for method_name, model in self._model.items():
            # Set model to evaluation mode for quantization
            model.eval()

            # Use the pre-exported model from initialization
            captured_model = self._exported_models[method_name]

            # Get the quantizer from the recipe
            quantizer = self._export_recipe.get_quantizer()

            # Prepare the model for quantization
            prepared_model = prepare_pt2e(captured_model, quantizer)  # type: ignore

            # Allow the exported model to switch between train and eval modes
            allow_exported_model_train_eval(prepared_model)

            # Calibrate the model with the provided calibration data
            for calibration_input in self._example_inputs[method_name]:  # type: ignore
                prepared_model(*calibration_input)

            # Convert the prepared model to a quantized model
            # Update the model in the model dictionary
            quantized_model = convert_pt2e(prepared_model)
            self._model[method_name] = quantized_model  # type: ignore

    def export(self, apply_quantization: bool = False) -> None:
        """
        Execute the full export process.

        This method orchestrates the export process with optional quantization:
        1. (Optional) Apply quantization to the model
        2. Export the PyTorch model to ExportedProgram
        3. Transform and lower to EdgeProgramManager
        4. Convert to ExecutorchProgramManager

        Args:
            apply_quantization: Whether to apply quantization before export, defaults to False
        """
        if apply_quantization and self._export_recipe.quantizer is not None:
            self.quantize()
            
        self._export_stage()
        self._to_edge_transform_and_lower_stage()
        self._to_executorch_stage()

    def _export_stage(self) -> None:
        """
        First stage: Export PyTorch model to ExportedProgram.

        Exports each model in the input to an ExportedProgram and applies
        any pre-edge transform passes if specified.
        """
        with torch.no_grad():
            for method_name, model in self._model.items():
                # Check if method_name exists in example_inputs
                if method_name not in self._example_inputs:
                    raise ValueError(
                        f"Example inputs for method {method_name} not found."
                    )

                # Get dynamic shapes if available
                dynamic_shapes = None
                if method_name in self._dynamic_shapes:
                    dynamic_shapes = self._dynamic_shapes[method_name]

                # Export the model
                self._exported_program[method_name] = torch.export.export(
                    model,
                    self._example_inputs[method_name][0],
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )

                # Apply pre-edge transform passes if available
                if self._export_recipe.pre_edge_transform_passes is not None:
                    self._exported_program[method_name] = (
                        self._export_recipe.pre_edge_transform_passes(
                            self._exported_program[method_name]
                        )
                    )

    def _to_edge_transform_and_lower_stage(self) -> None:
        """
        Second stage: Transform and lower to EdgeProgramManager.

        Applies partitioning and transformation passes to convert the
        ExportedProgram to an EdgeProgramManager.
        """
        self._edge_program_manager = to_edge_transform_and_lower(
            self._exported_program,
            partitioner=self._export_recipe.partitioners,
            transform_passes=self._export_recipe.edge_transform_passes,
            constant_methods=self._constant_methods,
            compile_config=self._export_recipe.edge_compile_config,
        )
        self._delegation_info = get_delegation_info(self._edge_program_manager.exported_program().graph_module)

    def _to_executorch_stage(self) -> None:
        """
        Third stage: Convert to ExecutorchProgramManager.

        Converts the EdgeProgramManager to an ExecutorchProgramManager
        using the specified backend configuration.
        """
        if self._edge_program_manager is None:
            raise RuntimeError(
                "Edge program manager is not initialized. Run _to_edge_transform_and_lower_stage first."
            )
        self._executorch_program_manager = self._edge_program_manager.to_executorch(
            self._export_recipe.executorch_backend_config
        )

    def save_pte_file(self, path: str) -> None:
        """
        Save the exported program to a PTE file.

        Args:
            path: Path where the PTE file will be saved

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        if self._executorch_program_manager is None:
            raise RuntimeError(
                "Executorch program manager is not initialized. Run export() first."
            )
        self._executorch_program_manager.save(path)

    def get_pte_buffer(self) -> bytes:
        """
        Get the PTE buffer as bytes.

        Returns:
            The PTE buffer as bytes

        Raises:
            RuntimeError: If the executorch program manager is not initialized
        """
        if self._executorch_program_manager is None:
            raise RuntimeError(
                "Executorch program manager is not initialized. Run export() first."
            )
        return self._executorch_program_manager.buffer

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
        print(self._delegation_info.get_summary())
        df = self._delegation_info.get_operator_delegation_dataframe()
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
