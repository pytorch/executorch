# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import Mock, patch

import torch
from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager
from executorch.export import ExportRecipe, QuantizationRecipe
from executorch.export.export import (
    EdgeTransformAndLowerStage,
    ExecutorchStage,
    ExportSession,
    ExportStage,
    QuantizeStage,
    SourceTransformStage,
)
from torch.export import ExportedProgram
from torchao.quantization.granularity import PerAxis
from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig


class SimpleTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestExportStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.models_dict = {"forward": self.model}
        self.export_config = {
            "example_inputs": {"forward": self.example_inputs},
            "dynamic_shapes": {},
        }

    @patch("torch.export.export")
    def test_export_stage_run_success(self, mock_torch_export: Mock) -> None:
        mock_exported_program = Mock(spec=ExportedProgram)
        mock_torch_export.return_value = mock_exported_program

        stage = ExportStage()
        stage.run({"model": self.models_dict}, self.export_config)

        mock_torch_export.assert_called_once_with(
            self.model,
            self.example_inputs[0],
            dynamic_shapes=None,
            strict=True,
        )

        # Verify artifacts
        artifacts = stage.get_artifacts()
        self.assertIn("forward", artifacts)
        self.assertEqual(artifacts["forward"], mock_exported_program)

    def test_export_stage_missing_example_inputs(self) -> None:
        stage = ExportStage()
        with self.assertRaises(ValueError) as context:
            stage.run({"model": self.models_dict}, {"example_inputs": {}})
        self.assertIn(
            "Example inputs for method forward not found", str(context.exception)
        )


class TestEdgeTransformAndLowerStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_exported_program = Mock(spec=ExportedProgram)
        self.exported_programs = {"forward": self.mock_exported_program}

    def test_edge_transform_stage_with_partitioners(self) -> None:
        """Test that EdgeTransformAndLowerStage can be initialized with partitioners."""
        mock_partitioner = Mock()
        stage = EdgeTransformAndLowerStage(partitioners=[mock_partitioner])
        self.assertEqual(stage.name, "edge_transform_and_lower")
        self.assertEqual(stage._partitioners, [mock_partitioner])

    def test_edge_transform_stage_with_config(self) -> None:
        """Test that EdgeTransformAndLowerStage can be initialized with compile config."""
        mock_config = Mock()
        stage = EdgeTransformAndLowerStage(compile_config=mock_config)
        self.assertEqual(stage.name, "edge_transform_and_lower")
        self.assertEqual(stage._compile_config, mock_config)

    def test_edge_transform_stage_get_artifacts_not_initialized(self) -> None:
        stage = EdgeTransformAndLowerStage()
        with self.assertRaises(RuntimeError) as context:
            stage.get_artifacts()
        self.assertIn("Edge program manager is not initialized", str(context.exception))


class TestExecutorchStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_edge_manager = Mock(spec=EdgeProgramManager)
        self.mock_backend_config = Mock()

    def test_executorch_stage_run_success(self) -> None:
        mock_executorch_manager = Mock(spec=ExecutorchProgramManager)
        self.mock_edge_manager.to_executorch.return_value = mock_executorch_manager

        stage = ExecutorchStage(self.mock_backend_config)
        stage.run(self.mock_edge_manager, {})

        # Verify to_executorch was called
        self.mock_edge_manager.to_executorch.assert_called_once_with(
            self.mock_backend_config
        )

        # Verify artifacts
        artifacts = stage.get_artifacts()
        self.assertEqual(artifacts, mock_executorch_manager)

    def test_executorch_stage_get_artifacts_not_initialized(self) -> None:
        stage = ExecutorchStage(self.mock_backend_config)
        with self.assertRaises(RuntimeError) as context:
            stage.get_artifacts()
        self.assertIn(
            "Executorch program manager is not initialized", str(context.exception)
        )


class TestSourceTransformStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.models_dict = {"forward": self.model}

    def test_source_transform_stage_no_quantization(self) -> None:
        stage = SourceTransformStage(None)
        stage.run(self.models_dict)

        artifacts = stage.get_artifacts()
        self.assertEqual(artifacts, self.models_dict)


class TestQuantizeStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.models_dict = {"forward": self.model}
        self.example_inputs = [(torch.randn(2, 10),)]
        self.calibration_config = {"example_inputs": {"forward": self.example_inputs}}

    def test_quantize_stage_missing_example_inputs(self) -> None:
        mock_quantizers = [Mock()]
        stage = QuantizeStage(mock_quantizers)

        with self.assertRaises(ValueError) as context:
            stage.run(self.models_dict, {"example_inputs": {}})
        self.assertIn(
            "Example inputs for method forward not found or empty",
            str(context.exception),
        )


class TestExportSession(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    def test_export_session_fp32_pipeline(self) -> None:
        """Test that FP32 export creates the expected pipeline stages."""
        recipe = ExportRecipe(name="test_fp32")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        # Verify pipeline stages for FP32
        expected_stages = ["export", "edge_transform_and_lower", "executorch"]
        actual_stages = [stage.name for stage in session._pipeline]
        self.assertEqual(actual_stages, expected_stages)

    def test_export_session_quantized_pipeline_with_quantizers(self) -> None:
        """Test that quantized export with quantizers creates the expected pipeline stages."""
        mock_quantizer = Mock()
        quant_recipe = QuantizationRecipe(quantizers=[mock_quantizer])
        recipe = ExportRecipe(name="test_quantized", quantization_recipe=quant_recipe)

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        # Verify pipeline stages for quantized export with quantizers
        # The quantize stage is followed by a re-export stage
        expected_stages = [
            "quantize",
            "export",
            "edge_transform_and_lower",
            "executorch",
        ]
        actual_stages = [stage.name for stage in session._pipeline]
        self.assertEqual(actual_stages, expected_stages)

    def test_export_session_source_transform_pipeline(self) -> None:
        """Test that source transform creates the expected pipeline stages."""
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(axis=0),
        )
        quant_recipe = QuantizationRecipe(ao_base_config=[config])
        recipe = ExportRecipe(
            name="test_source_transform", quantization_recipe=quant_recipe
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        # Verify pipeline stages for source transform
        expected_stages = [
            "source_transform",
            "export",
            "edge_transform_and_lower",
            "executorch",
        ]
        actual_stages = [stage.name for stage in session._pipeline]
        self.assertEqual(actual_stages, expected_stages)

    def test_export_session_full_quantization_pipeline(self) -> None:
        """Test that full quantization (source transform + quantizers) creates the expected pipeline stages."""
        mock_quantizer = Mock()
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerAxis(axis=0),
        )
        quant_recipe = QuantizationRecipe(
            quantizers=[mock_quantizer],
            ao_base_config=[config],
        )
        recipe = ExportRecipe(
            name="test_full_quantization", quantization_recipe=quant_recipe
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        # Verify pipeline stages for full quantization
        # The quantize stage is followed by a re-export stage
        expected_stages = [
            "source_transform",
            "quantize",
            "export",
            "edge_transform_and_lower",
            "executorch",
        ]
        actual_stages = [stage.name for stage in session._pipeline]
        self.assertEqual(actual_stages, expected_stages)

    @patch("executorch.export.export.ExportSession._run_pipeline")
    def test_export_session_export_calls_pipeline(
        self, mock_run_pipeline: Mock
    ) -> None:
        """Test that export() method calls the pipeline."""
        recipe = ExportRecipe(name="test")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()
        mock_run_pipeline.assert_called_once()

    def test_export_session_standardize_inputs(self) -> None:
        """Test that inputs are properly standardized to dictionary format."""
        recipe = ExportRecipe(name="test")

        # Test single model and example_inputs
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        self.assertIsInstance(session._model, dict)
        self.assertIn("forward", session._model)
        self.assertEqual(session._model["forward"], self.model)

        self.assertIsInstance(session._example_inputs, dict)
        self.assertIn("forward", session._example_inputs)
        self.assertEqual(session._example_inputs["forward"], self.example_inputs)

    def test_export_session_dict_inputs(self) -> None:
        """Test that dictionary inputs are preserved."""
        recipe = ExportRecipe(name="test")
        model_dict = {"method1": self.model, "method2": SimpleTestModel()}
        example_inputs_dict = {
            "method1": self.example_inputs,
            "method2": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,
            example_inputs=example_inputs_dict,
            export_recipe=recipe,
        )

        self.assertEqual(session._model, model_dict)
        self.assertEqual(session._example_inputs, example_inputs_dict)

    def test_export_session_get_example_input(self) -> None:
        """Test getting example input for a method."""
        recipe = ExportRecipe(name="test")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        example_input = session.get_example_input("forward")
        self.assertEqual(example_input, self.example_inputs[0])

    def test_export_session_get_example_input_missing_method(self) -> None:
        """Test error when getting example input for non-existent method."""
        recipe = ExportRecipe(name="test")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        with self.assertRaises(KeyError) as context:
            session.get_example_input("nonexistent")
        self.assertIn("Method name 'nonexistent' not found", str(context.exception))

    def test_export_session_runtime_errors_before_export(self) -> None:
        """Test that runtime errors are raised when accessing results before export."""
        recipe = ExportRecipe(name="test")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        with self.assertRaises(RuntimeError):
            session.get_executorch_program()

        with self.assertRaises(RuntimeError):
            session.get_executorch_program_manager()

        with self.assertRaises(RuntimeError):
            session.get_pte_buffer()

        with self.assertRaises(RuntimeError):
            session.save_to_pte("test.pte")


class TestExportSessionPipelineExecution(unittest.TestCase):
    """Test the actual pipeline execution with mocked stages."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    @patch("executorch.export.export.ExecutorchStage")
    @patch("executorch.export.export.EdgeTransformAndLowerStage")
    @patch("executorch.export.export.ExportStage")
    def test_pipeline_execution_order_fp32(
        self,
        mock_export_stage_class: Mock,
        mock_edge_stage_class: Mock,
        mock_executorch_stage_class: Mock,
    ) -> None:
        """Test that stages are executed in the correct order for FP32."""
        # Create mock stages
        mock_export_stage = Mock()
        mock_export_stage.name = "export"
        mock_export_stage.get_artifacts.return_value = {"forward": Mock()}

        mock_edge_stage = Mock()
        mock_edge_stage.name = "edge_transform_and_lower"
        mock_edge_stage.get_artifacts.return_value = Mock()
        mock_edge_stage.delegation_info = Mock()

        mock_executorch_stage = Mock()
        mock_executorch_stage.name = "executorch"
        mock_executorch_stage.get_artifacts.return_value = Mock()

        # Configure the mock classes to return our mock instances
        mock_export_stage_class.return_value = mock_export_stage
        mock_edge_stage_class.return_value = mock_edge_stage
        mock_executorch_stage_class.return_value = mock_executorch_stage

        recipe = ExportRecipe(name="test_fp32")
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        # Verify stages were called in the correct order
        mock_export_stage.run.assert_called_once()
        mock_edge_stage.run.assert_called_once()
        mock_executorch_stage.run.assert_called_once()

    @patch("executorch.export.export.ExecutorchStage")
    @patch("executorch.export.export.EdgeTransformAndLowerStage")
    @patch("executorch.export.export.ExportStage")
    @patch("executorch.export.export.QuantizeStage")
    def test_pipeline_execution_order_quantized(
        self,
        mock_quantize_stage_class: Mock,
        mock_export_stage_class: Mock,
        mock_edge_stage_class: Mock,
        mock_executorch_stage_class: Mock,
    ) -> None:
        """Test that stages are executed in the correct order for quantized export."""
        # Create mock stages
        mock_quantize_stage = Mock()
        mock_quantize_stage.name = "quantize"
        mock_quantize_stage.get_artifacts.return_value = {"forward": Mock()}

        mock_export_stage = Mock()
        mock_export_stage.name = "export"
        mock_export_stage.get_artifacts.return_value = {"forward": Mock()}

        mock_edge_stage = Mock()
        mock_edge_stage.name = "edge_transform_and_lower"
        mock_edge_stage.get_artifacts.return_value = Mock()
        mock_edge_stage.delegation_info = Mock()

        mock_executorch_stage = Mock()
        mock_executorch_stage.name = "executorch"
        mock_executorch_stage.get_artifacts.return_value = Mock()

        # Configure the mock classes to return our mock instances
        mock_quantize_stage_class.return_value = mock_quantize_stage
        mock_export_stage_class.return_value = mock_export_stage
        mock_edge_stage_class.return_value = mock_edge_stage
        mock_executorch_stage_class.return_value = mock_executorch_stage

        mock_quantizer = Mock()
        quant_recipe = QuantizationRecipe(quantizers=[mock_quantizer])
        recipe = ExportRecipe(name="test_quantized", quantization_recipe=quant_recipe)

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        # Verify stages were called in the correct order
        mock_quantize_stage.run.assert_called_once()
        mock_export_stage.run.assert_called_once()
        mock_edge_stage.run.assert_called_once()
        mock_executorch_stage.run.assert_called_once()


class TestExportFunction(unittest.TestCase):
    """Test the top-level export function."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    @patch("executorch.export.export.ExportSession")
    def test_export_function_creates_session_and_exports(
        self, mock_session_class: Mock
    ) -> None:
        """Test that export function creates session and calls export."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        recipe = ExportRecipe(name="test")
        from executorch.export import export

        result = export(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
            name="test_export",
        )
        mock_session_class.assert_called_once_with(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
            name="test_export",
            dynamic_shapes=None,
            constant_methods=None,
            artifact_dir=None,
        )
        mock_session.export.assert_called_once()
        self.assertEqual(result, mock_session)
