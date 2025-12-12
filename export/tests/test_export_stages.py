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
from executorch.export import AOQuantizationConfig, QuantizationRecipe, StageType
from executorch.export.stages import (
    AtenTransformStage,
    EdgeTransformAndLowerStage,
    ExecutorchStage,
    PipelineArtifact,
    QuantizeStage,
    SourceTransformStage,
    ToBackendStage,
    ToEdgeStage,
    TorchExportStage,
)
from torch.export import ExportedProgram
from torchao.quantization.pt2e.quantizer import Quantizer as TorchAOPT2EQuantizer


class SimpleTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestPipelineArtifact(unittest.TestCase):

    def test_copy_with_new_data(self) -> None:
        original_data = {"original": "data"}
        context = {"key": "value"}
        artifact = PipelineArtifact(data=original_data, context=context)

        new_data = {"new": "data"}
        new_artifact = artifact.copy_with_new_data(new_data)

        self.assertEqual(new_artifact.data, new_data)
        self.assertEqual(new_artifact.context, context)
        # Ensure original is unchanged
        self.assertEqual(artifact.data, original_data)


class TestTorchExportStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.models_dict = {"forward": self.model}
        self.context = {
            "example_inputs": {"forward": self.example_inputs},
            "dynamic_shapes": {},
        }

    @patch("torch.export.export")
    def test_export_stage_run_success(self, mock_torch_export: Mock) -> None:
        mock_exported_program = Mock(spec=ExportedProgram)
        mock_torch_export.return_value = mock_exported_program

        stage = TorchExportStage()
        artifact = PipelineArtifact(data=self.models_dict, context=self.context)

        stage.run(artifact)

        mock_torch_export.assert_called_once_with(
            self.model,
            self.example_inputs[0],
            dynamic_shapes=None,
            strict=True,
        )

        # Verify artifacts
        artifact = stage.get_artifacts()
        self.assertIn("forward", artifact.data)
        self.assertEqual(artifact.data["forward"], mock_exported_program)

    def test_export_stage_missing_example_inputs(self) -> None:
        stage = TorchExportStage()
        context = {"example_inputs": {}}
        artifact = PipelineArtifact(data=self.models_dict, context=context)

        with self.assertRaises(ValueError) as cm:
            stage.run(artifact)
        self.assertIn("Example inputs for method forward not found", str(cm.exception))

    def test_get_artifacts_before_run(self) -> None:
        """Test error when getting artifacts before running stage."""
        stage = TorchExportStage()
        with self.assertRaises(RuntimeError) as cm:
            stage.get_artifacts()
        self.assertIn("Stage: TorchExportStage not executed", str(cm.exception))


class TestEdgeTransformAndLowerStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_exported_program = Mock(spec=ExportedProgram)
        self.exported_programs = {"forward": self.mock_exported_program}
        self.context = {"constant_methods": None}

    @patch("executorch.export.stages.to_edge_transform_and_lower")
    @patch("executorch.export.stages.get_delegation_info")
    def test_run_with_partitioners_and_config(
        self, mock_get_delegation_info: Mock, mock_to_edge_transform_and_lower: Mock
    ) -> None:
        """Test execution with partitioners and compile config"""
        mock_delegation_info = {"delegation": "info"}
        mock_get_delegation_info.return_value = mock_delegation_info

        mock_partitioners = [Mock()]
        mock_compile_config = Mock()

        # Create a mock transform pass callable that we can verify
        mock_transform_pass = Mock()
        mock_pass1 = Mock()
        mock_pass2 = Mock()
        mock_transform_pass.return_value = [mock_pass1, mock_pass2]
        mock_transform_passes = [mock_transform_pass]

        mock_edge_program_manager = Mock(spec=EdgeProgramManager)
        mock_exported_program = Mock()
        mock_graph_module = Mock()
        mock_exported_program.graph_module = mock_graph_module
        mock_edge_program_manager.exported_program.return_value = mock_exported_program
        mock_to_edge_transform_and_lower.return_value = mock_edge_program_manager

        stage = EdgeTransformAndLowerStage(
            partitioners=mock_partitioners,
            transform_passes=mock_transform_passes,
            compile_config=mock_compile_config,
        )

        # Test that the stage has the right configuration
        self.assertEqual(stage.stage_type, StageType.TO_EDGE_TRANSFORM_AND_LOWER)
        self.assertEqual(stage._partitioners, mock_partitioners)
        self.assertEqual(stage._transform_passes, mock_transform_passes)
        self.assertEqual(stage._compile_config, mock_compile_config)

        # Test the run method
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)
        stage.run(artifact)

        # Verify the transform pass callable was called with correct parameters
        mock_transform_pass.assert_called_once_with(
            "forward", self.mock_exported_program
        )

        # Verify to_edge_transform_and_lower was called with the expected structure
        expected_transform_passes = {"forward": [mock_pass1, mock_pass2]}
        mock_to_edge_transform_and_lower.assert_called_once_with(
            self.exported_programs,
            partitioner=mock_partitioners,
            transform_passes=expected_transform_passes,
            constant_methods=None,
            compile_config=mock_compile_config,
            generate_etrecord=False,
        )

        # Verify artifacts are set correctly
        result_artifact = stage.get_artifacts()
        self.assertEqual(result_artifact.data, mock_edge_program_manager)
        self.assertEqual(
            result_artifact.get_context("delegation_info"), mock_delegation_info
        )


class TestExecutorchStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_edge_manager = Mock(spec=EdgeProgramManager)
        self.mock_backend_config = Mock()

    def test_executorch_stage_run_success(self) -> None:
        mock_executorch_manager = Mock(spec=ExecutorchProgramManager)
        self.mock_edge_manager.to_executorch.return_value = mock_executorch_manager

        stage = ExecutorchStage(self.mock_backend_config)
        artifact = PipelineArtifact(data=self.mock_edge_manager, context={})
        stage.run(artifact)

        # Verify to_executorch was called
        self.mock_edge_manager.to_executorch.assert_called_once_with(
            self.mock_backend_config
        )

        # Verify artifacts
        artifacts = stage.get_artifacts()
        self.assertEqual(artifacts.data, mock_executorch_manager)

    def test_executorch_stage_get_artifacts_not_initialized(self) -> None:
        stage = ExecutorchStage(self.mock_backend_config)
        artifact = PipelineArtifact(data=None, context={})

        with self.assertRaises(RuntimeError) as cm:
            stage.run(artifact)
        self.assertIn("Edge program manager is not set", str(cm.exception))


class TestSourceTransformStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.models_dict = {"forward": self.model}

    def test_source_transform_stage_no_quantization(self) -> None:
        mock_recipe = Mock(spec=QuantizationRecipe)
        mock_recipe.ao_quantization_configs = None
        stage = SourceTransformStage(mock_recipe)
        artifact = PipelineArtifact(data=self.models_dict, context={})

        stage.run(artifact)

        result_artifact = stage.get_artifacts()
        self.assertEqual(result_artifact.data, self.models_dict)

    @patch("executorch.export.stages.quantize_")
    @patch("executorch.export.stages.unwrap_tensor_subclass")
    def test_run_with_ao_quantization_configs(
        self, mock_unwrap: Mock, mock_quantize: Mock
    ) -> None:
        from torchao.core.config import AOBaseConfig

        mock_config = Mock(spec=AOBaseConfig)
        mock_filter_fn = Mock()
        mock_ao_config: AOQuantizationConfig = AOQuantizationConfig(
            ao_base_config=mock_config, filter_fn=mock_filter_fn
        )
        mock_recipe = Mock(spec=QuantizationRecipe)
        mock_recipe.ao_quantization_configs = [mock_ao_config]

        stage = SourceTransformStage(mock_recipe)

        models_dict = {"forward": self.model}
        artifact = PipelineArtifact(data=models_dict, context={})
        stage.run(artifact)

        # Verify quantize_ was called once (with the copied model, not the original)
        self.assertEqual(mock_quantize.call_count, 1)
        # Verify the config and filter_fn arguments are correct
        call_args = mock_quantize.call_args[0]
        self.assertNotEqual(self.model, call_args[0])
        self.assertEqual(call_args[1], mock_config)
        self.assertEqual(call_args[2], mock_filter_fn)

        # Verify unwrap_tensor_subclass was called once (with the copied model)
        self.assertEqual(mock_unwrap.call_count, 1)

        # Verify that the original models_dict is unchanged
        self.assertEqual(models_dict, {"forward": self.model})

        # Verify that the result artifact data contains valid models
        result_artifact = stage.get_artifacts()
        self.assertIn("forward", result_artifact.data)
        self.assertIsNotNone(result_artifact.data["forward"])
        # verify the result model is NOT the same object as the original
        self.assertIsNot(result_artifact.data["forward"], self.model)


class TestQuantizeStage(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.models_dict = {"forward": self.model}
        self.example_inputs = [(torch.randn(2, 10),)]
        self.context = {"example_inputs": {"forward": self.example_inputs}}

    @staticmethod
    def create_dummy_quantizer() -> TorchAOPT2EQuantizer:

        class DummyQuantizer(TorchAOPT2EQuantizer):
            def __init__(self):
                pass

            def annotate(self, model):
                return model

            def validate(self, model):
                pass

        return DummyQuantizer()

    def test_run_no_quantizers(self) -> None:
        """Test execution with no quantizers."""
        mock_recipe = Mock(spec=QuantizationRecipe)
        mock_recipe.quantizers = None
        stage = QuantizeStage(mock_recipe)
        artifact = PipelineArtifact(data=self.models_dict, context=self.context)
        stage.run(artifact)

        result_artifact = stage.get_artifacts()
        self.assertEqual(result_artifact, artifact)

    @patch("executorch.export.stages.convert_pt2e")
    @patch("executorch.export.stages.prepare_pt2e")
    @patch("executorch.export.stages.ComposableQuantizer")
    @patch("torch.export.export")
    def test_run_with_quantizers(
        self,
        mock_torch_export: Mock,
        mock_composable_quantizer: Mock,
        mock_prepare_pt2e: Mock,
        mock_convert_pt2e: Mock,
    ) -> None:
        """Test execution with quantizers"""
        mock_quantizer = self.create_dummy_quantizer()
        mock_recipe = Mock(spec=QuantizationRecipe)
        mock_recipe.quantizers = [mock_quantizer]
        stage = QuantizeStage(mock_recipe)

        # Mock the torch.export.export chain
        mock_exported_program = Mock(spec=ExportedProgram)
        mock_captured_graph = Mock()
        mock_exported_program.module.return_value = mock_captured_graph
        mock_torch_export.return_value = mock_exported_program

        # Mock the quantization chain
        mock_composed_quantizer = Mock()
        mock_composable_quantizer.return_value = mock_composed_quantizer
        mock_prepared_model = Mock()
        mock_prepare_pt2e.return_value = mock_prepared_model
        mock_quantized_model = Mock()
        mock_convert_pt2e.return_value = mock_quantized_model

        artifact = PipelineArtifact(data=self.models_dict, context=self.context)
        stage.run(artifact)

        # Verify torch.export.export was called
        mock_torch_export.assert_called_once_with(
            self.model, self.example_inputs[0], strict=True
        )

        # Verify ComposableQuantizer was created with the quantizers
        mock_composable_quantizer.assert_called_once_with([mock_quantizer])

        # Verify prepare_pt2e was called
        mock_prepare_pt2e.assert_called_once_with(
            mock_captured_graph, mock_composed_quantizer
        )

        # Verify calibration was performed (prepared model called with example inputs)
        mock_prepared_model.assert_called_once_with(*self.example_inputs[0])

        # Verify convert_pt2e was called
        mock_convert_pt2e.assert_called_once_with(mock_prepared_model)

        # Verify artifacts are returned correctly
        result_artifact = stage.get_artifacts()
        self.assertIn("forward", result_artifact.data)
        self.assertEqual(result_artifact.data["forward"], mock_quantized_model)

        # Verify that the original model in the input artifact is unchanged
        self.assertEqual(artifact.data["forward"], self.model)
        self.assertIsNot(result_artifact.data["forward"], self.model)

    def test_run_empty_example_inputs(self) -> None:
        """Test error when example inputs list is empty."""
        mock_quantizer = Mock()
        mock_recipe = Mock(spec=QuantizationRecipe)
        mock_recipe.quantizers = [mock_quantizer]
        stage = QuantizeStage(mock_recipe)
        context = {"example_inputs": {"forward": []}}
        artifact = PipelineArtifact(data=self.models_dict, context=context)

        with self.assertRaises(ValueError) as cm:
            stage.run(artifact)
        self.assertIn(
            "Example inputs for method forward not found or empty", str(cm.exception)
        )

    @patch("executorch.export.stages.ComposableQuantizer")
    def test_get_quantizer_for_prepare_pt2e(
        self, mock_composable_quantizer: Mock
    ) -> None:
        """Test _get_quantizer_for_prepare_pt2e method with different quantizer scenarios."""
        mock_recipe = Mock(spec=QuantizationRecipe)
        stage = QuantizeStage(mock_recipe)

        # Test empty quantizers list - should raise ValueError
        with self.assertRaises(ValueError) as cm:
            stage._get_quantizer_for_prepare_pt2e([])
        self.assertIn("No quantizers detected", str(cm.exception))

        # Test ComposableQuantizer path with multiple torchao quantizers
        # Create instances of dummy quantizers using the reusable method
        quantizer1 = self.create_dummy_quantizer()
        quantizer2 = self.create_dummy_quantizer()

        # Set up ComposableQuantizer mock
        mock_composed_quantizer = Mock()
        mock_composable_quantizer.return_value = mock_composed_quantizer

        # Call the method with multiple torchao quantizers
        result = stage._get_quantizer_for_prepare_pt2e([quantizer1, quantizer2])

        # Verify ComposableQuantizer was called with the quantizers
        mock_composable_quantizer.assert_called_once_with([quantizer1, quantizer2])
        self.assertEqual(result, mock_composed_quantizer)


class TestToEdgeStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_exported_program = Mock(spec=ExportedProgram)
        self.exported_programs = {"forward": self.mock_exported_program}
        self.context = {"constant_methods": None}

    @patch("executorch.export.stages.to_edge")
    def test_run_success(self, mock_to_edge: Mock) -> None:
        mock_edge_manager = Mock(spec=EdgeProgramManager)
        mock_to_edge.return_value = mock_edge_manager
        mock_config = Mock()

        stage = ToEdgeStage(edge_compile_config=mock_config)
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)
        stage.run(artifact)

        # Verify to_edge was called with correct parameters
        mock_to_edge.assert_called_once_with(
            self.exported_programs,
            constant_methods=None,
            compile_config=mock_config,
            generate_etrecord=False,
        )

        # Verify artifacts are set correctly
        result_artifact = stage.get_artifacts()
        self.assertEqual(result_artifact.data, mock_edge_manager)


class TestToBackendStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_edge_manager = Mock(spec=EdgeProgramManager)
        self.context = {}

    @patch("executorch.export.stages.get_delegation_info")
    def test_run_success_no_transforms_or_partitioners(
        self, mock_get_delegation_info: Mock
    ) -> None:
        # Test successful execution without transforms or partitioners
        mock_delegation_info = {"delegation": "info"}
        mock_get_delegation_info.return_value = mock_delegation_info
        mock_exported_program = Mock()
        mock_graph_module = Mock()
        mock_exported_program.graph_module = mock_graph_module

        self.mock_edge_manager.transform.return_value = self.mock_edge_manager
        self.mock_edge_manager.exported_program.return_value = mock_exported_program
        self.mock_edge_manager.methods = {"forward"}

        stage = ToBackendStage()
        artifact = PipelineArtifact(data=self.mock_edge_manager, context=self.context)
        stage.run(artifact)

        # Verify get_delegation_info was called
        mock_get_delegation_info.assert_called_once_with(mock_graph_module)

        # Verify artifacts are set correctly
        result_artifact = stage.get_artifacts()
        self.assertEqual(result_artifact.data, self.mock_edge_manager)
        self.assertEqual(
            result_artifact.get_context("delegation_info"), mock_delegation_info
        )

    def test_run_edge_manager_none(self) -> None:
        stage = ToBackendStage()
        artifact = PipelineArtifact(data=None, context=self.context)

        with self.assertRaises(RuntimeError) as cm:
            stage.run(artifact)
        self.assertIn("Edge program manager is not set", str(cm.exception))


class TestAtenTransformStage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_exported_program = Mock(spec=ExportedProgram)
        self.exported_programs = {"forward": self.mock_exported_program}
        self.context = {}

    def test_stage_properties(self) -> None:
        """Test that AtenTransformStage has correct properties."""
        stage = AtenTransformStage()

        self.assertEqual(stage.stage_type, StageType.ATEN_TRANSFORM)
        self.assertEqual(stage.valid_predecessor_stages, [StageType.TORCH_EXPORT])
        self.assertTrue(stage.can_start_pipeline)

    def test_run_no_passes_passthrough(self) -> None:
        """Test that stage passes through when no aten_transform_passes are configured."""
        stage = AtenTransformStage()
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)

        stage.run(artifact)

        result_artifact = stage.get_artifacts()
        # Should pass through the same artifact unchanged
        self.assertEqual(result_artifact, artifact)

    def test_run_with_empty_passes_list(self) -> None:
        """Test that stage passes through when aten_transform_passes is empty list."""
        stage = AtenTransformStage(aten_transform_passes=[])
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)

        stage.run(artifact)

        result_artifact = stage.get_artifacts()
        # Should pass through the same artifact unchanged
        self.assertEqual(result_artifact, artifact)

    def test_run_with_single_pass(self) -> None:
        """Test that stage applies a single aten_transform_pass correctly."""
        mock_transformed_program = Mock(spec=ExportedProgram)

        # Create a mock pass function
        mock_pass = Mock()
        mock_pass.return_value = mock_transformed_program

        stage = AtenTransformStage(aten_transform_passes=[mock_pass])
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)

        stage.run(artifact)

        # Verify the pass was called with correct arguments
        mock_pass.assert_called_once_with("forward", self.mock_exported_program)

        # Verify artifacts contain the transformed program
        result_artifact = stage.get_artifacts()
        self.assertIn("forward", result_artifact.data)
        self.assertEqual(result_artifact.data["forward"], mock_transformed_program)

    def test_run_with_multiple_passes(self) -> None:
        """Test that stage applies multiple aten_transform_passes in order."""
        mock_intermediate_program = Mock(spec=ExportedProgram)
        mock_final_program = Mock(spec=ExportedProgram)

        # Create mock pass functions
        mock_pass1 = Mock()
        mock_pass1.return_value = mock_intermediate_program
        mock_pass2 = Mock()
        mock_pass2.return_value = mock_final_program

        stage = AtenTransformStage(aten_transform_passes=[mock_pass1, mock_pass2])
        artifact = PipelineArtifact(data=self.exported_programs, context=self.context)

        stage.run(artifact)

        # Verify first pass was called with original program
        mock_pass1.assert_called_once_with("forward", self.mock_exported_program)
        # Verify second pass was called with result of first pass
        mock_pass2.assert_called_once_with("forward", mock_intermediate_program)

        # Verify artifacts contain the final transformed program
        result_artifact = stage.get_artifacts()
        self.assertIn("forward", result_artifact.data)
        self.assertEqual(result_artifact.data["forward"], mock_final_program)

    def test_run_with_multiple_methods(self) -> None:
        """Test that stage applies passes to all methods."""
        mock_program1 = Mock(spec=ExportedProgram)
        mock_program2 = Mock(spec=ExportedProgram)
        mock_transformed1 = Mock(spec=ExportedProgram)
        mock_transformed2 = Mock(spec=ExportedProgram)

        exported_programs = {"forward": mock_program1, "encode": mock_program2}

        # Create mock pass function that returns different results
        mock_pass = Mock(side_effect=[mock_transformed1, mock_transformed2])

        stage = AtenTransformStage(aten_transform_passes=[mock_pass])
        artifact = PipelineArtifact(data=exported_programs, context=self.context)

        stage.run(artifact)

        # Verify the pass was called for each method
        self.assertEqual(mock_pass.call_count, 2)

        # Verify artifacts contain all transformed programs
        result_artifact = stage.get_artifacts()
        self.assertEqual(len(result_artifact.data), 2)

    def test_from_recipe_with_none(self) -> None:
        """Test from_recipe with None recipe."""
        stage = AtenTransformStage.from_recipe(None)

        self.assertIsInstance(stage, AtenTransformStage)
        self.assertEqual(stage._aten_transform_passes, [])

    def test_from_recipe_with_recipe(self) -> None:
        """Test from_recipe with recipe containing aten_transform_passes."""
        mock_pass = Mock()
        mock_recipe = Mock()
        mock_recipe.aten_transform_passes = [mock_pass]

        stage = AtenTransformStage.from_recipe(mock_recipe)

        self.assertIsInstance(stage, AtenTransformStage)
        self.assertEqual(stage._aten_transform_passes, [mock_pass])

    def test_from_recipe_without_aten_transform_passes_attr(self) -> None:
        """Test from_recipe with recipe that doesn't have aten_transform_passes attribute."""
        mock_recipe = Mock(spec=[])  # Empty spec means no attributes

        stage = AtenTransformStage.from_recipe(mock_recipe)

        self.assertIsInstance(stage, AtenTransformStage)
        self.assertEqual(stage._aten_transform_passes, [])

    def test_get_artifacts_before_run(self) -> None:
        """Test error when getting artifacts before running stage."""
        stage = AtenTransformStage()

        with self.assertRaises(RuntimeError) as cm:
            stage.get_artifacts()
        self.assertIn("Stage: AtenTransformStage not executed", str(cm.exception))
