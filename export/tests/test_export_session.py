# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import Mock

import torch
from executorch.export import ExportRecipe, ExportSession
from executorch.export.recipe import LoweringRecipe, QuantizationRecipe
from executorch.export.stages import PipelineArtifact, StageType


class SimpleTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestExportSessionCoreFlow(unittest.TestCase):
    """Test core export flow and pipeline execution."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    def _create_mock_stage(self, stage_type: StageType) -> Mock:
        mock_stage = Mock()
        mock_artifact = Mock(spec=PipelineArtifact)
        mock_artifact.data = Mock()
        mock_artifact.context = {}
        mock_stage.get_artifacts.return_value = mock_artifact
        mock_stage.stage_type = stage_type
        return mock_stage

    def test_default_pipeline_execution_order(self) -> None:
        # Test that pipeline stages are executed in the correct order
        stage_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        mock_stages = [
            self._create_mock_stage(stage_type) for stage_type in stage_types
        ]

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        # Replace the pipeline with our mocked stages
        session._pipeline = mock_stages

        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 5)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

    def test_overriden_pipeline_execution_order(self) -> None:
        # Test when pipeline stages that are passed to export function
        stage_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        mock_stages = [
            self._create_mock_stage(stage_type) for stage_type in stage_types
        ]

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
            pipeline_stages=stage_types,
        )

        session._pipeline = mock_stages
        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 4)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

    def test_pipeline_validation_enforces_mandatory_stages(self) -> None:
        # Test missing mandatory TORCH_EXPORT stage
        invalid_stages = [StageType.TO_EXECUTORCH]

        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=self.model,
                example_inputs=self.example_inputs,
                export_recipe=self.recipe,
                pipeline_stages=invalid_stages,
            )._run_pipeline()

        self.assertIn("Invalid pipeline sequence", str(cm.exception))

    def test_pipeline_validation_enforces_valid_transitions(self) -> None:
        # Test invalid transition: TORCH_EXPORT -> TO_EXECUTORCH (skipping edge stage)
        invalid_stages = [StageType.TORCH_EXPORT, StageType.TO_EXECUTORCH]

        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=self.model,
                example_inputs=self.example_inputs,
                export_recipe=self.recipe,
                pipeline_stages=invalid_stages,
            )._run_pipeline()

        self.assertIn("Invalid pipeline sequence", str(cm.exception))

    def test_model_standardization_single_to_dict(self) -> None:
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        self.assertIsInstance(session._model, dict)
        self.assertIn("forward", session._model)
        self.assertEqual(session._model["forward"], self.model)

        self.assertIsInstance(session._example_inputs, dict)
        self.assertIn("forward", session._example_inputs)
        self.assertEqual(session._example_inputs["forward"], self.example_inputs)

    def test_model_standardization_preserves_dict(self) -> None:
        # Test that dictionary models are preserved as-is.
        model_dict = {"method1": self.model, "method2": SimpleTestModel()}
        inputs_dict = {
            "method1": self.example_inputs,
            "method2": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,  # pyre-ignore[6]
            example_inputs=inputs_dict,
            export_recipe=self.recipe,
        )

        self.assertEqual(session._model, model_dict)
        self.assertEqual(session._example_inputs, inputs_dict)

    def test_context_propagation_through_pipeline(self) -> None:
        # Test that context is properly propagated through the pipeline
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
            name="test_session",
            constant_methods={"const_method": lambda: torch.tensor([1, 2, 3])},
        )

        # Check that initial context is set up correctly
        expected_context_keys = {
            "example_inputs",
            "dynamic_shapes",
            "constant_methods",
            "export_recipe",
            "session_name",
            "artifact_dir",
        }
        self.assertEqual(set(session._run_context.keys()), expected_context_keys)
        self.assertEqual(session._run_context["session_name"], "test_session")
        self.assertIsNotNone(session._run_context["constant_methods"])

    def test_pipeline_building_unknown_stage_type(self) -> None:
        # Test error handling for unknown stage types
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=ExportRecipe(name="test"),
        )

        # Mock an unknown stage type
        unknown_stage = "UNKNOWN_STAGE"

        with self.assertRaises(ValueError) as cm:
            session._build_pipeline_from_stages([unknown_stage])  # pyre-ignore
        self.assertIn("Unknown stage type", str(cm.exception))

    def test_multi_method_model_export(self) -> None:
        # Test export with multi-method models
        model_dict = {
            "forward": self.model,
            "inference": SimpleTestModel(),
        }
        inputs_dict = {
            "forward": self.example_inputs,
            "inference": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,  # pyre-ignore[6]
            example_inputs=inputs_dict,
            export_recipe=ExportRecipe(name="multi_method_test"),
        )

        # Verify proper initialization
        self.assertEqual(session._model, model_dict)
        self.assertEqual(session._example_inputs, inputs_dict)

        # Test getting example inputs for different methods
        forward_input = session.get_example_input("forward")
        inference_input = session.get_example_input("inference")

        self.assertEqual(forward_input, self.example_inputs[0])
        self.assertEqual(inference_input, inputs_dict["inference"][0])


class TestExportSessionErrorHandling(unittest.TestCase):
    """Test error handling in export session."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    def test_access_results_before_export(self) -> None:
        """Test that accessing results before export raises appropriate errors."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_executorch_program_manager()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_executorch_program()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_pte_buffer()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

    def test_invalid_method_name_in_example_inputs(self) -> None:
        """Test error handling for invalid method names."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(KeyError) as cm:
            session.get_example_input("nonexistent_method")
        self.assertIn("Method name 'nonexistent_method' not found", str(cm.exception))

    def test_empty_example_inputs_list(self) -> None:
        """Test error handling for empty example inputs."""
        session = ExportSession(
            model={"forward": self.model},
            example_inputs={"forward": []},
            export_recipe=self.recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.get_example_input("forward")
        self.assertIn(
            "Example inputs list for method forward is empty", str(cm.exception)
        )

    def test_save_to_pte_invalid_name(self) -> None:
        """Test save_to_pte with invalid output name."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(AssertionError):
            session.save_to_pte("")

        with self.assertRaises(AssertionError):
            session.save_to_pte(None)  # pyre-ignore


class TestExportSessionPipelineBuilding(unittest.TestCase):
    """Test pipeline building and stage configuration."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    def test_pipeline_building_with_all_recipes(self) -> None:
        """Test pipeline building with quantization and lowering recipes."""
        # Create comprehensive recipes
        quant_recipe = QuantizationRecipe(
            ao_base_config=[Mock()],
            quantizers=[Mock()],
        )
        lowering_recipe = LoweringRecipe(
            partitioners=[Mock()],
            edge_transform_passes=[Mock()],
            edge_compile_config=Mock(),
        )
        recipe = ExportRecipe(
            name="comprehensive_test",
            quantization_recipe=quant_recipe,
            lowering_recipe=lowering_recipe,
            pre_edge_transform_passes=[Mock()],
            executorch_backend_config=Mock(),
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        self.assertEqual(len(session._pipeline), 5)
        stage_types = [stage.stage_type for stage in session._pipeline]
        expected_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        self.assertEqual(stage_types, expected_types)
