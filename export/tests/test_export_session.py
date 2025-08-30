# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List
from unittest.mock import Mock

import torch
from executorch.export import ExportRecipe, ExportSession
from executorch.export.recipe import (
    AOQuantizationConfig,
    LoweringRecipe,
    QuantizationRecipe,
)
from executorch.export.stages import PipelineArtifact
from executorch.export.types import StageType


class SimpleTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(10, 5)

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

        # Add the new properties required by the Stage interface
        if stage_type == StageType.SOURCE_TRANSFORM:
            mock_stage.valid_predecessor_stages = []
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.QUANTIZE:
            mock_stage.valid_predecessor_stages = [StageType.SOURCE_TRANSFORM]
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.TORCH_EXPORT:
            mock_stage.valid_predecessor_stages = [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
            ]
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.TO_EDGE_TRANSFORM_AND_LOWER:
            mock_stage.valid_predecessor_stages = [StageType.TORCH_EXPORT]
            mock_stage.can_start_pipeline = False
        elif stage_type == StageType.TO_EXECUTORCH:
            mock_stage.valid_predecessor_stages = [
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ]
            mock_stage.can_start_pipeline = True
        else:
            mock_stage.valid_predecessor_stages = []
            mock_stage.can_start_pipeline = True

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

        # Replace the stages in the registry with our mocked stages
        for stage_type, mock_stage in zip(stage_types, mock_stages):
            session.register_stage(stage_type, mock_stage)

        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 5)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

    def test_overriden_pipeline_execution_order(self) -> None:
        # Test when pipeline stages that are passed through recipe
        stage_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        mock_stages = [
            self._create_mock_stage(stage_type) for stage_type in stage_types
        ]

        self.recipe.pipeline_stages = stage_types
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        # Replace the stages in the registry with our mocked stages
        for stage_type, mock_stage in zip(stage_types, mock_stages):
            session.register_stage(stage_type, mock_stage)
        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 4)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

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
            "generate_etrecord",
        }
        self.assertEqual(set(session._run_context.keys()), expected_context_keys)
        self.assertEqual(session._run_context["session_name"], "test_session")
        self.assertIsNotNone(session._run_context["constant_methods"])

    def test_stage_registry_unknown_stage_type(self) -> None:
        # Test error handling for unknown stage types in pipeline
        unknown_stage_type = Mock()
        unknown_stage_type.name = "UNKNOWN_STAGE"
        recipe = ExportRecipe(name="test", pipeline_stages=[unknown_stage_type])

        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=self.model,
                example_inputs=self.example_inputs,
                export_recipe=recipe,
            )._run_pipeline()
        self.assertIn("not found in registry", str(cm.exception))

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


class TestPipelineValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    # pyre-ignore
    def _get_export_session(self, stages: List[StageType]):
        self.recipe.pipeline_stages = stages
        return ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

    def test_valid_pipeline_sequences(self) -> None:
        """Test various valid pipeline sequences."""
        valid_sequences = [
            # Full pipeline with to_edge_transform_lower
            [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Full pipeline with to_edge, to_backend
            [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
            # Skip quantize
            [
                StageType.SOURCE_TRANSFORM,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Skip source transform and tart with quantize
            [
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Start with torch export
            [
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
        ]

        for i, stages in enumerate(valid_sequences):
            with self.subTest(sequence=i, stages=[s.name for s in stages]):
                session = self._get_export_session(stages)
                # Should not raise any exception
                try:
                    session._validate_pipeline_sequence(stages)
                except Exception as e:
                    self.fail(f"Valid sequence {[s.name for s in stages]} raised {e}")

    def test_invalid_pipeline_start_stages(self) -> None:
        """Test stages that cannot start a pipeline."""
        invalid_stage_sequence = [
            # Edge stage cannot start pipeline
            [StageType.TO_EDGE_TRANSFORM_AND_LOWER],
            [StageType.TO_EDGE_TRANSFORM_AND_LOWER, StageType.TO_EXECUTORCH],
        ]

        for i, stages in enumerate(invalid_stage_sequence):
            with self.subTest(sequence=i, stages=[s.name for s in stages]):
                session = self._get_export_session(stages)
                with self.assertRaises(ValueError) as cm:
                    session._validate_pipeline_sequence(stages)
                self.assertIn("cannot start a pipeline", str(cm.exception))

    def test_pipeline_transitions(self) -> None:
        """Test both valid and invalid pipeline transitions"""
        test_cases = [
            # Valid cases
            ([StageType.SOURCE_TRANSFORM, StageType.QUANTIZE], True),
            ([StageType.QUANTIZE, StageType.TORCH_EXPORT], True),
            ([StageType.SOURCE_TRANSFORM, StageType.TORCH_EXPORT], True),
            ([StageType.TORCH_EXPORT, StageType.TO_EDGE_TRANSFORM_AND_LOWER], True),
            # Invalid cases - transitions
            ([StageType.QUANTIZE, StageType.TO_EDGE_TRANSFORM_AND_LOWER], False),
            (
                [StageType.SOURCE_TRANSFORM, StageType.TO_EDGE_TRANSFORM_AND_LOWER],
                False,
            ),
            (
                [
                    StageType.TORCH_EXPORT,
                    StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                    StageType.QUANTIZE,
                ],
                False,
            ),
            ([StageType.TO_EXECUTORCH, StageType.TORCH_EXPORT], False),
        ]

        for i, (stages, should_pass) in enumerate(test_cases):
            with self.subTest(
                sequence=i, stages=[s.name for s in stages], should_pass=should_pass
            ):
                session = self._get_export_session(stages)
                if should_pass:
                    try:
                        session._validate_pipeline_sequence(stages)
                    except Exception as e:
                        self.fail(
                            f"Expected valid sequence {[s.name for s in stages]} but got {e}"
                        )
                else:
                    with self.assertRaises(ValueError):
                        session._validate_pipeline_sequence(stages)

    def test_empty_pipeline_sequence(self) -> None:
        """Test empty pipeline sequence."""
        session = self._get_export_session([])
        with self.assertRaises(ValueError) as cm:
            session._validate_pipeline_sequence([])
        self.assertIn("Pipeline stages cannot be empty", str(cm.exception))


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
            ao_quantization_configs=[AOQuantizationConfig(Mock())],
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
            executorch_backend_config=Mock(),
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        registered_stages = session.get_all_registered_stages()

        self.assertEqual(len(registered_stages), 5)
        expected_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        self.assertListEqual(list(registered_stages.keys()), expected_types)
