# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from unittest.mock import MagicMock

import torch

from executorch.extension.llm.export.builder import DType, LLMEdgeManager


class TestLLMEdgeManager(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock model
        self.mock_model = MagicMock()
        self.modelname = "test_model"
        self.max_seq_len = 2048
        self.dtype = DType.fp32
        self.example_inputs = (torch.zeros((1, 10), dtype=torch.long),)
        self.example_kwarg_inputs = {"input_pos": torch.tensor([0])}

    def test_get_dynamic_shape_with_preset_dynamic_shapes(self) -> None:
        """Test that _get_dynamic_shape returns preset dynamic_shapes if available."""
        # Create a manager with preset dynamic_shapes
        preset_dynamic_shapes = {"preset": "shapes"}
        manager = LLMEdgeManager(
            model=self.mock_model,
            modelname=self.modelname,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            use_kv_cache=False,
            example_inputs=self.example_inputs,
            dynamic_shapes=preset_dynamic_shapes,
        )

        # Call _get_dynamic_shape and verify it returns the preset value
        result = manager._get_dynamic_shape()
        self.assertEqual(result, preset_dynamic_shapes)

    def test_get_dynamic_shape_with_dynamic_shape_enabled_no_kv_cache(self) -> None:
        """Test _get_dynamic_shape when enable_dynamic_shape=True and use_kv_cache=False."""
        # Create a manager with enable_dynamic_shape=True and use_kv_cache=False
        manager = LLMEdgeManager(
            model=self.mock_model,
            modelname=self.modelname,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            use_kv_cache=False,
            example_inputs=self.example_inputs,
            enable_dynamic_shape=True,
        )

        # Call _get_dynamic_shape
        result = manager._get_dynamic_shape()

        # Verify the result has the expected structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dict)
        self.assertIn(1, result[0])
        # Check that the value at key 1 is a torch.export.Dim with the correct max value
        self.assertEqual(result[0][1].max, self.max_seq_len - 1)

    def test_get_dynamic_shape_with_dynamic_shape_enabled_with_kv_cache(self) -> None:
        """Test _get_dynamic_shape when enable_dynamic_shape=True and use_kv_cache=True."""
        # Create a manager with enable_dynamic_shape=True and use_kv_cache=True
        manager = LLMEdgeManager(
            model=self.mock_model,
            modelname=self.modelname,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            use_kv_cache=True,
            example_inputs=self.example_inputs,
            enable_dynamic_shape=True,
        )

        # Call _get_dynamic_shape
        result = manager._get_dynamic_shape()

        # Verify the result has the expected structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check first element (tokens dimension)
        self.assertIsInstance(result[0], dict)
        self.assertIn(1, result[0])
        self.assertEqual(result[0][1].max, self.max_seq_len - 1)

        # Check second element (input_pos dimension)
        self.assertIsInstance(result[1], dict)
        self.assertIn("input_pos", result[1])
        self.assertIsInstance(result[1]["input_pos"], dict)
        self.assertIn(0, result[1]["input_pos"])
        self.assertEqual(result[1]["input_pos"][0], 1)

    def test_get_dynamic_shape_with_dynamic_shape_disabled(self) -> None:
        """Test _get_dynamic_shape when enable_dynamic_shape=False."""
        # Create a manager with enable_dynamic_shape=False
        manager = LLMEdgeManager(
            model=self.mock_model,
            modelname=self.modelname,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            use_kv_cache=True,  # Doesn't matter for this test
            example_inputs=self.example_inputs,
            enable_dynamic_shape=False,
        )

        # Call _get_dynamic_shape
        result = manager._get_dynamic_shape()

        # Verify the result is None
        self.assertIsNone(result)
