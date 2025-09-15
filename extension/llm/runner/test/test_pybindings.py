#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for the ExecuTorch LLM Runner Python bindings.

To run these tests:
    python -m pytest test_pybindings.py -v
"""

import os
import tempfile
import unittest

import numpy as np
from executorch.extension.llm.runner import (
    GenerationConfig,
    Image,
    make_image_input,
    make_text_input,
    MultimodalInput,
    MultimodalRunner,
    Stats,
)


class TestGenerationConfig(unittest.TestCase):
    """Test the GenerationConfig class."""

    def test_default_values(self):
        """Test that GenerationConfig has correct default values."""
        config = GenerationConfig()

        # Check defaults based on irunner.h
        self.assertEqual(config.echo, True)
        self.assertEqual(config.max_new_tokens, -1)
        self.assertEqual(config.warming, False)
        self.assertEqual(config.seq_len, -1)
        self.assertAlmostEqual(config.temperature, 0.8, places=5)
        self.assertEqual(config.num_bos, 0)
        self.assertEqual(config.num_eos, 0)

    def test_set_values(self):
        """Test setting values on GenerationConfig."""
        config = GenerationConfig()

        config.echo = False
        config.max_new_tokens = 100
        config.warming = True
        config.seq_len = 512
        config.temperature = 0.5
        config.num_bos = 1
        config.num_eos = 2

        self.assertEqual(config.echo, False)
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.warming, True)
        self.assertEqual(config.seq_len, 512)
        self.assertAlmostEqual(config.temperature, 0.5, places=5)
        self.assertEqual(config.num_bos, 1)
        self.assertEqual(config.num_eos, 2)

    def test_resolve_max_new_tokens(self):
        """Test the resolve_max_new_tokens method."""
        config = GenerationConfig()

        # Test case 1: Both seq_len and max_new_tokens are -1
        config.seq_len = -1
        config.max_new_tokens = -1
        result = config.resolve_max_new_tokens(1024, 100)
        self.assertEqual(result, 924)  # 1024 - 100

        # Test case 2: Only max_new_tokens is specified
        config.seq_len = -1
        config.max_new_tokens = 200
        result = config.resolve_max_new_tokens(1024, 100)
        self.assertEqual(result, 200)  # min(200, 1024-100)

        # Test case 3: Only seq_len is specified
        config.seq_len = 512
        config.max_new_tokens = -1
        result = config.resolve_max_new_tokens(1024, 100)
        self.assertEqual(result, 412)  # min(512, 1024) - 100

        # Test case 4: Both are specified
        config.seq_len = 512
        config.max_new_tokens = 200
        result = config.resolve_max_new_tokens(1024, 100)
        self.assertEqual(result, 200)  # min(min(512, 1024) - 100, 200)

        # Test case 5: Result would be negative
        config.seq_len = 50
        config.max_new_tokens = -1
        result = config.resolve_max_new_tokens(1024, 100)
        self.assertEqual(result, 0)  # max(0, 50 - 100)

    def test_repr(self):
        """Test the string representation."""
        config = GenerationConfig()
        config.max_new_tokens = 100
        config.seq_len = 512
        config.temperature = 0.7

        repr_str = repr(config)
        self.assertIn("GenerationConfig", repr_str)
        self.assertIn("max_new_tokens=100", repr_str)
        self.assertIn("seq_len=512", repr_str)
        self.assertIn("temperature=0.7", repr_str)
        self.assertIn("echo=True", repr_str)
        self.assertIn("warming=False", repr_str)


class TestStats(unittest.TestCase):
    """Test the Stats class."""

    def test_attributes(self):
        """Test that Stats has all expected attributes."""
        stats = Stats()

        # Check all timing attributes exist
        self.assertTrue(hasattr(stats, "SCALING_FACTOR_UNITS_PER_SECOND"))
        self.assertTrue(hasattr(stats, "model_load_start_ms"))
        self.assertTrue(hasattr(stats, "model_load_end_ms"))
        self.assertTrue(hasattr(stats, "inference_start_ms"))
        self.assertTrue(hasattr(stats, "token_encode_end_ms"))
        self.assertTrue(hasattr(stats, "model_execution_start_ms"))
        self.assertTrue(hasattr(stats, "model_execution_end_ms"))
        self.assertTrue(hasattr(stats, "prompt_eval_end_ms"))
        self.assertTrue(hasattr(stats, "first_token_ms"))
        self.assertTrue(hasattr(stats, "inference_end_ms"))
        self.assertTrue(hasattr(stats, "aggregate_sampling_time_ms"))
        self.assertTrue(hasattr(stats, "num_prompt_tokens"))
        self.assertTrue(hasattr(stats, "num_generated_tokens"))

    def test_scaling_factor(self):
        """Test the scaling factor constant."""
        stats = Stats()
        self.assertEqual(stats.SCALING_FACTOR_UNITS_PER_SECOND, 1000)

    def test_methods(self):
        """Test Stats methods."""
        stats = Stats()

        # Test on_sampling_begin and on_sampling_end
        stats.on_sampling_begin()
        stats.on_sampling_end()

        # Test reset without all_stats
        stats.model_load_start_ms = 100
        stats.model_load_end_ms = 200
        stats.inference_start_ms = 300
        stats.num_prompt_tokens = 10
        stats.num_generated_tokens = 20

        stats.reset(False)

        # Model load times should be preserved
        self.assertEqual(stats.model_load_start_ms, 100)
        self.assertEqual(stats.model_load_end_ms, 200)
        # Other stats should be reset
        self.assertEqual(stats.inference_start_ms, 0)
        self.assertEqual(stats.num_prompt_tokens, 0)
        self.assertEqual(stats.num_generated_tokens, 0)

        # Test reset with all_stats
        stats.reset(True)
        self.assertEqual(stats.model_load_start_ms, 0)
        self.assertEqual(stats.model_load_end_ms, 0)

    def test_to_json_string(self):
        """Test JSON string conversion."""
        stats = Stats()
        stats.num_prompt_tokens = 10
        stats.num_generated_tokens = 20
        stats.model_load_start_ms = 100
        stats.model_load_end_ms = 200
        stats.inference_start_ms = 300
        stats.inference_end_ms = 1300

        json_str = stats.to_json_string()
        self.assertIn('"prompt_tokens":10', json_str)
        self.assertIn('"generated_tokens":20', json_str)
        self.assertIn('"model_load_start_ms":100', json_str)
        self.assertIn('"model_load_end_ms":200', json_str)

    def test_repr(self):
        """Test string representation."""
        stats = Stats()
        stats.num_prompt_tokens = 10
        stats.num_generated_tokens = 20
        stats.inference_start_ms = 1000
        stats.inference_end_ms = 2000

        repr_str = repr(stats)
        self.assertIn("Stats", repr_str)
        self.assertIn("num_prompt_tokens=10", repr_str)
        self.assertIn("num_generated_tokens=20", repr_str)
        self.assertIn("tokens_per_second=20", repr_str)  # 20 tokens / 1 second


class TestImage(unittest.TestCase):
    """Test the Image class."""

    def test_creation(self):
        """Test creating an Image object."""
        image = Image()

        # Set properties
        image.data = [1, 2, 3, 4]
        image.width = 2
        image.height = 2
        image.channels = 1

        self.assertEqual(image.data, [1, 2, 3, 4])
        self.assertEqual(image.width, 2)
        self.assertEqual(image.height, 2)
        self.assertEqual(image.channels, 1)

    def test_repr(self):
        """Test string representation."""
        image = Image()
        image.width = 640
        image.height = 480
        image.channels = 3

        repr_str = repr(image)
        self.assertIn("Image", repr_str)
        self.assertIn("height=480", repr_str)
        self.assertIn("width=640", repr_str)
        self.assertIn("channels=3", repr_str)


class TestMultimodalInput(unittest.TestCase):
    """Test the MultimodalInput class."""

    def test_text_input(self):
        """Test creating a text MultimodalInput."""
        # Test direct constructor
        text_input = MultimodalInput("Hello, world!")
        self.assertTrue(text_input.is_text())
        self.assertFalse(text_input.is_image())
        self.assertEqual(text_input.get_text(), "Hello, world!")

        # Test helper function
        text_input2 = make_text_input("Test text")
        self.assertTrue(text_input2.is_text())
        self.assertEqual(text_input2.get_text(), "Test text")

    def test_image_input(self):
        """Test creating an image MultimodalInput."""
        # Create an image
        image = Image()
        image.data = [255] * (100 * 100 * 3)
        image.width = 100
        image.height = 100
        image.channels = 3

        # Test direct constructor
        image_input = MultimodalInput(image)
        self.assertTrue(image_input.is_image())
        self.assertFalse(image_input.is_text())

        # Test helper function with numpy array
        img_array = np.ones((50, 60, 3), dtype=np.uint8) * 128
        image_input2 = make_image_input(img_array)
        self.assertTrue(image_input2.is_image())
        self.assertFalse(image_input2.is_text())

    def test_invalid_image_array(self):
        """Test error handling for invalid image arrays."""
        # Wrong dimensions
        with self.assertRaises(RuntimeError) as cm:
            make_image_input(np.ones((100,), dtype=np.uint8))
        self.assertIn("3-dimensional", str(cm.exception))

        # Wrong number of channels
        with self.assertRaises(RuntimeError) as cm:
            make_image_input(np.ones((100, 100, 2), dtype=np.uint8))
        self.assertIn("3 (RGB) or 4 (RGBA)", str(cm.exception))

    def test_repr(self):
        """Test string representation."""
        # Text input
        text_input = MultimodalInput("This is a test")
        repr_str = repr(text_input)
        self.assertIn("MultimodalInput", repr_str)
        self.assertIn("type=text", repr_str)
        self.assertIn("This is a test", repr_str)

        # Long text input (should be truncated)
        long_text = "a" * 100
        text_input2 = MultimodalInput(long_text)
        repr_str2 = repr(text_input2)
        self.assertIn("...", repr_str2)

        # Image input
        image = Image()
        image_input = MultimodalInput(image)
        repr_str3 = repr(image_input)
        self.assertIn("type=image", repr_str3)


class TestMultimodalRunner(unittest.TestCase):
    """Test the MultimodalRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pte")
        self.tokenizer_path = os.path.join(self.temp_dir, "tokenizer.bin")

        # Create dummy files (these won't actually work, but we can test initialization failure)
        with open(self.model_path, "wb") as f:
            f.write(b"dummy model")
        with open(self.tokenizer_path, "wb") as f:
            f.write(b"dummy tokenizer")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_failure(self):
        """Test that initialization fails gracefully with invalid files."""
        with self.assertRaises(RuntimeError) as cm:
            runner = MultimodalRunner(self.model_path, self.tokenizer_path)
        # Should fail because the tokenizer file is not valid
        self.assertIn("Failed to", str(cm.exception))


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_make_text_input(self):
        """Test make_text_input helper."""
        text_input = make_text_input("Hello")
        self.assertTrue(text_input.is_text())
        self.assertEqual(text_input.get_text(), "Hello")

    def test_make_image_input(self):
        """Test make_image_input helper."""
        # Create a test image array (RGB)
        img_array = np.zeros((100, 150, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel

        image_input = make_image_input(img_array)
        self.assertTrue(image_input.is_image())

        # Test with RGBA
        img_array_rgba = np.ones((50, 50, 4), dtype=np.uint8) * 128
        image_input_rgba = make_image_input(img_array_rgba)
        self.assertTrue(image_input_rgba.is_image())
