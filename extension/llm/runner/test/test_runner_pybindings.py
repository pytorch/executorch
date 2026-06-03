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

import threading
import unittest

import torch
from executorch.extension.llm.runner import (
    GenerationConfig,
    Image,
    LLMEngine,
    LLMSession,
    make_image_input,
    make_text_input,
    MultimodalInput,
    MultimodalRunner,
    TextLLMRunner,
)


class TestSessionApiBoundary(unittest.TestCase):
    """The Python serving boundary: token-step primitives live ONLY on
    LLMSession, never on the legacy TextLLMRunner (whose token-step methods are
    C++ implementation details behind TextLLMSession). Pure class introspection,
    so no model/.pte is needed."""

    TOKEN_STEP = ("prefill_tokens", "decode_one", "seek", "position")

    def test_text_llm_runner_does_not_expose_token_step(self):
        for name in self.TOKEN_STEP:
            self.assertFalse(
                hasattr(TextLLMRunner, name),
                f"TextLLMRunner must not expose token-step method {name!r} to "
                f"Python; drive sessions through LLMSession instead.",
            )

    def test_llm_session_exposes_token_step(self):
        for name in (*self.TOKEN_STEP, "reset", "stop"):
            self.assertTrue(
                hasattr(LLMSession, name), f"LLMSession must expose {name!r}"
            )

    def test_llm_engine_exposes_serving_api(self):
        for name in ("create_session", "serving_capacity"):
            self.assertTrue(hasattr(LLMEngine, name), f"LLMEngine must expose {name!r}")


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

        # Test case 6: Use keyword argument with new name
        config.seq_len = -1
        config.max_new_tokens = -1
        result = config.resolve_max_new_tokens(
            max_context_len=1024, num_tokens_occupied=100
        )
        self.assertEqual(result, 924)

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


class TestImage(unittest.TestCase):
    """Test the Image class."""

    def test_creation(self):
        """Test creating an Image object."""
        # Construct using binding constructor (uint8 data)
        image = Image([1, 2, 3, 4], 2, 2, 1)

        # Properties are read-only
        self.assertEqual(image.uint8_data, [1, 2, 3, 4])
        self.assertEqual(image.width, 2)
        self.assertEqual(image.height, 2)
        self.assertEqual(image.channels, 1)

    def test_repr(self):
        """Test string representation."""
        image = Image([0] * (480 * 640 * 3), 640, 480, 3)

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
        image = Image([255] * (100 * 100 * 3), 100, 100, 3)

        # Test direct constructor
        image_input = MultimodalInput(image)
        self.assertTrue(image_input.is_image())
        self.assertFalse(image_input.is_text())

        # Test helper function with torch tensor (CHW)
        img_tensor = torch.ones((3, 50, 60), dtype=torch.uint8) * 128
        image_input2 = make_image_input(img_tensor)
        self.assertTrue(image_input2.is_image())
        self.assertFalse(image_input2.is_text())

    def test_invalid_image_array(self):
        """Test error handling for invalid image arrays."""
        # Wrong dimensions (expects 3D or 4D tensor)
        with self.assertRaises(RuntimeError) as cm:
            make_image_input(torch.ones((100,), dtype=torch.uint8))
        self.assertIn("3-dimensional", str(cm.exception))

        # Wrong number of channels
        with self.assertRaises(RuntimeError) as cm:
            make_image_input(torch.ones((2, 100, 100), dtype=torch.uint8))
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
        image = Image([0, 0, 0], 1, 1, 3)
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
            MultimodalRunner(self.model_path, self.tokenizer_path, None)
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
        # Create a test image tensor (RGB, CHW)
        img_tensor = torch.zeros((3, 100, 150), dtype=torch.uint8)
        img_tensor[0, :, :] = 255  # Red channel

        image_input = make_image_input(img_tensor)
        self.assertTrue(image_input.is_image())

        # Test with RGBA (CHW)
        img_tensor_rgba = torch.ones((4, 50, 50), dtype=torch.uint8) * 128
        image_input_rgba = make_image_input(img_tensor_rgba)
        self.assertTrue(image_input_rgba.is_image())


# Real-engine tests need a model; gated on env vars so they're skipped in CI
# environments without one (fake-runner unit tests can't exercise the real
# shared-Program / serialization behavior).
_MODEL = os.environ.get("ET_TEST_TEXT_LLM_MODEL")
_TOKENIZER = os.environ.get("ET_TEST_TEXT_LLM_TOKENIZER")


@unittest.skipUnless(
    _MODEL and _TOKENIZER,
    "set ET_TEST_TEXT_LLM_MODEL and ET_TEST_TEXT_LLM_TOKENIZER to run",
)
class TestLLMEngineSessions(unittest.TestCase):
    """LLMEngine: sessions share weights, stay isolated, and serialize backend
    execution so concurrent sessions don't corrupt each other."""

    @classmethod
    def setUpClass(cls):
        # LLM .pte files use custom/quantized ops; register them (the server's
        # runner_pool does this automatically, but a direct engine test must).
        try:
            import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
            import executorch.kernels.quantized  # noqa: F401
        except Exception:  # noqa: BLE001 - assume statically linked otherwise
            pass
        # The session API takes token ids; tokenize prompts in Python (the server
        # does the same). Load the model's tokenizer.json directly.
        from tokenizers import Tokenizer as HFTokenizer

        cls._hf = HFTokenizer.from_file(_TOKENIZER)

    @classmethod
    def _ids(cls, prompt):
        return cls._hf.encode(prompt).ids

    @staticmethod
    def _gen_text(runner, prompt):  # standalone TextLLMRunner baseline
        out = []
        runner.reset()
        runner.generate(
            prompt,
            GenerationConfig(echo=False, max_new_tokens=12, temperature=0.0),
            lambda t: out.append(t),
        )
        return "".join(out)

    def _session_ids(self, session, prompt_ids, n=12):
        """Drive a session via prefill_tokens + a decode_one loop (the actual
        new path); return the exact generated token ids."""
        session.reset()
        session.prefill_tokens(prompt_ids)
        ids = []
        for _ in range(n):
            step = session.decode_one(0.0)
            ids.append(step["token_id"])
            if step["is_eos"]:
                break
        return ids

    def test_sessions_isolated_and_match_baseline(self):
        p1 = "<|im_start|>user\nName one primary color.<|im_end|>\n<|im_start|>assistant\n"
        p2 = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        base = TextLLMRunner(model_path=_MODEL, tokenizer_path=_TOKENIZER)
        b1, b2 = self._gen_text(base, p1), self._gen_text(base, p2)

        engine = LLMEngine(model_path=_MODEL, tokenizer_path=_TOKENIZER)
        s1, s2 = engine.create_session(), engine.create_session()
        ids1 = self._session_ids(s1, self._ids(p1))
        ids2 = self._session_ids(s2, self._ids(p2))
        ids1b = self._session_ids(s1, self._ids(p1))  # after s2 ran
        # The session's decode_one ids, decoded, match the standalone generation.
        self.assertEqual(self._hf.decode(ids1).strip(), b1.strip())
        self.assertEqual(self._hf.decode(ids2).strip(), b2.strip())
        self.assertEqual(ids1, ids1b, "session1 must be unaffected by session2")

    def test_concurrent_sessions_do_not_crash(self):
        # The original num_runners>1 path crashed (heap corruption) under
        # concurrent backend calls; the engine lock must serialize them safely.
        p = self._ids(
            "<|im_start|>user\nCount to five.<|im_end|>\n<|im_start|>assistant\n"
        )
        engine = LLMEngine(model_path=_MODEL, tokenizer_path=_TOKENIZER)
        s1, s2 = engine.create_session(), engine.create_session()
        expect = self._session_ids(s1, p)
        errors = []

        def worker(sess):
            try:
                for _ in range(3):
                    self.assertEqual(self._session_ids(sess, p), expect)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(s,)) for s in (s1, s2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], "concurrent sessions crashed or drifted")
