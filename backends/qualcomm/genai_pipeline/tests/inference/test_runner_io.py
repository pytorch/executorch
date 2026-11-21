# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteRequest,
    ForwardRequest,
    GenerateOutput,
    GenerateRequest,
)


class TestExecuteRequest(unittest.TestCase):
    def test_defaults_run_method_zero_once(self):
        """Method index and iteration default to the first method, run once."""
        request = ExecuteRequest(inputs=((1,),))
        self.assertEqual(request.method_index, 0)
        self.assertEqual(request.iteration, 1)

    def test_rejects_invalid_values(self):
        """No samples, a negative method index, or zero iterations raise ValueError."""
        for kwargs in (
            {"inputs": ()},
            {"inputs": ((1,),), "method_index": -1},
            {"inputs": ((1,),), "iteration": 0},
        ):
            with self.subTest(**{key: str(value) for key, value in kwargs.items()}):
                with self.assertRaises(ValueError):
                    ExecuteRequest(**kwargs)


class TestGenerateRequest(unittest.TestCase):
    def test_rejects_empty_prompts(self):
        """A request without prompts raises ValueError."""
        with self.assertRaisesRegex(ValueError, "at least one prompt"):
            GenerateRequest(prompts=[])

    def test_is_frozen(self):
        """A request cannot be edited once planned."""
        request = GenerateRequest(prompts=["p"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.prompts = ["q"]


class TestForwardRequest(unittest.TestCase):
    def test_rejects_empty_tokens(self):
        """A request without tokens raises ValueError."""
        with self.assertRaisesRegex(ValueError, "at least one token"):
            ForwardRequest(token_ids=[])

    def test_rejects_seq_len_not_exceeding_tokens(self):
        """A decode bound no longer than the tokens raises ValueError."""
        with self.assertRaisesRegex(ValueError, "must exceed the token count"):
            ForwardRequest(token_ids=[1, 2], seq_len=2)

    def test_accepts_seq_len_past_tokens(self):
        """A decode bound past the tokens is kept."""
        self.assertEqual(ForwardRequest(token_ids=[1, 2], seq_len=3).seq_len, 3)


class TestGenerateOutput(unittest.TestCase):
    def test_files_default_to_empty(self):
        """An output without collected files has an empty, unshared file list."""
        first = GenerateOutput(texts=["a"], tokens_per_sec=1.0)
        second = GenerateOutput(texts=["b"], tokens_per_sec=1.0)
        self.assertEqual(first.files, [])
        self.assertIsNot(first.files, second.files)


if __name__ == "__main__":
    unittest.main()
