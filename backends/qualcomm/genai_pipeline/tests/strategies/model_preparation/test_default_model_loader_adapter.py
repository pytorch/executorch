# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.default_model_loader_adapter import (
    DefaultModelLoaderAdapter,
)

TEST_TOKENIZER_CONFIG = "tokenizer_config.json"
TEST_SPECIAL_TOKENS_MAP = "special_tokens_map.json"
TEST_TOKENIZER_JSON = "tokenizer.json"
TEST_TOKENIZER_MODEL = "tokenizer.model"
TEST_ADDED_TOKENS = "added_tokens.json"


class TestExportTokenizer(unittest.TestCase):
    """The runtime (llm::load_tokenizer / pytorch_tokenizers.get_tokenizer)
    expects a single tokenizer *file*, never the containing directory."""

    def setUp(self):
        self.adapter = DefaultModelLoaderAdapter()
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmp.name) / "tokenizer_out"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tokenizer(self, artifacts):
        tokenizer = MagicMock()
        tokenizer.save_pretrained.return_value = artifacts
        return tokenizer

    def test_returns_tokenizer_file_not_the_directory(self):
        artifacts = (
            str(self.output_dir / TEST_TOKENIZER_CONFIG),
            str(self.output_dir / TEST_SPECIAL_TOKENS_MAP),
            str(self.output_dir / TEST_TOKENIZER_JSON),
        )
        result = self.adapter.export_tokenizer(
            self._make_tokenizer(artifacts), self.output_dir
        )
        self.assertEqual(result, self.output_dir / TEST_TOKENIZER_JSON)
        self.assertNotEqual(result, self.output_dir)

    def test_creates_output_directory(self):
        artifacts = (str(self.output_dir / TEST_TOKENIZER_JSON),)
        self.adapter.export_tokenizer(self._make_tokenizer(artifacts), self.output_dir)
        self.assertTrue(self.output_dir.is_dir())

    def test_raises_when_no_artifacts_written(self):
        for artifacts in (None, ()):
            with self.subTest(artifacts=artifacts):
                with self.assertRaises(FileNotFoundError):
                    self.adapter.export_tokenizer(
                        self._make_tokenizer(artifacts), self.output_dir
                    )

    def test_selects_by_name_not_by_position(self):
        # save_pretrained's ordering is an implementation detail: a fast
        # tokenizer may append added_tokens.json after tokenizer.json. Selecting
        # positionally would hand the runtime the wrong file, and because
        # get_tokenizer dispatches on the extension it would construct the wrong
        # tokenizer class rather than fail.
        artifacts = (
            str(self.output_dir / TEST_TOKENIZER_CONFIG),
            str(self.output_dir / TEST_TOKENIZER_JSON),
            str(self.output_dir / TEST_ADDED_TOKENS),
        )
        result = self.adapter.export_tokenizer(
            self._make_tokenizer(artifacts), self.output_dir
        )
        self.assertEqual(result, self.output_dir / TEST_TOKENIZER_JSON)

    def test_prefers_tokenizer_json_over_tokenizer_model(self):
        artifacts = (
            str(self.output_dir / TEST_TOKENIZER_MODEL),
            str(self.output_dir / TEST_TOKENIZER_JSON),
        )
        result = self.adapter.export_tokenizer(
            self._make_tokenizer(artifacts), self.output_dir
        )
        self.assertEqual(result, self.output_dir / TEST_TOKENIZER_JSON)

    def test_falls_back_to_tokenizer_model_when_no_json(self):
        artifacts = (
            str(self.output_dir / TEST_TOKENIZER_CONFIG),
            str(self.output_dir / TEST_TOKENIZER_MODEL),
            str(self.output_dir / TEST_SPECIAL_TOKENS_MAP),
        )
        result = self.adapter.export_tokenizer(
            self._make_tokenizer(artifacts), self.output_dir
        )
        self.assertEqual(result, self.output_dir / TEST_TOKENIZER_MODEL)

    def test_falls_back_to_last_artifact_when_no_known_name(self):
        # e.g. stories110m ships tokenizer.bin; keep save_pretrained's ordering
        # as a last resort rather than failing outright.
        artifacts = (
            str(self.output_dir / TEST_TOKENIZER_CONFIG),
            str(self.output_dir / "tokenizer.bin"),
        )
        result = self.adapter.export_tokenizer(
            self._make_tokenizer(artifacts), self.output_dir
        )
        self.assertEqual(result, self.output_dir / "tokenizer.bin")


class TestGetExampleInputs(unittest.TestCase):
    """Export inputs describe the *model's* signature, so a model that already
    knows its own signature must win over anything synthesized here."""

    def setUp(self):
        self.adapter = DefaultModelLoaderAdapter()

    def test_prefers_example_inputs_provided_by_the_model(self):
        expected = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))
        model = MagicMock()
        model.get_example_inputs.return_value = expected

        result = self.adapter.get_example_inputs(model)

        self.assertEqual(result, expected)
        model.get_example_inputs.assert_called_once_with()

    def test_synthesizes_int64_token_ids_when_model_provides_none(self):
        import torch

        # `spec=[]` gives an object with no attributes, so the adapter cannot
        # find a `get_example_inputs` to defer to.
        model = MagicMock(spec=[])

        result = self.adapter.get_example_inputs(
            model, extra_options={"batch_size": 2, "ar_len": 8}
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (2, 8))
        # Token ids index an embedding table, so they must be integral.
        self.assertEqual(result[0].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
