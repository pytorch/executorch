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


if __name__ == "__main__":
    unittest.main()
