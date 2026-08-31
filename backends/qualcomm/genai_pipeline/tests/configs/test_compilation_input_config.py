# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from executorch.backends.qualcomm.genai_pipeline.configs.compilation_input_config import (
    CompilationInputConfig,
)


class TestCompilationInputConfig(unittest.TestCase):

    def test_artifact_dir_default(self):
        config = CompilationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        self.assertEqual(config.artifact_dir, Path("."))

    def test_required_fields(self):
        with self.assertRaises(TypeError):
            CompilationInputConfig()

    def test_optional_fields_default_to_none(self):
        config = CompilationInputConfig(soc_model=MagicMock(), backend_type=MagicMock())
        self.assertIsNone(config.model)
        self.assertIsNone(config.example_inputs)
        self.assertIsNone(config.compile_specs)

    def test_example_inputs_carries_export_signature(self):
        example_inputs = (MagicMock(name="tokens"), MagicMock(name="attn_mask"))
        config = CompilationInputConfig(
            soc_model=MagicMock(),
            backend_type=MagicMock(),
            example_inputs=example_inputs,
        )
        self.assertIs(config.example_inputs, example_inputs)


if __name__ == "__main__":
    unittest.main()
