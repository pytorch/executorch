# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from executorch.extension.llm.export.export_llm import (
    main,
    parse_config_arg,
    pop_config_arg,
)


class TestExportLlm(unittest.TestCase):
    def test_parse_config_arg_with_config(self) -> None:
        """Test parse_config_arg when --config is provided."""
        # Mock sys.argv to include --config
        test_argv = ["export_llm.py", "--config", "test_config.yaml", "extra", "args"]
        with patch.object(sys, "argv", test_argv):
            config_path, remaining = parse_config_arg()
            self.assertEqual(config_path, "test_config.yaml")
            self.assertEqual(remaining, ["extra", "args"])

    def test_parse_config_arg_without_config(self) -> None:
        """Test parse_config_arg when --config is not provided."""
        test_argv = ["export_llm.py", "debug.verbose=True"]
        with patch.object(sys, "argv", test_argv):
            config_path, remaining = parse_config_arg()
            self.assertIsNone(config_path)
            self.assertEqual(remaining, ["debug.verbose=True"])

    def test_pop_config_arg(self) -> None:
        """Test pop_config_arg removes --config and its value from sys.argv."""
        test_argv = ["export_llm.py", "--config", "test_config.yaml", "other", "args"]
        with patch.object(sys, "argv", test_argv):
            config_path = pop_config_arg()
            self.assertEqual(config_path, "test_config.yaml")
            self.assertEqual(sys.argv, ["export_llm.py", "other", "args"])

    def test_with_cli_args(self) -> None:
        """Test main function with only hydra CLI args."""
        test_argv = ["export_llm.py", "debug.verbose=True"]
        with patch.object(sys, "argv", test_argv):
            with patch(
                "executorch.extension.llm.export.export_llm.hydra_main"
            ) as mock_hydra:
                main()
                mock_hydra.assert_called_once()

    @patch("executorch.extension.llm.export.export_llm.export_llama")
    def test_with_config(self, mock_export_llama: MagicMock) -> None:
        """Test main function with --config file and no hydra args."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
base:
  model_class: llama2
  tokenizer_path: /path/to/tokenizer.json
  preq_mode: preq_8da4w
model:
  dtype_override: fp16
export:
  max_seq_length: 128
quantization:
  pt2e_quantize: xnnpack_dynamic
  use_spin_quant: cuda
backend:
  coreml:
    quantize: c4w
    compute_units: cpu_and_gpu
"""
            )
            config_file = f.name

        try:
            test_argv = ["export_llm.py", "--config", config_file]
            with patch.object(sys, "argv", test_argv):
                main()

            # Verify export_llama was called with config
            mock_export_llama.assert_called_once()
            called_config = mock_export_llama.call_args[0][0]
            self.assertEqual(
                called_config.base.tokenizer_path, "/path/to/tokenizer.json"
            )
            self.assertEqual(called_config.base.model_class, "llama2")
            self.assertEqual(called_config.base.preq_mode.value, "8da4w")
            self.assertEqual(called_config.model.dtype_override.value, "fp16")
            self.assertEqual(called_config.export.max_seq_length, 128)
            self.assertEqual(
                called_config.quantization.pt2e_quantize.value, "xnnpack_dynamic"
            )
            self.assertEqual(called_config.quantization.use_spin_quant.value, "cuda")
            self.assertEqual(called_config.backend.coreml.quantize.value, "c4w")
            self.assertEqual(
                called_config.backend.coreml.compute_units.value, "cpu_and_gpu"
            )
        finally:
            os.unlink(config_file)

    @patch("executorch.extension.llm.export.export_llm.export_llama")
    def test_with_config_and_cli(self, mock_export_llama: MagicMock) -> None:
        """Test main function with --config file and no hydra args."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
base:
  model_class: llama2
model:
  dtype_override: fp16
backend:
  xnnpack:
    enabled: False
"""
            )
            config_file = f.name

        try:
            test_argv = [
                "export_llm.py",
                "--config",
                config_file,
                "base.model_class=stories110m",
                "backend.xnnpack.enabled=True",
            ]
            with patch.object(sys, "argv", test_argv):
                main()

            # Verify export_llama was called with config
            mock_export_llama.assert_called_once()
            called_config = mock_export_llama.call_args[0][0]
            self.assertEqual(
                called_config.base.model_class, "stories110m"
            )  # Override from CLI.
            self.assertEqual(
                called_config.model.dtype_override.value, "fp16"
            )  # From yaml.
            self.assertEqual(
                called_config.backend.xnnpack.enabled,
                True,  # Override from CLI.
            )
        finally:
            os.unlink(config_file)


if __name__ == "__main__":
    unittest.main()
