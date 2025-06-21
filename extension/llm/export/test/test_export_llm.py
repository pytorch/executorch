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

from executorch.examples.models.llama.config.llm_config import LlmConfig
from executorch.extension.llm.export.export_llm import main, parse_config_arg, pop_config_arg


class TestExportLlm(unittest.TestCase):
    def test_parse_config_arg_with_config(self) -> None:
        """Test parse_config_arg when --config is provided."""
        # Mock sys.argv to include --config
        test_argv = ["script.py", "--config", "test_config.yaml", "extra", "args"]
        with patch.object(sys, "argv", test_argv):
            config_path, remaining = parse_config_arg()
            self.assertEqual(config_path, "test_config.yaml")
            self.assertEqual(remaining, ["extra", "args"])

    def test_parse_config_arg_without_config(self) -> None:
        """Test parse_config_arg when --config is not provided."""
        test_argv = ["script.py", "debug.verbose=True"]
        with patch.object(sys, "argv", test_argv):
            config_path, remaining = parse_config_arg()
            self.assertIsNone(config_path)
            self.assertEqual(remaining, ["debug.verbose=True"])

    def test_pop_config_arg(self) -> None:
        """Test pop_config_arg removes --config and its value from sys.argv."""
        test_argv = ["script.py", "--config", "test_config.yaml", "other", "args"]
        with patch.object(sys, "argv", test_argv):
            config_path = pop_config_arg()
            self.assertEqual(config_path, "test_config.yaml")
            self.assertEqual(sys.argv, ["script.py", "other", "args"])

    @patch("executorch.extension.llm.export.export_llm.export_llama")
    def test_with_config(self, mock_export_llama: MagicMock) -> None:
        """Test main function with --config file and no hydra args."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
base:
  model_class: llama2
  tokenizer_path: /path/to/tokenizer.json
  preq_mode: preq_8da4w
model:
  dtype_override: fp16
export:
  max_seq_length: 256
quantization:
  pt2e_quantize: xnnpack_dynamic
  use_spin_quant: cuda
backend:
  coreml:
    quantize: c4w
    compute_units: cpu_and_gpu
""")
            config_file = f.name

        try:
            test_argv = ["script.py", "--config", config_file]
            with patch.object(sys, "argv", test_argv):
                main()

            # Verify export_llama was called with config
            mock_export_llama.assert_called_once()
            called_config = mock_export_llama.call_args[0][0]
            self.assertEqual(called_config["base"]["tokenizer_path"], "/path/to/tokenizer.json")
            self.assertEqual(called_config["base"]["model_class"], "llama2")
            self.assertEqual(called_config["base"]["preq_mode"].value, "8da4w")
            self.assertEqual(called_config["model"]["dtype_override"].value, "fp16")
            self.assertEqual(called_config["export"]["max_seq_length"], 256)
            self.assertEqual(called_config["quantization"]["pt2e_quantize"].value, "xnnpack_dynamic")
            self.assertEqual(called_config["quantization"]["use_spin_quant"].value, "cuda")
            self.assertEqual(called_config["backend"]["coreml"]["quantize"].value, "c4w")
            self.assertEqual(called_config["backend"]["coreml"]["compute_units"].value, "cpu_and_gpu")
        finally:
            os.unlink(config_file)

    def test_with_cli_args(self) -> None:
        """Test main function with only hydra CLI args."""
        test_argv = ["script.py", "debug.verbose=True"]
        with patch.object(sys, "argv", test_argv):
            with patch("executorch.extension.llm.export.export_llm.hydra_main") as mock_hydra:
                main()
                mock_hydra.assert_called_once()

    def test_config_with_cli_args_error(self) -> None:
        """Test that --config rejects additional CLI arguments to prevent mixing approaches."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("base:\n  checkpoint: /path/to/checkpoint.pth")
            config_file = f.name

        try:
            test_argv = ["script.py", "--config", config_file, "debug.verbose=True"]
            with patch.object(sys, "argv", test_argv):
                with self.assertRaises(ValueError) as cm:
                    main()
                
                error_msg = str(cm.exception)
                self.assertIn("Cannot specify additional CLI arguments when using --config", error_msg)
        finally:
            os.unlink(config_file)

    def test_config_rejects_multiple_cli_args(self) -> None:
        """Test that --config rejects multiple CLI arguments (not just single ones)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("export:\n  max_seq_length: 128")
            config_file = f.name

        try:
            test_argv = ["script.py", "--config", config_file, "debug.verbose=True", "export.output_dir=/tmp"]
            with patch.object(sys, "argv", test_argv):
                with self.assertRaises(ValueError):
                    main()
        finally:
            os.unlink(config_file)


if __name__ == "__main__":
    unittest.main()

