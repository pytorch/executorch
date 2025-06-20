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

from executorch.examples.models.llama.config.llm_config import (
    LlmConfig,
    ModelType, 
    PreqMode,
    DtypeOverride,
    Pt2eQuantize,
    SpinQuant,
    CoreMLQuantize,
    CoreMLComputeUnit
)
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
  tokenizer_path: /path/to/tokenizer.json
export:
  max_seq_length: 256
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
            self.assertEqual(called_config["export"]["max_seq_length"], 256)
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

    def test_enum_fields(self) -> None:
        """Test that all enum fields work correctly with their lowercase keys."""
        # Test ModelType enum
        for enum_value in ModelType:
            self.assertIsNotNone(enum_value.value)
            self.assertTrue(isinstance(enum_value.value, str))
        
        # Test specific enum values that were changed from uppercase to lowercase
        self.assertEqual(ModelType.stories110m.value, "stories110m")
        self.assertEqual(ModelType.llama2.value, "llama2")
        self.assertEqual(ModelType.llama3.value, "llama3")
        self.assertEqual(ModelType.llama3_1.value, "llama3_1")
        self.assertEqual(ModelType.llama3_2.value, "llama3_2")
        self.assertEqual(ModelType.llama3_2_vision.value, "llama3_2_vision")
        self.assertEqual(ModelType.static_llama.value, "static_llama")
        self.assertEqual(ModelType.qwen2_5.value, "qwen2_5")
        self.assertEqual(ModelType.qwen3_0_6b.value, "qwen3-0_6b")
        self.assertEqual(ModelType.qwen3_1_7b.value, "qwen3-1_7b")
        self.assertEqual(ModelType.qwen3_4b.value, "qwen3-4b")
        self.assertEqual(ModelType.phi_4_mini.value, "phi_4_mini")
        self.assertEqual(ModelType.smollm2.value, "smollm2")
        
        # Test PreqMode enum
        self.assertEqual(PreqMode.preq_8da4w.value, "8da4w")
        self.assertEqual(PreqMode.preq_8da4w_out_8da8w.value, "8da4w_output_8da8w")
        
        # Test DtypeOverride enum
        self.assertEqual(DtypeOverride.fp32.value, "fp32")
        self.assertEqual(DtypeOverride.fp16.value, "fp16")
        self.assertEqual(DtypeOverride.bf16.value, "bf16")
        
        # Test Pt2eQuantize enum
        self.assertEqual(Pt2eQuantize.xnnpack_dynamic.value, "xnnpack_dynamic")
        self.assertEqual(Pt2eQuantize.xnnpack_dynamic_qc4.value, "xnnpack_dynamic_qc4")
        self.assertEqual(Pt2eQuantize.qnn_8a8w.value, "qnn_8a8w")
        self.assertEqual(Pt2eQuantize.qnn_16a16w.value, "qnn_16a16w")
        self.assertEqual(Pt2eQuantize.qnn_16a4w.value, "qnn_16a4w")
        self.assertEqual(Pt2eQuantize.coreml_c4w.value, "coreml_c4w")
        self.assertEqual(Pt2eQuantize.coreml_8a_c8w.value, "coreml_8a_c8w")
        self.assertEqual(Pt2eQuantize.coreml_8a_c4w.value, "coreml_8a_c4w")
        self.assertEqual(Pt2eQuantize.coreml_baseline_8a_c8w.value, "coreml_baseline_8a_c8w")
        self.assertEqual(Pt2eQuantize.coreml_baseline_8a_c4w.value, "coreml_baseline_8a_c4w")
        self.assertEqual(Pt2eQuantize.vulkan_8w.value, "vulkan_8w")
        
        # Test SpinQuant enum
        self.assertEqual(SpinQuant.cuda.value, "cuda")
        self.assertEqual(SpinQuant.native.value, "native")
        
        # Test CoreMLQuantize enum
        self.assertEqual(CoreMLQuantize.b4w.value, "b4w")
        self.assertEqual(CoreMLQuantize.c4w.value, "c4w")
        
        # Test CoreMLComputeUnit enum
        self.assertEqual(CoreMLComputeUnit.cpu_only.value, "cpu_only")
        self.assertEqual(CoreMLComputeUnit.cpu_and_gpu.value, "cpu_and_gpu")
        self.assertEqual(CoreMLComputeUnit.cpu_and_ne.value, "cpu_and_ne")
        self.assertEqual(CoreMLComputeUnit.all.value, "all")

    def test_enum_configuration(self) -> None:
        """Test that enum fields can be properly set in LlmConfig."""
        config = LlmConfig()
        
        # Test setting ModelType
        config.base.model_class = ModelType.llama3
        self.assertEqual(config.base.model_class.value, "llama3")
        
        # Test setting DtypeOverride  
        config.model.dtype_override = DtypeOverride.fp16
        self.assertEqual(config.model.dtype_override.value, "fp16")
        
        # Test setting PreqMode
        config.base.preq_mode = PreqMode.preq_8da4w
        self.assertEqual(config.base.preq_mode.value, "8da4w")
        
        # Test setting Pt2eQuantize
        config.quantization.pt2e_quantize = Pt2eQuantize.xnnpack_dynamic
        self.assertEqual(config.quantization.pt2e_quantize.value, "xnnpack_dynamic")
        
        # Test setting SpinQuant
        config.quantization.use_spin_quant = SpinQuant.cuda
        self.assertEqual(config.quantization.use_spin_quant.value, "cuda")
        
        # Test setting CoreMLQuantize
        config.backend.coreml.quantize = CoreMLQuantize.c4w
        self.assertEqual(config.backend.coreml.quantize.value, "c4w")
        
        # Test setting CoreMLComputeUnit
        config.backend.coreml.compute_units = CoreMLComputeUnit.cpu_and_gpu
        self.assertEqual(config.backend.coreml.compute_units.value, "cpu_and_gpu")


if __name__ == "__main__":
    unittest.main()

