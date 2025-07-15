# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

from executorch.extension.llm.export.config.llm_config import (
    BackendConfig,
    BaseConfig,
    CoreMLComputeUnit,
    CoreMLConfig,
    DebugConfig,
    ExportConfig,
    LlmConfig,
    ModelConfig,
    QuantizationConfig,
    XNNPackConfig,
)


class TestValidation(unittest.TestCase):
    def test_invalid_attention_sink(self):
        with self.assertRaises(ValueError):
            ModelConfig(use_attention_sink="4,2048")

    def test_invalid_local_global_attention_format(self):
        with self.assertRaises(ValueError):
            ModelConfig(local_global_attention="notalist")

    def test_quantize_kv_without_kv(self):
        with self.assertRaises(ValueError):
            ModelConfig(quantize_kv_cache=True)

    def test_local_global_attention_without_kv(self):
        with self.assertRaises(ValueError):
            ModelConfig(local_global_attention="[16]", use_kv_cache=False)

    def test_invalid_export_config_context_length(self):
        with self.assertRaises(ValueError):
            ExportConfig(max_seq_length=256, max_context_length=128)

    def test_invalid_qmode(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(qmode="unknown")

    def test_invalid_coreml_ios(self):
        with self.assertRaises(ValueError):
            CoreMLConfig(ios=14)

    def test_lowbit_conflict_with_xnnpack(self):
        qcfg = QuantizationConfig(qmode="torchao:8da4w")
        bcfg = BackendConfig(xnnpack=XNNPackConfig(enabled=True))
        model_cfg = ModelConfig(use_shared_embedding=True)

        with self.assertRaises(ValueError):
            LlmConfig(model=model_cfg, quantization=qcfg, backend=bcfg)

    def test_shared_embedding_without_lowbit(self):
        model_cfg = ModelConfig(use_shared_embedding=True)
        qcfg = QuantizationConfig(qmode="int8")

        with self.assertRaises(ValueError):
            LlmConfig(model=model_cfg, quantization=qcfg)


class TestValidConstruction(unittest.TestCase):

    def test_valid_llm_config(self):
        LlmConfig(
            base=BaseConfig(
                model_class="llama3",
                checkpoint="checkpoints/model.pt",
                tokenizer_path="tokenizer.json",
                use_lora=8,
            ),
            model=ModelConfig(
                dtype_override="fp32",
                use_attention_sink="4,2048,1024",
                use_kv_cache=True,
                local_global_attention="[16, 32]",
            ),
            export=ExportConfig(
                max_seq_length=128,
                max_context_length=256,
                output_dir="/tmp/export",
                output_name="model.pte",
            ),
            debug=DebugConfig(profile_memory=True, verbose=True),
            quantization=QuantizationConfig(qmode="torchao:8da4w"),
            backend=BackendConfig(
                xnnpack=XNNPackConfig(enabled=False),
                coreml=CoreMLConfig(
                    enabled=True, ios=17, compute_units=CoreMLComputeUnit.cpu_only
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
