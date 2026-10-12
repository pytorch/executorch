# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
import warnings
from dataclasses import asdict, replace

from executorch.extension.llm.export.config.llm_config import (
    BackendConfig,
    BaseConfig,
    CoreMLComputeUnit,
    CoreMLConfig,
    DebugConfig,
    ExportConfig,
    LlmConfig,
    ModelConfig,
    ModelType,
    QuantizationConfig,
    XNNPackConfig,
)
from executorch.extension.llm.export.config.model_identity import (
    LEGACY_MODEL_CLASS_TO_FAMILY,
    LEGACY_MODEL_CLASS_TO_ID,
    MODEL_SPECS,
    ModelFamily,
    ModelIdentity,
    resolve_model_identity,
)

from omegaconf import OmegaConf


class TestValidation(unittest.TestCase):
    def test_invalid_attention_sink(self):
        with self.assertRaises(ValueError):
            ModelConfig(use_attention_sink="4")
        with self.assertRaises(ValueError):
            ModelConfig(use_attention_sink="4,2048,1024")

    def test_invalid_local_global_attention_format(self):
        with self.assertRaises(ValueError):
            ModelConfig(local_global_attention="notalist")

    def test_quantize_kv_without_kv(self):
        with self.assertRaises(ValueError):
            ModelConfig(quantize_kv_cache=True)

    def test_static_quantize_kv_without_kv(self):
        with self.assertRaises(ValueError):
            ModelConfig(static_quantize_kv_cache=True)

    def test_static_quantize_kv_rejects_dynamic_shape(self):
        with self.assertRaises(ValueError):
            ModelConfig(use_kv_cache=True, static_quantize_kv_cache=True)

    def test_static_quantize_kv_rejects_dynamic_quantized_kv(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                use_kv_cache=True,
                enable_dynamic_shape=False,
                quantize_kv_cache=True,
                static_quantize_kv_cache=True,
            )

    def test_static_quantize_kv_rejects_specialized_cache_modes(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                use_kv_cache=True,
                enable_dynamic_shape=False,
                static_quantize_kv_cache=True,
                local_global_attention=[16],
            )
        with self.assertRaises(ValueError):
            ModelConfig(
                use_kv_cache=True,
                enable_dynamic_shape=False,
                static_quantize_kv_cache=True,
                use_attention_sink="4,2048",
            )

    def test_static_quantize_kv_rejects_non_finite_scale(self):
        for scale in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ModelConfig(static_quantize_kv_cache_scale=scale)

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

    def test_model_identity_fields_are_optional(self):
        config = BaseConfig()

        self.assertIsNone(config.model_class)
        self.assertIsNone(config.model_family)
        self.assertIsNone(config.model_id)
        identity = LlmConfig(base=config).model_identity
        self.assertIs(identity.family, ModelFamily.llama3)
        self.assertIsNone(identity.model_id)
        self.assertEqual(resolve_model_identity(config), identity)

    def test_model_id_infers_family(self):
        config = BaseConfig(model_id="qwen2_5_0_5b")

        self.assertIsNone(config.model_family)
        identity = LlmConfig(base=config).model_identity
        self.assertIs(identity.family, ModelFamily.qwen2_5)
        self.assertEqual(identity.model_id, "qwen2_5_0_5b")

    def test_identity_resolution_uses_current_config_fields(self):
        config = LlmConfig(base=BaseConfig(model_id="qwen2_5_0_5b"))
        previous = config.model_identity
        config.base.model_id = "qwen3_0_6b"

        self.assertIsNone(config.base.model_family)
        self.assertEqual(
            config.model_identity, ModelIdentity(ModelFamily.qwen3, "qwen3_0_6b")
        )
        self.assertIs(previous.family, ModelFamily.qwen2_5)
        config.base = replace(config.base, checkpoint="/tmp/local.pth")
        self.assertIs(config.model_identity.family, ModelFamily.qwen3)

        config.base.model_family = ModelFamily.qwen3
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _ = config.model_identity
        config.base = BaseConfig(model_family=ModelFamily.llama3_2)
        self.assertEqual(
            config.model_identity, ModelIdentity(ModelFamily.llama3_2, None)
        )

    def test_model_identity_is_not_serialized_with_selectors(self):
        config = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(LlmConfig), {"base": {"model_id": "qwen3_0_6b"}}
            )
        )
        self.assertIsNone(config.base.model_family)
        self.assertNotIn("model_identity", asdict(config))
        restored = OmegaConf.to_object(OmegaConf.structured(config))
        self.assertIsNone(restored.base.model_family)
        self.assertEqual(
            restored.model_identity, ModelIdentity(ModelFamily.qwen3, "qwen3_0_6b")
        )

    def test_spark_model_id_infers_family(self):
        config = BaseConfig(model_id="spark_x2_5_1_7b")

        self.assertEqual(
            LlmConfig(base=config).model_identity,
            ModelIdentity(ModelFamily.spark_x2_5, "spark_x2_5_1_7b"),
        )

    def test_model_family_is_used_without_model_id(self):
        config = BaseConfig(model_family="llama3_2")

        self.assertIsInstance(config.model_family, ModelFamily)
        self.assertEqual(config.model_family, ModelFamily.llama3_2)
        self.assertEqual(
            LlmConfig(base=config).model_identity,
            ModelIdentity(ModelFamily.llama3_2, None),
        )

    def test_model_class_remains_backward_compatible_with_warning(self):
        with self.assertWarnsRegex(
            FutureWarning,
            r"base\.model_class is deprecated and will be removed "
            r"in a future release\. Use base\.model_id, or "
            r"base\.model_family with base\.checkpoint for local models\.",
        ):
            config = BaseConfig(model_class="qwen3_1_7b")

        self.assertEqual(
            LlmConfig(base=config).model_identity,
            ModelIdentity(ModelFamily.qwen3, "qwen3_1_7b"),
        )

    def test_all_legacy_model_types_resolve_to_family_and_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for model_type in ModelType:
                with self.subTest(model_type=model_type):
                    config = BaseConfig(model_class=model_type)
                    expected_family = LEGACY_MODEL_CLASS_TO_FAMILY[model_type.value]
                    expected_model_id = LEGACY_MODEL_CLASS_TO_ID.get(model_type.value)

                    identity = LlmConfig(base=config).model_identity
                    self.assertIs(identity.family, expected_family)
                    self.assertEqual(identity.model_id, expected_model_id)

    def test_unrecognized_model_class_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown legacy model_class"):
            BaseConfig(model_class="totally_unknown_model")

    def test_model_class_cannot_be_combined_with_new_model_fields(self):
        for new_fields in (
            {"model_family": "llama3_2"},
            {"model_id": "qwen3_1_7b"},
        ):
            with self.subTest(new_fields=new_fields), self.assertRaisesRegex(
                ValueError, "base.model_class cannot be used"
            ):
                BaseConfig(model_class="qwen3_1_7b", **new_fields)

    def test_model_family_cannot_be_combined_with_model_id(self):
        for model_family in ("qwen3", "qwen2_5"):
            with self.subTest(model_family=model_family), self.assertRaisesRegex(
                ValueError, "mutually exclusive"
            ):
                BaseConfig(model_family=model_family, model_id="qwen3_0_6b")

    def test_model_id_must_be_registered_with_an_hf_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "Unknown model_id"):
            BaseConfig(model_id="future_model")
        with self.assertRaisesRegex(ValueError, "Unknown model_id"):
            BaseConfig(
                model_id="future_model",
                checkpoint="/tmp/custom.pth",
            )

    def test_all_model_specs_have_hf_checkpoint_sources(self):
        self.assertTrue(all(spec.hf_repo_id for spec in MODEL_SPECS.values()))

    def test_valid_llm_config(self):
        with self.assertWarnsRegex(
            FutureWarning,
            r"base\.model_class is deprecated and will be removed "
            r"in a future release\. Use base\.model_id, or "
            r"base\.model_family with base\.checkpoint for local models\.",
        ):
            base = BaseConfig(
                model_class="llama3",
                checkpoint="checkpoints/model.pt",
                tokenizer_path="tokenizer.json",
                use_lora=8,
            )

        LlmConfig(
            base=base,
            model=ModelConfig(
                dtype_override="fp32",
                use_attention_sink="4,2048",
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
