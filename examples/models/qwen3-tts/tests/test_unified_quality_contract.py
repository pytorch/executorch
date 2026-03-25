from pathlib import Path
import unittest


class UnifiedQualityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.export_source = (root / "export_unified.py").read_text(encoding="utf-8")
        cls.model_source = (root / "model.py").read_text(encoding="utf-8")
        cls.header = (root / "qwen3_tts_unified_runner.h").read_text(
            encoding="utf-8"
        )
        cls.runner = (root / "qwen3_tts_unified_runner.cpp").read_text(
            encoding="utf-8"
        )
        cls.main = (root / "main_unified.cpp").read_text(encoding="utf-8")

    def test_runner_uses_real_codec_control_token_ids(self):
        self.assertIn("int64_t codec_pad_id_ = 2148;", self.header)
        self.assertIn("int64_t codec_bos_id_ = 2149;", self.header)
        self.assertIn("int64_t codec_eos_id_ = 2150;", self.header)
        self.assertIn("int64_t codec_nothink_id_ = 2155;", self.header)
        self.assertIn("int64_t codec_think_bos_id_ = 2156;", self.header)
        self.assertIn("int64_t codec_think_eos_id_ = 2157;", self.header)

    def test_export_does_not_hardcode_wrong_codec_token_band(self):
        for token_id in ("4196", "4197", "4198", "4203", "4204", "4205"):
            self.assertNotIn(token_id, self.export_source)

    def test_runner_suppresses_talker_special_token_band(self):
        self.assertIn("talker_vocab_size_ - 1024", self.runner)
        self.assertIn("suppress_tokens", self.runner)

    def test_runner_does_not_silently_clamp_invalid_codes_to_zero(self):
        self.assertNotIn("code = 0;", self.runner)

    def test_runner_has_last_token_extraction_helper(self):
        self.assertIn("extract_last_token_slice", self.runner)

    def test_runner_does_not_slice_last_token_twice_after_export(self):
        self.assertNotIn(
            "extract_last_token_slice(full_logits, seq_len, talker_vocab_size_, logits);",
            self.runner,
        )
        self.assertNotIn(
            "extract_last_token_slice(full_hidden, seq_len, talker_dim_, hidden);",
            self.runner,
        )

    def test_sampler_deduplicates_tokens_before_repetition_penalty(self):
        self.assertIn("std::sort(unique_tokens.begin(), unique_tokens.end());", self.runner)
        self.assertIn("unique_tokens.erase(", self.runner)
        self.assertIn(
            "std::unique(unique_tokens.begin(), unique_tokens.end())",
            self.runner,
        )

    def test_sampler_preserves_eos_logit_across_filtering(self):
        self.assertIn("int64_t eos_token_id", self.header)
        self.assertIn("const float preserved_eos_logit", self.runner)
        self.assertIn(
            "adjusted[static_cast<size_t>(eos_token_id)] = preserved_eos_logit;",
            self.runner,
        )

    def test_repetition_penalty_is_exposed_for_text_mode(self):
        self.assertIn("float repetition_penalty = 1.05f;", self.header)
        self.assertIn("DEFINE_double(repetition_penalty, 1.05", self.main)
        self.assertIn(
            "config.repetition_penalty = static_cast<float>(FLAGS_repetition_penalty);",
            self.main,
        )

    def test_cp_generate_export_uses_sampling_aware_contract(self):
        self.assertIn("sample_uniforms: torch.Tensor", self.export_source)
        self.assertIn("torch.topk(logits, k=50", self.export_source)
        self.assertIn("torch.cumsum(probs, dim=0)", self.export_source)
        self.assertIn("torch.stack(sampled_codes, dim=0), embed_sum", self.export_source)

    def test_runner_uses_session_rng_instead_of_static_global_rng(self):
        self.assertIn("std::mt19937* gen", self.header)
        self.assertIn("config.seed == 0 ? std::random_device{}() : config.seed", self.runner)
        self.assertNotIn("static std::mt19937 gen(42);", self.runner)

    def test_runner_has_fused_cp_generate_fast_path_and_legacy_fallback(self):
        self.assertIn("cp_generate_contract_version_ >= 2", self.runner)
        self.assertIn("config_.top_k == runner->cp_generate_fast_top_k_", self.runner)
        self.assertIn("config_.temperature >= 1e-6f", self.runner)
        self.assertIn("use_fused_cp_generate", self.runner)
        self.assertIn("Falling back to legacy code predictor loop", self.runner)
        self.assertIn("sample_uniforms", self.runner)

    def test_decoder_wrapper_shims_missing_transformers_check_model_inputs(self):
        self.assertIn('hasattr(hf_generic, "check_model_inputs")', self.model_source)
        self.assertIn("hf_generic.check_model_inputs = _identity_check_model_inputs", self.model_source)

    def test_decoder_wrapper_shims_missing_default_rope_initializer(self):
        self.assertIn('if "default" not in hf_rope_utils.ROPE_INIT_FUNCTIONS:', self.model_source)
        self.assertIn('hf_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters', self.model_source)


if __name__ == "__main__":
    unittest.main()
