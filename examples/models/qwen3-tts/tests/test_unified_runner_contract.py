from pathlib import Path
import unittest


class UnifiedRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.header = (root / "qwen3_tts_unified_runner.h").read_text(
            encoding="utf-8"
        )
        cls.runner = (root / "qwen3_tts_unified_runner.cpp").read_text(
            encoding="utf-8"
        )
        cls.main = (root / "main_unified.cpp").read_text(encoding="utf-8")

    def test_runner_header_exposes_top_p_sampling_config(self):
        self.assertIn("float top_p = -1.0f;", self.header)
        self.assertIn("float top_p,", self.header)
        self.assertIn("uint64_t seed = 0;", self.header)
        self.assertIn("struct SynthesisTiming", self.header)
        self.assertIn("class SynthesisSession;", self.header)
        self.assertIn("create_synthesis_session", self.header)

    def test_main_cli_validates_text_mode_requirements(self):
        self.assertIn('DEFINE_double(top_p, -1.0, "Top-p sampling.', self.main)
        self.assertIn('Provide either --codes_path or text synthesis inputs, not both.', self.main)
        self.assertIn('Provide either --text or --prompts_path, not both.', self.main)
        self.assertIn('Text synthesis requires --tokenizer_path.', self.main)
        self.assertIn('DEFINE_string(\n    prompts_path,', self.main)
        self.assertIn('DEFINE_int32(repeat, 1, "Repeat count', self.main)
        self.assertIn('DEFINE_uint64(seed, 42, "Base RNG seed', self.main)
        self.assertIn("disable_fused_cp_generate", self.main)
        self.assertIn("Benchmark mode defaulting top_k to %d", self.main)
        self.assertIn("create_synthesis_session", self.main)

    def test_runner_uses_assistant_wrapped_prompt_contract(self):
        self.assertIn("build_assistant_prompt_text", self.runner)
        self.assertIn("text_prompt_min_token_count_", self.runner)
        self.assertIn("text_prompt_prefill_token_count_", self.runner)
        self.assertIn("text_prompt_trailing_template_token_count_", self.runner)
        self.assertIn(
            "Tokenized assistant prompt: %d tokens",
            self.runner,
        )

    def test_runner_matches_generate_codes_english_language_prefix(self):
        self.assertIn("int64_t codec_think_id_ = 2154;", self.header)
        self.assertIn("int64_t codec_language_english_id_ = 2050;", self.header)
        self.assertIn('language_lower == "english"', self.runner)
        self.assertIn("text_prompt_prefill_token_count_with_language_", self.runner)

    def test_runner_warmup_and_fast_path_cover_full_text_pipeline(self):
        self.assertIn('ET_LOG(Info, "Warming up full text synthesis path...', self.runner)
        self.assertIn('ensure_method("cp_generate")', self.runner)
        self.assertIn("run_cp_generate(", self.runner)
        self.assertIn("use_fused_cp_generate", self.runner)


if __name__ == "__main__":
    unittest.main()
