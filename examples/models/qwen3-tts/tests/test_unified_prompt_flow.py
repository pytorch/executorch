import importlib.util
from pathlib import Path
import unittest

import torch


def _load_prompt_contract_module():
    script_path = Path(__file__).resolve().parents[1] / "text_prompt_contract.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_tts_text_prompt_contract", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UnifiedPromptFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_prompt_contract_module()

    def test_build_assistant_prompt_text_matches_expected_template(self):
        text = "Hello from ExecuTorch."
        prompt = self.mod.build_assistant_prompt_text(text)
        self.assertEqual(
            prompt,
            "<|im_start|>assistant\n"
            "Hello from ExecuTorch."
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_split_prompt_embeddings_uses_first_text_token_in_prefill(self):
        prompt_embeds = torch.arange(12.0, dtype=torch.float32).reshape(1, 12, 1)
        tts_eos_embed = torch.tensor([[[999.0]]], dtype=torch.float32)

        parts = self.mod.split_prompt_embeddings(prompt_embeds, tts_eos_embed)

        self.assertTrue(torch.equal(parts.role_embed, prompt_embeds[:, :3, :]))
        self.assertTrue(torch.equal(parts.first_text_embed, prompt_embeds[:, 3:4, :]))
        self.assertTrue(
            torch.equal(
                parts.trailing_text_hidden,
                torch.tensor([[[4.0], [5.0], [6.0], [999.0]]], dtype=torch.float32),
            )
        )

    def test_split_prompt_embeddings_rejects_too_short_prompt(self):
        prompt_embeds = torch.zeros(1, 8, 4, dtype=torch.float32)
        tts_eos_embed = torch.zeros(1, 1, 4, dtype=torch.float32)

        with self.assertRaises(ValueError):
            self.mod.split_prompt_embeddings(prompt_embeds, tts_eos_embed)

    def test_build_text_only_runtime_plan_reports_prefill_and_trailing_lengths(self):
        plan = self.mod.build_text_only_runtime_plan(
            prompt_token_count=12,
            max_seq_len=64,
            max_new_tokens=16,
        )

        self.assertEqual(plan.prefill_token_count, 8)
        self.assertEqual(plan.trailing_token_count, 4)
        self.assertEqual(plan.min_required_generation_steps, 4)

    def test_build_text_only_runtime_plan_supports_language_prefix_budget(self):
        plan = self.mod.build_text_only_runtime_plan(
            prompt_token_count=12,
            max_seq_len=64,
            max_new_tokens=16,
            use_language_prefix=True,
        )

        self.assertEqual(plan.prefill_token_count, 9)
        self.assertEqual(plan.trailing_token_count, 4)
        self.assertEqual(plan.min_required_generation_steps, 4)

    def test_build_text_only_runtime_plan_rejects_insufficient_generation_budget(self):
        with self.assertRaises(ValueError):
            self.mod.build_text_only_runtime_plan(
                prompt_token_count=14,
                max_seq_len=64,
                max_new_tokens=5,
            )

    def test_build_text_only_runtime_plan_rejects_max_seq_len_overflow(self):
        with self.assertRaises(ValueError):
            self.mod.build_text_only_runtime_plan(
                prompt_token_count=12,
                max_seq_len=12,
                max_new_tokens=8,
            )


if __name__ == "__main__":
    unittest.main()
