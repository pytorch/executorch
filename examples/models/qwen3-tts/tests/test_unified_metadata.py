import json
from pathlib import Path
import unittest


class UnifiedMetadataTest(unittest.TestCase):
    def test_checked_in_unified_manifests_expose_current_method_surface(self):
        root = Path(__file__).resolve().parents[1]
        manifests = [
            root / "qwen3_tts_exports_unified" / "export_manifest.json",
            root / "qwen3_tts_exports_unified_q4emb" / "export_manifest.json",
            root / "qwen3_tts_exports_unified_q8emb" / "export_manifest.json",
        ]
        expected_methods = [
            "encode_text",
            "talker",
            "code_predictor",
            "codec_embed",
            "cp_head",
            "cp_generate",
            "decode_audio",
            "decode_audio_stream",
        ]

        for manifest_path in manifests:
            with self.subTest(manifest_path=manifest_path.name):
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                self.assertEqual(manifest["methods"], expected_methods)

    def test_checked_in_unified_manifests_capture_text_prompt_contract(self):
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "qwen3_tts_exports_unified" / "export_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["prompt_contract"], "assistant_chat_text_v1")
        self.assertTrue(manifest["requires_tokenizer"])
        self.assertTrue(manifest["supports_text_only_synthesis"])
        self.assertFalse(manifest["supports_voice_clone_synthesis"])
        self.assertEqual(manifest["text_prompt_min_token_count"], 9)
        self.assertEqual(manifest["text_prompt_prefill_token_count"], 8)
        self.assertEqual(manifest["text_prompt_prefill_token_count_with_language"], 9)
        self.assertEqual(manifest["text_prompt_trailing_template_token_count"], 5)
        self.assertEqual(manifest["cp_generate_contract_version"], 2)
        self.assertEqual(manifest["cp_generate_fast_top_k"], 50)
        self.assertEqual(manifest["cp_generate_sampler"], "cdf_topk50_no_top_p_v2")
        self.assertEqual(manifest["generation_backend_code"], 1)
        self.assertEqual(manifest["decoder_backend_code"], 1)
        self.assertEqual(manifest["prefer_streaming_decoder_surface"], 0)
        self.assertEqual(manifest["streaming_decoder_contract_version"], 1)
        self.assertEqual(manifest["streaming_decoder_chunk_size"], 300)
        self.assertEqual(manifest["streaming_decoder_left_context_size"], 25)
        self.assertEqual(manifest["streaming_decoder_max_codes"], 325)
        self.assertEqual(manifest["codec_think_id"], 2154)
        self.assertEqual(manifest["codec_language_english_id"], 2050)


if __name__ == "__main__":
    unittest.main()
