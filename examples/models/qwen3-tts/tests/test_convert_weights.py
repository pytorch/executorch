import importlib.util
from pathlib import Path
import unittest

import torch


def _load_convert_module():
    script_path = Path(__file__).resolve().parents[1] / "convert_weights.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_convert_weights", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConvertWeightsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_convert_module()

    def test_extract_prefixed_state_dict_strips_prefix(self):
        state = {
            "decoder.layer.weight": torch.randn(2, 2),
            "decoder.layer.bias": torch.randn(2),
            "encoder.layer.weight": torch.randn(2, 2),
        }
        out = self.mod._extract_prefixed_state_dict(state, "decoder.")
        self.assertIn("layer.weight", out)
        self.assertIn("layer.bias", out)
        self.assertNotIn("decoder.layer.weight", out)
        self.assertNotIn("encoder.layer.weight", out)

    def test_sanitize_model_id(self):
        self.assertEqual(self.mod._sanitize_model_id("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), "Qwen_Qwen3-TTS-12Hz-0.6B-Base")
        self.assertEqual(self.mod._sanitize_model_id("  "), "qwen3_tts_model")

    def test_build_decoder_metadata(self):
        root_cfg = {
            "tokenizer_type": "qwen3_tts_tokenizer_v2",
            "tts_model_type": "base",
        }
        speech_cfg = {
            "output_sample_rate": 24000,
            "decode_upsample_rate": 1920,
            "decoder_config": {
                "num_quantizers": 16,
                "codebook_size": 2048,
            },
        }
        meta = self.mod._build_decoder_metadata(
            model_id_or_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            root_cfg=root_cfg,
            speech_tokenizer_cfg=speech_cfg,
            decoder_checkpoint_name="qwen3_tts_decoder.pth",
        )
        self.assertEqual(meta["model_id_or_path"], "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.assertEqual(meta["tokenizer_type"], "qwen3_tts_tokenizer_v2")
        self.assertEqual(meta["tts_model_type"], "base")
        self.assertEqual(meta["output_sample_rate"], 24000)
        self.assertEqual(meta["decode_upsample_rate"], 1920)
        self.assertEqual(meta["num_quantizers"], 16)
        self.assertEqual(meta["codebook_size"], 2048)


if __name__ == "__main__":
    unittest.main()
