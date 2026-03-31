from pathlib import Path
import unittest


class MlxBackendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.backend = (root / "mlx_backend.py").read_text(encoding="utf-8")
        cls.benchmark = (root / "benchmark_mlx.py").read_text(encoding="utf-8")

    def test_backend_defines_cached_icl_session(self):
        self.assertIn("class Qwen3TTSMlxIclSession", self.backend)
        self.assertIn("_prepare_cached_icl_generation_inputs", self.backend)
        self.assertIn("self.model._prepare_icl_generation_inputs =", self.backend)

    def test_backend_caches_reference_conditioning(self):
        self.assertIn("ref_codes = self.model.speech_tokenizer.encode(ref_audio)", self.backend)
        self.assertIn("ref_text_embed = self.model.talker.text_projection(", self.backend)
        self.assertIn("role_embed = self.model.talker.text_projection(", self.backend)
        self.assertIn("codec_with_text_pad", self.backend)
        self.assertIn("ref_text_with_codec_pad", self.backend)
        self.assertIn("combined_prefix", self.backend)

    def test_benchmark_compares_baseline_and_cached_session(self):
        self.assertIn("backend.create_icl_session(", self.benchmark)
        self.assertIn("Cached session speedup", self.benchmark)
        self.assertIn("Average throughput", self.benchmark)
        self.assertIn("default=4.0", self.benchmark)


if __name__ == "__main__":
    unittest.main()
