from pathlib import Path
import unittest


class StreamingReferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.capture_script = (root / "capture_reference_streaming_contract.py").read_text(
            encoding="utf-8"
        )

    def test_reference_capture_script_pins_upstream_defaults(self):
        self.assertIn("default=0.9", self.capture_script)
        self.assertIn("default=50", self.capture_script)
        self.assertIn("default=1.0", self.capture_script)
        self.assertIn("default=1.05", self.capture_script)
        self.assertIn("default=2.0", self.capture_script)

    def test_reference_capture_script_records_chunk_contract(self):
        self.assertIn('"streaming_chunk_size": 300', self.capture_script)
        self.assertIn('"streaming_left_context_size": 25', self.capture_script)
        self.assertIn('"codec_steps_per_second"', self.capture_script)
        self.assertIn('"codec_trace"', self.capture_script)


if __name__ == "__main__":
    unittest.main()
