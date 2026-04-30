# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical equivalence test for Gemma 4 speech transform vs HF."""

import unittest

import numpy as np
import torch

from executorch.examples.models.gemma4.speech_transform import (
    Gemma4SpeechTransformModel,
)
from transformers.models.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)


class TestSpeechTransform(unittest.TestCase):
    """Test Gemma4SpeechTransformModel against HF Gemma4AudioFeatureExtractor."""

    def setUp(self):
        self.model = Gemma4SpeechTransformModel()
        self.model.eval()
        self.hf_extractor = Gemma4AudioFeatureExtractor()

    def _run_hf(self, waveform_np: np.ndarray) -> torch.Tensor:
        """Run HF _extract_spectrogram directly and return spectrogram as tensor."""
        # Prepare input: HF expects (1, num_samples) with attention mask
        waveform_2d = waveform_np.reshape(1, -1)
        mask = np.ones(waveform_2d.shape[1], dtype=np.float32)
        spectrogram, frame_mask = self.hf_extractor._extract_spectrogram(
            waveform_2d, mask
        )
        # spectrogram shape: [num_frames, n_mels]
        # Apply mask to zero out invalid frames (matching HF __call__ behavior)
        spectrogram = spectrogram * frame_mask[..., None]
        return torch.from_numpy(spectrogram.astype(np.float32))

    def _run_custom(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run our speech transform model."""
        with torch.no_grad():
            return self.model(waveform)

    def _assert_equivalence(self, num_samples: int):
        """Generate random audio and compare custom vs HF output."""
        waveform_np = np.random.RandomState(42).randn(num_samples).astype(np.float32)
        waveform_pt = torch.from_numpy(waveform_np)

        hf_output = self._run_hf(waveform_np)
        custom_output = self._run_custom(waveform_pt)

        # Compare overlapping frames (HF may produce slightly different count
        # due to padding differences)
        min_frames = min(hf_output.shape[0], custom_output.shape[0])
        self.assertGreater(min_frames, 0)
        torch.testing.assert_close(
            custom_output[:min_frames],
            hf_output[:min_frames],
            atol=1e-4,
            rtol=0,
        )

    def test_short_audio_equivalence(self):
        """1 second audio."""
        self._assert_equivalence(16000)

    def test_long_audio_equivalence(self):
        """10 second audio."""
        self._assert_equivalence(160000)

    def test_max_audio_equivalence(self):
        """30 second audio (max supported)."""
        self._assert_equivalence(480000)

    def test_output_shape(self):
        """Verify output has expected shape [num_frames, 128]."""
        num_samples = 16000
        waveform = torch.randn(num_samples)

        with torch.no_grad():
            output = self.model(waveform)

        self.assertEqual(output.dim(), 2)
        self.assertEqual(output.shape[1], 128)
        # Unfold uses frame_size = frame_length + 1 = 321, then drops last sample.
        # With semicausal pad_left = frame_length // 2 = 160:
        # num_frames = (num_samples + pad_left - (frame_length + 1)) // hop_length + 1
        frame_size = 321  # frame_length + 1
        pad_left = 160  # frame_length // 2
        expected_frames = (num_samples + pad_left - frame_size) // 160 + 1
        self.assertEqual(output.shape[0], expected_frames)

    def test_deterministic(self):
        """Same input produces identical output."""
        waveform = torch.randn(16000)
        with torch.no_grad():
            out1 = self.model(waveform)
            out2 = self.model(waveform)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
