# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host waveform regression using the real eight-layer static codec path."""

import unittest

import torch
from executorch.examples.qualcomm.oss_scripts.lfm3.codec import (
    codec_window_plan,
    StaticCodecDecoder,
)


class CodecWindowTest(unittest.TestCase):
    def make_codec(self, frames: int, hop: int = 8) -> StaticCodecDecoder:
        cfg = {
            "fsq_levels": [8, 7, 6, 6],
            "n_q": 12,
            "stft_frame_stack": 6,
            "stft_n_fft": hop * 4,
            "stft_win_length": hop * 4,
            "stft_hop_length": hop,
            "decoder_window": 30,
            "encoder_dim": 16,
            "quantizer_dim": 16,
        }
        codec = StaticCodecDecoder(
            cfg,
            frames,
            torch.hann_window(hop * 4),
            hidden_ffn=32,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
        ).eval()
        with torch.no_grad():
            for layer in codec.layers:
                if not layer.is_attention:
                    layer.conv.conv_weight.normal_(0, 0.15)
            codec.lin.weight.mul_(0.2)
            codec.lin.bias.zero_()
        return codec

    def overlap_decode(
        self, codec: StaticCodecDecoder, codes: torch.Tensor, left: int, right: int
    ) -> tuple[torch.Tensor, list[int]]:
        pieces, boundaries = [], []
        emitted = 0
        for start, valid, offset, count in codec_window_plan(
            codes.shape[1], codec.frames, left, right
        ):
            padded = torch.full(
                (codec.n_q, codec.frames), codec.audio_eos_code, dtype=torch.int32
            )
            padded[:, :valid] = codes[:, start : start + valid]
            audio = codec(padded, torch.tensor([valid], dtype=torch.int32))
            begin = offset * codec.frame_samples
            pieces.append(audio[begin : begin + count * codec.frame_samples])
            emitted += count
            boundaries.append(emitted * codec.frame_samples)
        return torch.cat(pieces), boundaries[:-1]

    def test_context_is_composed_across_all_layers(self):
        codec = self.make_codec(128)
        self.assertEqual(codec.history_steps, 3 * 29 + 5 * 6)
        self.assertEqual((codec.left_context, codec.right_context), (20, 1))
        self.assertEqual(
            list(codec_window_plan(257, 128, 20, 1)),
            [(0, 128, 0, 127), (107, 128, 20, 107), (214, 43, 20, 23)],
        )
        with self.assertRaises(ValueError):
            list(codec_window_plan(257, 21, 20, 1))
        self.assertEqual(list(codec_window_plan(0, 128, 20, 1)), [])

    def test_overlap_matches_one_shot_beyond_128_frames(self):
        # Production temporal geometry: 75 Hz spectra, 12.5 Hz codec frames,
        # 1280-point iSTFT, 320-sample hop, three attention and five conv layers.
        # Narrow channel/FFN widths keep this host test independent of weights.
        torch.manual_seed(41)
        windowed = self.make_codec(128, hop=320)
        one_shot = self.make_codec(257, hop=320)
        one_shot.load_state_dict(windowed.state_dict(), strict=True)
        codes = torch.randint(0, windowed.audio_eos_code, (12, 257), dtype=torch.int32)
        with torch.no_grad():
            expected = one_shot(codes)
            got, boundaries = self.overlap_decode(windowed, codes, 20, 1)
        self.assertEqual(got.numel(), 257 * 1920)
        torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-5)
        for boundary in boundaries + [128 * 1920]:
            torch.testing.assert_close(
                got[boundary - 960 : boundary + 960],
                expected[boundary - 960 : boundary + 960],
                rtol=2e-4,
                atol=2e-5,
            )
        torch.testing.assert_close(got[-1920:], expected[-1920:], rtol=2e-4, atol=2e-5)

    def test_regression_detects_missing_history_and_missing_future_spectra(self):
        torch.manual_seed(9)
        windowed = self.make_codec(128)
        one_shot = self.make_codec(257)
        one_shot.load_state_dict(windowed.state_dict(), strict=True)
        codes = torch.randint(0, windowed.audio_eos_code, (12, 257), dtype=torch.int32)
        with torch.no_grad():
            expected = one_shot(codes)
            got, _ = self.overlap_decode(windowed, codes, 20, 1)
            no_history, _ = self.overlap_decode(windowed, codes, 0, 1)
            no_future, _ = self.overlap_decode(windowed, codes, 20, 0)
        torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-5)
        self.assertGreater((no_history - expected).abs().max().item(), 1e-3)
        self.assertGreater((no_future - expected).abs().max().item(), 1e-3)

    def test_partial_final_window_excludes_padding_from_synthesis_and_envelope(self):
        torch.manual_seed(7)
        windowed = self.make_codec(128)
        for length in (1, 20, 127, 128, 129, 234, 235):
            with self.subTest(length=length):
                one_shot = self.make_codec(length)
                one_shot.load_state_dict(windowed.state_dict(), strict=True)
                codes = torch.randint(
                    0, windowed.audio_eos_code, (12, length), dtype=torch.int32
                )
                with torch.no_grad():
                    expected = one_shot(codes)
                    got, _ = self.overlap_decode(windowed, codes, 20, 1)
                torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-5)
                self.assertTrue(torch.isfinite(got).all())
