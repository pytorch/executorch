import io
import os
import random
import unittest

import numpy as np
import requests
import torch
import torch.nn as nn
import torchaudio

from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch.export import export, ExportedProgram


def read_mp3_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure request is successful
    audio_stream = io.BytesIO(response.content)
    waveform, sample_rate = torchaudio.load(audio_stream, format="mp3")
    return waveform.numpy(), sample_rate


class TestMimiModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup once for all tests: Load model and prepare test data."""

        # Get environment variables (if set), otherwise use default values
        mimi_weight = os.getenv("MIMI_WEIGHT", None)
        hf_repo = os.getenv("HF_REPO", loaders.DEFAULT_REPO)
        device = "cuda" if torch.cuda.device_count() else "cpu"

        def seed_all(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_all(42424242)

        if mimi_weight is None:
            mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        cls.mimi = loaders.get_mimi(mimi_weight, device)
        cls.device = device
        cls.sample_pcm, cls.sample_sr = read_mp3_from_url(
            "https://huggingface.co/lmz/moshi-swift/resolve/main/bria-24khz.mp3"
        )

    def test_mp3_loading(self):
        """Ensure MP3 file loads correctly."""
        self.assertIsInstance(self.sample_pcm, np.ndarray)
        self.assertGreater(self.sample_sr, 0)

    def test_encoding(self):
        """Ensure encoding produces expected tensor shape."""
        pcm_chunk_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        sample_pcm = torch.tensor(self.sample_pcm, device=self.device)
        sample_pcm = sample_pcm[None]
        chunk = sample_pcm[..., 0:pcm_chunk_size]
        encoded = self.mimi.encode(chunk)
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertGreater(encoded.shape[-1], 0)

    def test_decoding(self):
        """Ensure decoding produces expected output."""
        pcm_chunk_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        sample_pcm = torch.tensor(self.sample_pcm, device=self.device)[None]
        chunk = sample_pcm[..., 0:pcm_chunk_size]
        encoded = self.mimi.encode(chunk)
        decoded = self.mimi.decode(encoded)
        self.assertIsInstance(decoded, torch.Tensor)

    def test_streaming_encoding_decoding(self):
        """Test streaming encoding and decoding consistency."""
        pcm_chunk_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        sample_rate = self.mimi.sample_rate
        max_duration_sec = 10.0
        max_duration_len = int(sample_rate * max_duration_sec)

        sample_pcm = torch.tensor(self.sample_pcm, device=self.device)
        if sample_pcm.shape[-1] > max_duration_len:
            sample_pcm = sample_pcm[..., :max_duration_len]
        sample_pcm = sample_pcm[None].to(device=self.device)

        all_codes = []
        for start_idx in range(0, sample_pcm.shape[-1], pcm_chunk_size):
            end_idx = min(sample_pcm.shape[-1], start_idx + pcm_chunk_size)
            chunk = sample_pcm[..., start_idx:end_idx]
            codes = self.mimi.encode(chunk)
            if codes.shape[-1]:
                all_codes.append(codes)

        all_codes_th = torch.cat(all_codes, dim=-1)

        all_pcms = []
        with self.mimi.streaming(1):
            for i in range(all_codes_th.shape[-1]):
                codes = all_codes_th[..., i : i + 1]
                pcm = self.mimi.decode(codes)
                all_pcms.append(pcm)
        all_pcms = torch.cat(all_pcms, dim=-1)

        pcm_ref = self.mimi.decode(all_codes_th)
        self.assertTrue(torch.allclose(pcm_ref, all_pcms, atol=1e-5))

    def test_exported_decoding(self):
        """Ensure exported decoding model is consistent with reference output."""

        class MimiDecode(nn.Module):
            def __init__(self, mimi: nn.Module):
                super().__init__()
                self.mimi_model = mimi

            def forward(self, x):
                return self.mimi_model.decode(x)

        sample_pcm = torch.tensor(self.sample_pcm, device=self.device)[None]
        pcm_chunk_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        chunk = sample_pcm[..., 0:pcm_chunk_size]
        input = self.mimi.encode(chunk)

        mimi_decode = MimiDecode(self.mimi)
        ref_decode_output = mimi_decode(input)
        exported_decode: ExportedProgram = export(mimi_decode, (input,), strict=False)
        ep_decode_output = exported_decode.module()(input)
        self.assertTrue(torch.allclose(ep_decode_output, ref_decode_output, atol=1e-6))

    def test_exported_encoding(self):
        """Ensure exported encoding model is consistent with reference output."""

        class MimiEncode(nn.Module):
            def __init__(self, mimi: nn.Module):
                super().__init__()
                self.mimi_model = mimi

            def forward(self, x):
                return self.mimi_model.encode(x)

        mimi_encode = MimiEncode(self.mimi)
        chunk = torch.tensor(self.sample_pcm, device=self.device)[None][
            ..., 0 : int(self.mimi.sample_rate / self.mimi.frame_rate)
        ]
        ref_encode_output = mimi_encode(chunk)
        exported_encode = export(mimi_encode, (chunk,), strict=False)
        ep_encode_output = exported_encode.module()(chunk)
        self.assertTrue(torch.allclose(ep_encode_output, ref_encode_output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
