import io
import os
import random
import unittest

import numpy as np
import requests
import torch
import torch.nn as nn
import torchaudio
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch.export import export, ExportedProgram
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

proxies = {
    "http": "http://fwdproxy:8080",
    "https": "http://fwdproxy:8080",
}


def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> float:
    assert x.shape == y.shape, "Tensor shapes do not match"
    x = x.float()
    y = y.float()
    error = x - y
    original_power = torch.mean(torch.pow(x, 2))
    error_power = torch.mean(torch.pow(error, 2))
    sqnr = 10 * torch.log10(original_power / error_power)
    return sqnr.item()


def read_mp3_from_url(url):
    try:
        response = requests.get(url)
    except:
        # FB-only hack, need to use a forwarding proxy to get url
        response = requests.get(url, proxies=proxies)

    response.raise_for_status()  # Ensure request is successful
    audio_stream = io.BytesIO(response.content)
    waveform, sample_rate = torchaudio.load(audio_stream, format="mp3")
    return waveform.numpy(), sample_rate


class TestMimiModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup once for all tests: Load model and prepare test data."""

        # Get environment variables (if set), otherwise use default values
        cls.mimi_weight = os.getenv("MIMI_WEIGHT", None)
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

        if cls.mimi_weight is None:
            try:
                cls.mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
            except:
                cls.mimi_weight = hf_hub_download(
                    hf_repo, loaders.MIMI_NAME, proxies=proxies
                )

        cls.mimi = loaders.get_mimi(cls.mimi_weight, device)
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

        pcm_ref = self.mimi.decode(all_codes_th)

        all_pcms = []
        for i in range(all_codes_th.shape[-1]):
            codes = all_codes_th[..., i : i + 1]
            pcm = self.mimi.decode(codes)
            all_pcms.append(pcm)
        all_pcms = torch.cat(all_pcms, dim=-1)
        sqnr = compute_sqnr(pcm_ref, all_pcms)
        print(f"sqnr = {sqnr} dB")
        self.assertTrue(sqnr > 4)

        all_pcms_streaming = []
        with self.mimi.streaming(1):
            for i in range(all_codes_th.shape[-1]):
                codes = all_codes_th[..., i : i + 1]
                pcm_streaming = self.mimi.decode(codes)
                all_pcms_streaming.append(pcm_streaming)
        all_pcms_streaming = torch.cat(all_pcms_streaming, dim=-1)
        sqnr_streaming = compute_sqnr(pcm_ref, all_pcms_streaming)
        print(f"sqnr_streaming = {sqnr_streaming} dB")
        self.assertTrue(sqnr_streaming > 70)

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

    def test_exported_decoder_xnnpack(self):
        class MimiDecode(nn.Module):
            def __init__(self, mimi: nn.Module):
                super().__init__()
                self.mimi_model = mimi

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.mimi_model.upsample(x)
                (emb,) = self.mimi_model.decoder_transformer(x)
                emb.transpose(1, 2)
                out = self.mimi_model.decoder(emb)
                return out

        emb_input = torch.rand(1, 1, 512, device="cpu")
        mimi_cpu = loaders.get_mimi(self.mimi_weight, "cpu")
        mimi_decode = MimiDecode(mimi_cpu)
        mimi_decode.eval()
        mimi_decode(emb_input)

        exported_decode: ExportedProgram = export(
            mimi_decode, (emb_input,), strict=False
        )
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
        )
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(quantization_config)
        m = exported_decode.module()
        m = prepare_pt2e(m, quantizer)
        m(emb_input)
        m = convert_pt2e(m)
        print("quantized graph:")
        print(m.graph)
        # Export quantized module
        exported_decode: ExportedProgram = export(m, (emb_input,), strict=False)
        # Lower
        edge_manager = to_edge_transform_and_lower(
            exported_decode,
            partitioner=[XnnpackPartitioner()],
        )
        print("delegate graph:")
        print_delegation_info(edge_manager.exported_program().graph_module)
        exec_prog = edge_manager.to_executorch()
        output_file = "/tmp/mimi_decode.pte"
        with open(output_file, "wb") as file:
            exec_prog.write_to_file(file)

        eager_res = mimi_decode(emb_input)
        runtime = Runtime.get()
        program = runtime.load_program(output_file)
        method = program.load_method("forward")
        flattened_x = tree_flatten(emb_input)[0]
        res = method.execute(flattened_x)
        # Compare results
        sqnr = compute_sqnr(eager_res, res[0])
        print(f"SQNR: {sqnr}")
        # Don't check for exact equality, but check that the SQNR is high enough
        # torch.testing.assert_close(eager_res, res[0], atol=4e-3, rtol=1e-3)
        self.assertGreater(sqnr, 25.0)


if __name__ == "__main__":
    unittest.main()
