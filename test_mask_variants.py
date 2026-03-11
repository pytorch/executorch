#!/usr/bin/env python3
"""Narrow down: which specific mask operation breaks TRT?
Test each mask variant incrementally.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")

from executorch.exir import to_edge
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from executorch.runtime import Runtime
from export_parakeet_tdt import PreprocessorWrapper

print("Loading model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()
encoder = model.encoder

# Prepare mel
import soundfile as sf
data, sr = sf.read("/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav", dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)
audio_1d = torch.from_numpy(data)
audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

preproc = PreprocessorWrapper(model.preprocessor)
preproc.float().eval()

with torch.no_grad():
    mel, mel_len = preproc(audio_1d, audio_len)

mel_len_val = mel_len.item()
mel_len_t = torch.tensor([mel_len_val], dtype=torch.int64)
print(f"Mel: {mel.shape}, mel_len={mel_len_val}")

runtime = Runtime.get()


class EncoderMaskVariant(nn.Module):
    """Encoder with configurable mask variant."""
    def __init__(self, enc, mask_variant):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])  # Just 1 block for speed
        self.mask_variant = mask_variant

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        max_len = audio_signal.shape[1]

        if self.mask_variant == "simple":
            # True = valid
            pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
            att_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        elif self.mask_variant == "inverted_padmask":
            # Same as simple, but invert pad_mask (True = padding)
            pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
            att_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            pad_mask = ~pad_mask  # INVERT

        elif self.mask_variant == "inverted_attmask":
            # Same as simple, but invert att_mask
            pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
            att_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            att_mask = ~att_mask  # INVERT

        elif self.mask_variant == "full_2d_attmask":
            # Create 2D attention mask [1, T, T] (all True for valid pairs)
            pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)
            # Build 2D: pad_mask[b,i] AND pad_mask[b,j]
            att_mask_2d = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)  # [B, T, T]
            att_mask = att_mask_2d  # NOT inverted
            # pad_mask stays True=valid

        elif self.mask_variant == "real_noinvert":
            # Real mask logic but WITHOUT the final inversions
            pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
                length.size(0), -1
            ) < length.unsqueeze(-1)
            att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
            pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
            pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
            att_mask = torch.logical_and(pad_mask_for_att, att_mask)
            # Do NOT invert: pad_mask True=valid, att_mask True=attend

        elif self.mask_variant == "real_full":
            # Full real mask with inversions
            pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
                length.size(0), -1
            ) < length.unsqueeze(-1)
            att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
            pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
            pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
            att_mask = torch.logical_and(pad_mask_for_att, att_mask)
            att_mask = ~att_mask  # INVERT
            pad_mask = ~pad_mask  # INVERT

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


def test_trt(submodel, name, mel, mel_len_t):
    with torch.no_grad():
        eager_out = submodel(mel, mel_len_t)
    eager_main = eager_out[0]

    try:
        ep = torch.export.export(submodel, (mel,), kwargs={"length": mel_len_t})
        edge = to_edge(ep)
        edge = edge.to_backend(TensorRTPartitioner())
        et_prog = edge.to_executorch()

        tmp_path = f"/tmp/test_{name}.pte"
        with open(tmp_path, "wb") as f:
            f.write(et_prog.buffer)

        program = runtime.load_program(tmp_path)
        method = program.load_method("forward")
        trt_out = method.execute([mel, mel_len_t])
        trt_main = trt_out[0]

        eager_np = eager_main.detach().numpy()
        trt_np = trt_main.detach().numpy()

        min_shape = [min(e, t) for e, t in zip(eager_np.shape, trt_np.shape)]
        eager_np = eager_np[tuple(slice(0, s) for s in min_shape)]
        trt_np = trt_np[tuple(slice(0, s) for s in min_shape)]

        cos = np.dot(eager_np.flatten(), trt_np.flatten()) / (
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8
        )
        max_diff = np.abs(eager_np - trt_np).max()
        os.unlink(tmp_path)
        return cos, max_diff
    except Exception as e:
        print(f"  {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


variants = [
    "simple",            # Baseline: True=valid, 4D att_mask
    "inverted_padmask",  # Just invert pad_mask
    "inverted_attmask",  # Just invert att_mask
    "full_2d_attmask",   # Use 2D att_mask [B,T,T] without inversion
    "real_noinvert",     # Real mask logic without final inversions
    "real_full",         # Real mask logic with inversions (should break)
]

print(f"\n{'='*60}")
print("MASK VARIANT TRT ACCURACY TEST (1 block)")
print(f"{'='*60}\n")

results = []
for v in variants:
    m = EncoderMaskVariant(encoder, v)
    m.eval()
    cos, maxd = test_trt(m, f"mask_{v}", mel, mel_len_t)
    status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
    cos_s = f"{cos:.8f}" if cos else "ERROR"
    maxd_s = f"{maxd:.6f}" if maxd else "N/A"
    print(f"  {v:25s} cosine={cos_s}  max_diff={maxd_s}  [{status}]")
    results.append((v, cos, maxd))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for v, cos, maxd in results:
    status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
    cos_s = f"{cos:.8f}" if cos else "ERROR"
    print(f"  {v:25s} cosine={cos_s}  [{status}]")
