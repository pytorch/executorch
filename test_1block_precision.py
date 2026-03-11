#!/usr/bin/env python3
"""Quick test: does mask=None vs mask=allFalse make a difference in TRT precision?
Tests ONLY 1-block variants (fast builds)."""
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
mel_len_t = torch.tensor([mel_len.item()], dtype=torch.int64)
print(f"Mel: {mel.shape}, mel_len={mel_len.item()}")

runtime = Runtime.get()


class Block_NoMask(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        for layer in self.layers:
            audio_signal = layer(x=audio_signal, att_mask=None, pos_emb=pos_emb, pad_mask=None)
        return audio_signal, length


class Block_AllFalseMask(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        max_len = audio_signal.shape[1]
        att_mask = torch.zeros(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.zeros(length.size(0), max_len, dtype=torch.bool, device=audio_signal.device)
        for layer in self.layers:
            audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        return audio_signal, length


class Block_RealMask(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList([enc.layers[0]])

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        max_len = audio_signal.shape[1]
        att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1) < length.unsqueeze(-1)
        pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
        pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
        att_mask = torch.logical_and(pad_mask_for_att, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask
        for layer in self.layers:
            audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
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
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8)
        max_diff = np.abs(eager_np - trt_np).max()
        mean_diff = np.abs(eager_np - trt_np).mean()
        os.unlink(tmp_path)
        return cos, max_diff, mean_diff
    except Exception as e:
        print(f"  {name} FAILED: {e}")
        import traceback; traceback.print_exc()
        return None, None, None


print(f"\n{'='*80}")
print("1-BLOCK PRECISION: no_mask vs allFalse vs real_mask")
print(f"{'='*80}\n")

# Check eager outputs are identical
print("--- Eager cross-check ---")
m1 = Block_NoMask(encoder); m1.eval()
m2 = Block_AllFalseMask(encoder); m2.eval()
m3 = Block_RealMask(encoder); m3.eval()
with torch.no_grad():
    out1 = m1(mel, mel_len_t)[0]
    out2 = m2(mel, mel_len_t)[0]
    out3 = m3(mel, mel_len_t)[0]

def cos_sim(a, b):
    a, b = a.detach().numpy().flatten(), b.detach().numpy().flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

c12 = cos_sim(out1, out2)
c13 = cos_sim(out1, out3)
c23 = cos_sim(out2, out3)
md12 = np.abs(out1.numpy() - out2.numpy()).max()
md13 = np.abs(out1.numpy() - out3.numpy()).max()
md23 = np.abs(out2.numpy() - out3.numpy()).max()
print(f"  no_mask vs allFalse:  cos={c12:.8f}  max_diff={md12:.8f}")
print(f"  no_mask vs real_mask: cos={c13:.8f}  max_diff={md13:.8f}")
print(f"  allFalse vs real:     cos={c23:.8f}  max_diff={md23:.8f}")

print(f"\n--- TRT precision ---")
for name, cls in [
    ("1block_no_mask", Block_NoMask),
    ("1block_allFalse_mask", Block_AllFalseMask),
    ("1block_real_mask", Block_RealMask),
]:
    m = cls(encoder); m.eval()
    cos, maxd, meand = test_trt(m, name, mel, mel_len_t)
    status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
    print(f"  {name:35s} cos={cos:.8f}  max_diff={maxd:.6f}  mean_diff={meand:.8f}  [{status}]")

print(f"\nDone.")
