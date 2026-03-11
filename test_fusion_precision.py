#!/usr/bin/env python3
"""Test if TRT Myelin fusion causes precision loss in the actual Conformer block.
Compare: no mask (no where ops) vs all-False mask (where ops present but no-op)."""
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


class Block_NoMask(nn.Module):
    """1 block with mask=None (no where ops in graph)."""
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
            audio_signal = layer(
                x=audio_signal,
                att_mask=None,
                pos_emb=pos_emb,
                pad_mask=None,
            )
        return audio_signal, length


class Block_AllFalseMask(nn.Module):
    """1 block with all-False mask (where ops present but no-op)."""
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

        # All-False mask (real mask convention with no padding)
        att_mask = torch.zeros(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.zeros(length.size(0), max_len, dtype=torch.bool, device=audio_signal.device)

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


class Block_RealMask(nn.Module):
    """1 block with real mask creation (all the logical ops)."""
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

        # Real mask creation from _create_masks
        att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)
        pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
        pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
        att_mask = torch.logical_and(pad_mask_for_att, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


class NBlocks_NoMask(nn.Module):
    """N blocks with mask=None."""
    def __init__(self, enc, n):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList(list(enc.layers[:n]))

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=None,
                pos_emb=pos_emb,
                pad_mask=None,
            )
        return audio_signal, length


class NBlocks_RealMask(nn.Module):
    """N blocks with real mask creation."""
    def __init__(self, enc, n):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList(list(enc.layers[:n]))

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        max_len = audio_signal.shape[1]

        att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=audio_signal.device)
        pad_mask = torch.arange(0, max_len, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)
        pad_mask_for_att = pad_mask.unsqueeze(1).repeat([1, max_len, 1])
        pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
        att_mask = torch.logical_and(pad_mask_for_att, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask

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
        import traceback; traceback.print_exc()
        return None, None


print(f"\n{'='*80}")
print("CONFORMER BLOCK PRECISION: mask=None vs all-False mask vs real mask")
print(f"{'='*80}\n")

# Test 1-block variants
for name_cls in [
    ("1block_no_mask", Block_NoMask),
    ("1block_allFalse_mask", Block_AllFalseMask),
    ("1block_real_mask", Block_RealMask),
]:
    name, cls = name_cls
    m = cls(encoder)
    m.eval()
    cos, maxd = test_trt(m, name, mel, mel_len_t)
    status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
    print(f"  {name:35s} cos={cos:.8f}  max_diff={maxd:.6f}  [{status}]")

# Check if eager outputs match between variants
print(f"\n--- Eager cross-check (all variants should produce same output) ---")
m1 = Block_NoMask(encoder); m1.eval()
m2 = Block_AllFalseMask(encoder); m2.eval()
m3 = Block_RealMask(encoder); m3.eval()
with torch.no_grad():
    out1 = m1(mel, mel_len_t)[0]
    out2 = m2(mel, mel_len_t)[0]
    out3 = m3(mel, mel_len_t)[0]

def eager_cos(a, b):
    a, b = a.detach().numpy().flatten(), b.detach().numpy().flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"  no_mask vs allFalse_mask:  cos={eager_cos(out1, out2):.8f}")
print(f"  no_mask vs real_mask:      cos={eager_cos(out1, out3):.8f}")
print(f"  allFalse vs real_mask:     cos={eager_cos(out2, out3):.8f}")

# Test multi-block: no_mask vs real_mask
print(f"\n{'='*80}")
print("MULTI-BLOCK: no_mask vs real_mask compounding")
print(f"{'='*80}\n")

for n in [1, 4, 12, 24]:
    for mask_type, cls in [("no_mask", NBlocks_NoMask), ("real_mask", NBlocks_RealMask)]:
        m = cls(encoder, n)
        m.eval()
        name = f"{n}blocks_{mask_type}"
        cos, maxd = test_trt(m, name, mel, mel_len_t)
        status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
        maxd_s = f"{maxd:.4f}" if maxd else "N/A"
        print(f"  {name:35s} cos={cos:.8f}  max_diff={maxd_s}  [{status}]")
    print()
