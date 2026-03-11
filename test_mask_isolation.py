#!/usr/bin/env python3
"""Test if the mask creation logic causes the TRT accuracy drop.
Compare: my simple masks vs the real encoder's _create_masks.
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


class EncoderSimpleMask(nn.Module):
    """Encoder 24 blocks with SIMPLE mask (True=valid)."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList(list(enc.layers))

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        # Simple mask: True=valid, False=padding
        max_len = audio_signal.shape[1]
        pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=pad_mask.unsqueeze(1).unsqueeze(1),
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length


class EncoderRealMask(nn.Module):
    """Encoder 24 blocks with REAL mask logic (same as _create_masks)."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList(list(enc.layers))

    def forward(self, audio_signal, length):
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        max_audio_length = audio_signal.size(1)

        # Real mask creation (from _create_masks with att_context_size=[-1,-1])
        att_mask = torch.ones(1, max_audio_length, max_audio_length,
                              dtype=torch.bool, device=audio_signal.device)
        # att_context_size is [-1, -1], so no triu/tril

        pad_mask = torch.arange(0, max_audio_length, device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)

        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(
            pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
        )
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length


def test_trt(submodel, name, mel, mel_len_t):
    """Export through TRT and compare."""
    with torch.no_grad():
        eager_out = submodel(mel, mel_len_t)
    eager_main = eager_out[0]

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"  Eager output shape: {eager_main.shape}")

    try:
        ep = torch.export.export(submodel, (mel,), kwargs={"length": mel_len_t})
        print(f"  Export: {len(list(ep.graph.nodes))} nodes")

        # Count ops
        from collections import Counter
        ops = Counter()
        for n in ep.graph.nodes:
            if n.op == "call_function":
                ops[str(n.target)] += 1
        # Show ops unique to this graph
        print(f"  Top ops: {ops.most_common(10)}")

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
        print(f"  Cosine: {cos:.8f}")
        print(f"  Max diff: {max_diff:.6f}")

        os.unlink(tmp_path)
        return cos
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test both
simple = EncoderSimpleMask(encoder)
simple.eval()
cos_simple = test_trt(simple, "simple_mask", mel, mel_len_t)

real = EncoderRealMask(encoder)
real.eval()
cos_real = test_trt(real, "real_mask", mel, mel_len_t)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Simple mask (True=valid):    cosine = {cos_simple:.8f}" if cos_simple else "  Simple mask: FAILED")
print(f"  Real mask (True=padding):    cosine = {cos_real:.8f}" if cos_real else "  Real mask: FAILED")
if cos_simple and cos_real:
    print(f"\n  => Mask logic is {'THE CAUSE' if cos_real < 0.9 and cos_simple > 0.99 else 'NOT the cause'}")
