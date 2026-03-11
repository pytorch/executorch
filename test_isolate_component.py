#!/usr/bin/env python3
"""Isolate: is the 0.66 cosine from encoder or from projection?
Test each component separately through TRT.
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
from export_parakeet_tdt import EncoderWithProjection, PreprocessorWrapper

print("Loading model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()

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


def test_trt(submodel, name, args, kwargs, et_inputs):
    """Export through TRT and compare."""
    with torch.no_grad():
        eager_out = submodel(*args, **kwargs)

    if isinstance(eager_out, tuple):
        eager_main = eager_out[0]
    else:
        eager_main = eager_out

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"  Eager output shape: {eager_main.shape}")
    print(f"  Eager range: [{eager_main.min():.4f}, {eager_main.max():.4f}]")

    try:
        ep = torch.export.export(submodel, args, kwargs=kwargs)
        print(f"  Export OK, {len(list(ep.graph.nodes))} nodes")

        edge = to_edge(ep)
        edge = edge.to_backend(TensorRTPartitioner())
        et_prog = edge.to_executorch()

        tmp_path = f"/tmp/test_{name}.pte"
        with open(tmp_path, "wb") as f:
            f.write(et_prog.buffer)

        program = runtime.load_program(tmp_path)
        method = program.load_method("forward")
        trt_out = method.execute(et_inputs)
        trt_main = trt_out[0]

        eager_np = eager_main.detach().numpy()
        trt_np = trt_main.detach().numpy()

        # Match shapes
        min_shape = [min(e, t) for e, t in zip(eager_np.shape, trt_np.shape)]
        eager_np = eager_np[tuple(slice(0, s) for s in min_shape)]
        trt_np = trt_np[tuple(slice(0, s) for s in min_shape)]

        cos = np.dot(eager_np.flatten(), trt_np.flatten()) / (
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8
        )
        max_diff = np.abs(eager_np - trt_np).max()
        print(f"  TRT range: [{trt_np.min():.4f}, {trt_np.max():.4f}]")
        print(f"  Cosine: {cos:.8f}")
        print(f"  Max diff: {max_diff:.6f}")

        os.unlink(tmp_path)
        return cos
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# Wrapper to avoid NeMo's @typecheck issue
class EncoderOnly(nn.Module):
    """Wraps model.encoder to avoid NeMo @typecheck."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length):
        return self.encoder(audio_signal=audio_signal, length=length)


class EncoderNoFinalTranspose(nn.Module):
    """Encoder that returns [B, T, D] instead of [B, D, T].
    This matches what my progressive test does.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal, length):
        encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        # encoded is [B, D, T], transpose to [B, T, D]
        return encoded.transpose(1, 2), enc_len


class ProjectionOnly(nn.Module):
    def __init__(self, project_encoder):
        super().__init__()
        self.project_encoder = project_encoder

    def forward(self, encoded_t):
        return self.project_encoder(encoded_t)


# Get encoder output eagerly for projection test
with torch.no_grad():
    encoded, enc_len = model.encoder(audio_signal=mel, length=mel_len_t)
print(f"Encoder output: {encoded.shape}, enc_len={enc_len.item()}")

# Transpose for projection input
encoded_t = encoded.transpose(1, 2)  # [B, T, D]
print(f"Transposed: {encoded_t.shape}")

results = {}

# Test 1: Actual encoder through TRT
enc_wrapper = EncoderOnly(model.encoder)
enc_wrapper.eval()
cos = test_trt(enc_wrapper, "encoder_only",
               (mel,), {"length": mel_len_t}, [mel, mel_len_t])
results["Encoder only"] = cos

# Test 2: Projection only through TRT
proj = ProjectionOnly(model.joint.project_encoder)
proj.eval()
cos = test_trt(proj, "projection_only",
               (encoded_t,), {}, [encoded_t])
results["Projection only"] = cos

# Test 3: Full EncoderWithProjection
enc_proj = EncoderWithProjection(model.encoder, model.joint)
enc_proj.eval()
cos = test_trt(enc_proj, "encoder_with_projection",
               (mel,), {"length": mel_len_t}, [mel, mel_len_t])
results["EncoderWithProjection"] = cos

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, cos in results.items():
    if cos:
        status = "PASS" if cos > 0.99 else "WARN" if cos > 0.9 else "FAIL"
        print(f"  {name:30s} cosine = {cos:.8f}  [{status}]")
    else:
        print(f"  {name:30s} FAILED")
