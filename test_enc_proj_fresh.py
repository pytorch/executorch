#!/usr/bin/env python3
"""Test the full EncoderWithProjection via fresh TRT export.
Compare with the pre-built PTE to understand the 0.66 cosine.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")

from export_parakeet_tdt import EncoderWithProjection, PreprocessorWrapper
from executorch.exir import to_edge
from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

print("Loading Parakeet model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()

encoder_proj = EncoderWithProjection(model.encoder, model.joint)
encoder_proj.eval()

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
T_actual = mel.shape[2]
mel_len_t = torch.tensor([mel_len_val], dtype=torch.int64)
print(f"Mel: {mel.shape}, mel_len={mel_len_val}")

# Eager reference
with torch.no_grad():
    eager_out, eager_enc_len = encoder_proj(mel, mel_len_t)
valid = eager_enc_len.item()
print(f"Eager: {eager_out.shape}, enc_len={valid}")

# Test 1: Fresh TRT export at actual shape (744)
print(f"\n{'='*60}")
print(f"Test 1: Fresh TRT export at actual mel shape ({T_actual})")
print(f"{'='*60}")

ep = torch.export.export(
    encoder_proj,
    (mel,),
    kwargs={"length": mel_len_t},
)
print(f"Exported: {len(list(ep.graph.nodes))} nodes")

edge = to_edge(ep)
edge = edge.to_backend(TensorRTPartitioner())
et_prog = edge.to_executorch()

tmp_path = "/tmp/test_enc_proj_actual.pte"
with open(tmp_path, "wb") as f:
    f.write(et_prog.buffer)

from executorch.runtime import Runtime
runtime = Runtime.get()
program = runtime.load_program(tmp_path)
method = program.load_method("forward")
trt_out = method.execute([mel, mel_len_t])
trt_main = trt_out[0].detach().numpy()
eager_np = eager_out.detach().numpy()

# Compare valid region only
trt_valid = trt_main[0, :valid, :]
eager_valid = eager_np[0, :valid, :]

cos = np.dot(eager_valid.flatten(), trt_valid.flatten()) / (
    np.linalg.norm(eager_valid.flatten()) * np.linalg.norm(trt_valid.flatten()) + 1e-8
)
max_diff = np.abs(eager_valid - trt_valid).max()
print(f"  Cosine (valid region): {cos:.8f}")
print(f"  Max diff: {max_diff:.6f}")
os.unlink(tmp_path)

# Test 2: Fresh TRT export at padded shape (5000)
print(f"\n{'='*60}")
print(f"Test 2: Fresh TRT export at padded mel shape (5000)")
print(f"{'='*60}")

T_padded = 5000
mel_padded = torch.zeros(1, 128, T_padded, dtype=torch.float32)
mel_padded[:, :, :T_actual] = mel

with torch.no_grad():
    eager_padded, eager_padded_len = encoder_proj(mel_padded, mel_len_t)
valid_padded = eager_padded_len.item()
print(f"Eager padded: {eager_padded.shape}, enc_len={valid_padded}")

ep2 = torch.export.export(
    encoder_proj,
    (mel_padded,),
    kwargs={"length": mel_len_t},
)
print(f"Exported: {len(list(ep2.graph.nodes))} nodes")

edge2 = to_edge(ep2)
edge2 = edge2.to_backend(TensorRTPartitioner())
et_prog2 = edge2.to_executorch()

tmp_path2 = "/tmp/test_enc_proj_padded.pte"
with open(tmp_path2, "wb") as f:
    f.write(et_prog2.buffer)

program2 = runtime.load_program(tmp_path2)
method2 = program2.load_method("forward")
trt_out2 = method2.execute([mel_padded, mel_len_t])
trt_main2 = trt_out2[0].detach().numpy()
eager_padded_np = eager_padded.detach().numpy()

trt_valid2 = trt_main2[0, :valid_padded, :]
eager_valid2 = eager_padded_np[0, :valid_padded, :]

cos2 = np.dot(eager_valid2.flatten(), trt_valid2.flatten()) / (
    np.linalg.norm(eager_valid2.flatten()) * np.linalg.norm(trt_valid2.flatten()) + 1e-8
)
max_diff2 = np.abs(eager_valid2 - trt_valid2).max()
print(f"  Cosine (valid region): {cos2:.8f}")
print(f"  Max diff: {max_diff2:.6f}")
os.unlink(tmp_path2)

# Also compare eager@actual vs eager@padded to check if padding affects eager
eager_valid_actual = eager_np[0, :valid, :]
eager_valid_padded = eager_padded_np[0, :valid_padded, :]
min_valid = min(valid, valid_padded)
cos_eager = np.dot(eager_valid_actual[:min_valid].flatten(), eager_valid_padded[:min_valid].flatten()) / (
    np.linalg.norm(eager_valid_actual[:min_valid].flatten()) * np.linalg.norm(eager_valid_padded[:min_valid].flatten()) + 1e-8
)
print(f"\n  Eager@actual vs Eager@padded cosine: {cos_eager:.8f}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Fresh TRT @ actual ({T_actual}):  cosine = {cos:.8f}")
print(f"  Fresh TRT @ padded (5000):  cosine = {cos2:.8f}")
print(f"  Eager actual vs padded:     cosine = {cos_eager:.8f}")
