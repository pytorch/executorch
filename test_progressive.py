#!/usr/bin/env python3
"""Test TRT accuracy for minimal encoder submodels.
Progressively adds layers to find where the accuracy drops.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")

print("Loading Parakeet model...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
)
model.eval()

encoder = model.encoder


class SubsamplingOnly(nn.Module):
    """Just the subsampling (ConvSubsampling)."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode

    def forward(self, audio_signal, length):
        audio_signal = audio_signal.transpose(1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        return audio_signal, length


class SubsamplingPlusPosEnc(nn.Module):
    """Subsampling + positional encoding."""
    def __init__(self, enc):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc

    def forward(self, audio_signal, length):
        audio_signal = audio_signal.transpose(1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)
        return audio_signal, pos_emb, length


class EncoderNBlocks(nn.Module):
    """Subsampling + pos_enc + N Conformer blocks."""
    def __init__(self, enc, n_blocks):
        super().__init__()
        self.pre_encode = enc.pre_encode
        self.pos_enc = enc.pos_enc
        self.layers = nn.ModuleList(list(enc.layers[:n_blocks]))

    def forward(self, audio_signal, length):
        audio_signal = audio_signal.transpose(1, 2)
        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        # Create padding mask
        max_len = audio_signal.shape[1]
        pad_mask = torch.arange(max_len, device=audio_signal.device).unsqueeze(0) < length.unsqueeze(1)

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=pad_mask.unsqueeze(1).unsqueeze(1),
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )
        return audio_signal, length


def test_model_trt(model_cls, name, enc, mel, mel_len, **kwargs):
    """Export model to TRT and compare with eager."""
    from executorch.exir import to_edge
    from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

    submodel = model_cls(enc, **kwargs)
    submodel.eval()

    mel_len_t = torch.tensor([mel_len], dtype=torch.int64)

    # Eager reference
    with torch.no_grad():
        eager_out = submodel(mel, mel_len_t)

    if isinstance(eager_out, tuple):
        eager_main = eager_out[0]
    else:
        eager_main = eager_out

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"  Eager output shape: {eager_main.shape}")
    print(f"  Eager range: [{eager_main.min():.4f}, {eager_main.max():.4f}]")

    # Export to TRT
    try:
        ep = torch.export.export(
            submodel,
            (mel,),
            kwargs={"length": mel_len_t},
        )
        print(f"  Export OK, {len(list(ep.graph.nodes))} nodes")

        # List ops
        from collections import Counter
        ops = Counter()
        for n in ep.graph.nodes:
            if n.op == "call_function":
                ops[str(n.target)] += 1
        print(f"  Top ops: {ops.most_common(5)}")

        # Convert to edge + TRT
        edge = to_edge(ep)
        partitioner = TensorRTPartitioner()
        edge = edge.to_backend(partitioner)

        et_prog = edge.to_executorch()

        # Save and reload
        tmp_path = f"/tmp/test_{name}.pte"
        with open(tmp_path, "wb") as f:
            f.write(et_prog.buffer)

        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(tmp_path)
        method = program.load_method("forward")

        et_out = method.execute([mel, mel_len_t])
        et_main = et_out[0]

        # Compare
        eager_np = eager_main.detach().numpy()
        et_np = et_main.detach().numpy()

        # Match shapes for comparison
        min_shape = [min(e, t) for e, t in zip(eager_np.shape, et_np.shape)]
        eager_np = eager_np[tuple(slice(0, s) for s in min_shape)]
        et_np = et_np[tuple(slice(0, s) for s in min_shape)]

        cos = np.dot(eager_np.flatten(), et_np.flatten()) / (
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(et_np.flatten()) + 1e-8
        )
        max_diff = np.abs(eager_np - et_np).max()
        mean_diff = np.abs(eager_np - et_np).mean()

        print(f"  TRT range: [{et_np.min():.4f}, {et_np.max():.4f}]")
        print(f"  Cosine: {cos:.8f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        os.unlink(tmp_path)
        return cos

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test with real speech mel
from export_parakeet_tdt import PreprocessorWrapper
import soundfile as sf

data, sr = sf.read("/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav", dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)
audio = torch.from_numpy(data).unsqueeze(0)
audio_1d = audio.squeeze(0)
audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

preproc = PreprocessorWrapper(model.preprocessor)
preproc.float().eval()

with torch.no_grad():
    mel, mel_len = preproc(audio_1d, audio_len)

mel_len_val = mel_len.item()
T_actual = mel.shape[2]
print(f"Mel: {mel.shape}, mel_len={mel_len_val}")

print("\n" + "=" * 60)
print("PROGRESSIVE ENCODER TRT ACCURACY TEST")
print("=" * 60)

results = []

# Stage 1: Subsampling only
cos = test_model_trt(SubsamplingOnly, "subsampling_only", encoder, mel, mel_len_val)
results.append(("Subsampling only", cos))

# Stage 2: Subsampling + positional encoding
cos = test_model_trt(SubsamplingPlusPosEnc, "subsamp_posenc", encoder, mel, mel_len_val)
results.append(("+ PosEnc", cos))

# Stage 3: + 1 conformer block
cos = test_model_trt(EncoderNBlocks, "1_block", encoder, mel, mel_len_val, n_blocks=1)
results.append(("+ 1 block", cos))

# Stage 4: + 4 blocks
cos = test_model_trt(EncoderNBlocks, "4_blocks", encoder, mel, mel_len_val, n_blocks=4)
results.append(("+ 4 blocks", cos))

# Stage 5: + 12 blocks
cos = test_model_trt(EncoderNBlocks, "12_blocks", encoder, mel, mel_len_val, n_blocks=12)
results.append(("+ 12 blocks", cos))

# Stage 6: all 24 blocks
cos = test_model_trt(EncoderNBlocks, "24_blocks", encoder, mel, mel_len_val, n_blocks=24)
results.append(("+ 24 blocks (full)", cos))

# Summary
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, cos in results:
    status = "PASS" if cos and cos > 0.99 else "WARN" if cos and cos > 0.9 else "FAIL"
    cos_str = f"{cos:.8f}" if cos else "ERROR"
    print(f"  {name:25s} cosine={cos_str}  [{status}]")
