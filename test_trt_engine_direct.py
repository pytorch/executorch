#!/usr/bin/env python3
"""Test TRT engine directly via TRT Python API, bypassing ET backend."""
import json
import struct
import sys

import numpy as np
import soundfile as sf
import tensorrt as trt
import torch

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import EncoderWithProjection, PreprocessorWrapper

def extract_trt_engine_from_pte(pte_path, subgraph_idx=1):
    """Extract the TRT engine blob from a PTE file.

    PTE blob format: 8-byte header_size (LE), header JSON, engine bytes
    Each delegate blob starts with this pattern.
    """
    with open(pte_path, "rb") as f:
        data = f.read()

    # Find all TRT blobs - they start with a JSON header containing io_bindings
    # The blob structure is: [8-byte header_size][header JSON][engine bytes]
    # We need to find the encoder subgraph (subgraph_idx)

    # Look for the TRT blob header magic/signature
    # The blob header contains: header_size(8 bytes) + JSON header + engine
    # JSON header has "io_bindings" key

    # Find all occurrences of "io_bindings" in the binary
    positions = []
    pos = 0
    while True:
        pos = data.find(b'"io_bindings"', pos)
        if pos < 0:
            break
        positions.append(pos)
        pos += 1

    print(f"Found {len(positions)} io_bindings sections")

    # For each position, find the blob start (8 bytes before the JSON start)
    for idx, io_pos in enumerate(positions):
        # Find the start of the JSON object
        json_start = data.rfind(b'{', max(0, io_pos - 2000), io_pos)
        if json_start < 0:
            continue

        # The 8 bytes before json_start should be the header size
        if json_start < 8:
            continue
        header_size = struct.unpack('<Q', data[json_start - 8:json_start])[0]

        # Validate: header_size should match the JSON region
        if header_size > 50000 or header_size < 10:
            continue

        # Parse the JSON header
        try:
            json_data = data[json_start:json_start + header_size].decode('utf-8')
            header = json.loads(json_data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        bindings = header.get('io_bindings', [])
        inputs = [b for b in bindings if b['is_input']]
        print(f"\nSubgraph {idx}: {len(inputs)} inputs, {len(bindings) - len(inputs)} outputs")
        for b in bindings:
            shape_flag = ' [SHAPE]' if b.get('is_shape_tensor') else ''
            io = 'IN' if b['is_input'] else 'OUT'
            print(f"  {io} {b['name']}: {b['dtype']} {b['shape']}{shape_flag}")

        if idx == subgraph_idx:
            # Extract engine bytes (everything after the header)
            engine_start = json_start + header_size
            # Need to find the engine size - look at the full blob
            # The blob starts 8 bytes before json_start
            blob_start = json_start - 8

            # Find the engine bytes - they follow immediately after the header
            # The engine is a valid TRT serialized engine
            # We need to find where it ends - look for TRT magic or use the
            # next blob start

            # For simplicity, extract a large chunk and let TRT validate
            remaining = data[engine_start:]

            # TRT engines start with specific bytes - find the engine
            # Actually, the engine should start right after the header
            print(f"\nEngine starts at offset {engine_start}")
            print(f"First 16 bytes: {remaining[:16].hex()}")

            return remaining, header

    return None, None


def main():
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

    # Load eager model for comparison
    print("Loading eager model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()

    # Load audio and get mel
    data, sr = sf.read(audio_path, dtype="float32")
    audio = torch.from_numpy(data).unsqueeze(0)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.float().eval()
    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)

    mel_len_tensor = torch.tensor([eager_mel_len.item()], dtype=torch.int64)
    mel_len_val = eager_mel_len.item()

    # Get eager encoder output
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_tensor
        )
    print(f"Eager encoder: {eager_f_proj.shape}, range=[{eager_f_proj.min():.6f}, {eager_f_proj.max():.6f}]")

    # Now test via ET backend (same as compare script)
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program("/home/gasoonjia/trt/executorch/parakeet_trt_dynamic_fp32/model.pte")
    encoder_method = program.load_method("encoder")
    et_enc_result = encoder_method.execute([eager_mel, mel_len_tensor])
    et_f_proj = et_enc_result[0]
    et_enc_len = et_enc_result[1]
    print(f"ET encoder: {et_f_proj.shape}, range=[{et_f_proj.min():.6f}, {et_f_proj.max():.6f}]")

    valid = min(eager_enc_len.item(), et_enc_len.item())
    a = eager_f_proj[:, :valid, :].detach().cpu().float().numpy().flatten()
    b = et_f_proj[:, :valid, :].detach().cpu().float().numpy().flatten()
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Cosine: {cos:.8f}")

    # Print first few values for comparison
    print(f"\nEager first 10 values: {a[:10]}")
    print(f"ET    first 10 values: {b[:10]}")
    print(f"\nEager frame 0 stats: min={a[:640].min():.6f} max={a[:640].max():.6f} mean={a[:640].mean():.6f}")
    print(f"ET    frame 0 stats: min={b[:640].min():.6f} max={b[:640].max():.6f} mean={b[:640].mean():.6f}")

    # Per-frame cosine
    print("\nPer-frame cosine similarity:")
    for t in range(min(valid, 10)):
        frame_a = a[t*640:(t+1)*640]
        frame_b = b[t*640:(t+1)*640]
        frame_cos = np.dot(frame_a, frame_b) / (np.linalg.norm(frame_a) * np.linalg.norm(frame_b) + 1e-8)
        print(f"  Frame {t}: cosine={frame_cos:.6f} abs_diff_max={np.abs(frame_a-frame_b).max():.6f}")

    # Check if there's a pattern - maybe the outputs are shifted
    print("\nChecking if outputs are shifted/reordered...")
    best_cos = -1
    best_shift = 0
    for shift in range(-5, 6):
        if 0 <= shift < valid and shift + valid <= len(b) // 640:
            shifted_b = b[shift*640:(shift+valid)*640]
            if len(shifted_b) >= len(a):
                shifted_b = shifted_b[:len(a)]
                cos_shifted = np.dot(a, shifted_b) / (np.linalg.norm(a) * np.linalg.norm(shifted_b) + 1e-8)
                if cos_shifted > best_cos:
                    best_cos = cos_shifted
                    best_shift = shift
    print(f"  Best shift: {best_shift} with cosine: {best_cos:.8f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
