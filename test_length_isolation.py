#!/usr/bin/env python3
"""Isolate whether the cosine issue is caused by mel_len or audio content.
Test 1: real_speech.wav with its own mel_len (743) → expect 0.66
Test 2: real_speech.wav with test_audio's mel_len (500) → if 0.97, it's length-dependent
Test 3: test_audio.wav with real_speech's mel_len (743) → if 0.66, it's length-dependent
Test 4: real_speech.wav with mel_len=0 (no masking?) → see what happens
"""
import sys
import struct
import json
import numpy as np
import torch
import torch.cuda
import soundfile as sf
import tensorrt as trt

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import EncoderWithProjection, PreprocessorWrapper


def load_audio_sf(audio_path, target_sr=16000):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return torch.from_numpy(data).unsqueeze(0)


def extract_encoder_engine(pte_path):
    with open(pte_path, "rb") as f:
        data = f.read()
    pattern = b'"audio_signal"'
    idx = data.find(pattern)
    json_start = data.rfind(b'{"io_bindings"', max(0, idx - 5000), idx + 1)
    depth = 0
    end = json_start
    for i in range(json_start, min(json_start + 50000, len(data))):
        if data[i:i+1] == b'{':
            depth += 1
        elif data[i:i+1] == b'}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    json_str = data[json_start:end].decode("utf-8")
    hdr_start = data.rfind(b"TR01", max(0, json_start - 100), json_start)
    total_meta = struct.unpack("<I", data[hdr_start+12:hdr_start+16])[0]
    engine_size = struct.unpack("<Q", data[hdr_start+16:hdr_start+24])[0]
    engine_start = hdr_start + total_meta
    return data[engine_start:engine_start + engine_size]


def run_engine(engine, mel_padded, mel_len_val, T_padded=5000):
    context = engine.create_execution_context()
    gpu_tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            if name == "audio_signal":
                context.set_input_shape(name, (1, 128, T_padded))
                t = mel_padded.cuda().contiguous()
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
            elif name == "length":
                context.set_input_shape(name, (1,))
                t = torch.tensor([mel_len_val], dtype=torch.int64, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
            else:
                shape = engine.get_tensor_shape(name)
                dtype_trt = engine.get_tensor_dtype(name)
                td = torch.int32 if dtype_trt == trt.int32 else torch.float32
                if dtype_trt == trt.int64: td = torch.int64
                t = torch.zeros([max(d,1) for d in shape], dtype=td, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
        else:
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            td = torch.float32
            if dtype_trt == trt.int64: td = torch.int64
            elif dtype_trt == trt.int32: td = torch.int32
            t = torch.zeros([max(d,1) for d in shape], dtype=td, device="cuda")
            context.set_tensor_address(name, t.data_ptr())
            gpu_tensors[name] = t

    stream = torch.cuda.current_stream()
    ok = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    if not ok:
        return None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT and "view_copy" in name:
            shape = context.get_tensor_shape(name)
            t_np = gpu_tensors[name].cpu().numpy()
            actual_size = [d for d in shape]
            return t_np.flatten()[:int(np.prod(actual_size))].reshape(actual_size)
    return None


def cosine(a, b):
    a_f = a.flatten()
    b_f = b.flatten()
    return np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-8)


def main():
    static_pte = "/home/gasoonjia/trt/executorch/parakeet_trt_static_fp32/model.pte"

    print("Loading model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    enc_proj = EncoderWithProjection(model.encoder, model.joint)
    enc_proj.eval()
    preproc = PreprocessorWrapper(model.preprocessor)
    preproc.float().eval()

    # Prepare both audio files
    real_audio = load_audio_sf("/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav")
    test_audio = load_audio_sf("/home/gasoonjia/trt/executorch/examples/models/parakeet/test_audio.wav")

    with torch.no_grad():
        real_mel, real_mel_len = preproc(real_audio.squeeze(0), torch.tensor([real_audio.shape[1]], dtype=torch.int64))
        test_mel, test_mel_len = preproc(test_audio.squeeze(0), torch.tensor([test_audio.shape[1]], dtype=torch.int64))

    real_mel_val = real_mel_len.item()  # 743
    test_mel_val = test_mel_len.item()  # 500
    print(f"real_speech mel_len={real_mel_val}, test_audio mel_len={test_mel_val}")

    # Pad both to 5000
    T_padded = 5000
    real_padded = torch.zeros(1, 128, T_padded, dtype=torch.float32)
    real_padded[:, :, :real_mel.shape[2]] = real_mel
    test_padded = torch.zeros(1, 128, T_padded, dtype=torch.float32)
    test_padded[:, :, :test_mel.shape[2]] = test_mel

    # Load engine
    engine_data = extract_encoder_engine(static_pte)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    tests = [
        # (description, mel_tensor, mel_len_for_trt, mel_len_for_eager)
        ("real_speech + len=743 (baseline)", real_padded, real_mel_val, real_mel_val),
        ("real_speech + len=500 (swapped len)", real_padded, test_mel_val, test_mel_val),
        ("test_audio + len=500 (baseline)", test_padded, test_mel_val, test_mel_val),
        ("test_audio + len=743 (swapped len)", test_padded, real_mel_val, real_mel_val),
        ("real_speech + len=743 but TRT gets len=500", real_padded, test_mel_val, real_mel_val),
    ]

    print(f"\n{'='*70}")
    for desc, mel, trt_len, eager_len in tests:
        print(f"\nTest: {desc}")
        print(f"  TRT mel_len={trt_len}, Eager mel_len={eager_len}")

        # TRT
        trt_out = run_engine(engine, mel, trt_len)
        if trt_out is None:
            print("  TRT FAILED!")
            continue

        # Eager reference
        mel_len_t = torch.tensor([eager_len], dtype=torch.int64)
        with torch.no_grad():
            eager_out, eager_len_out = enc_proj(audio_signal=mel, length=mel_len_t)
        valid = eager_len_out.item()

        trt_valid = trt_out[0, :valid, :]
        eager_valid = eager_out[0, :valid, :].numpy()
        cos = cosine(trt_valid, eager_valid)
        max_diff = np.abs(trt_valid - eager_valid).max()
        print(f"  Valid frames: {valid}")
        print(f"  Cosine: {cos:.8f}, Max diff: {max_diff:.6f}")

    # Extra test: does TRT output change with different mel_len?
    print(f"\n{'='*70}")
    print("CRITICAL: Does TRT output change when mel_len changes?")
    print("(Same audio, different mel_len values)")
    trt_a = run_engine(engine, real_padded, 743)
    trt_b = run_engine(engine, real_padded, 500)
    trt_c = run_engine(engine, real_padded, 100)
    if trt_a is not None and trt_b is not None and trt_c is not None:
        cos_ab = cosine(trt_a[0, :93, :], trt_b[0, :93, :])
        cos_ac = cosine(trt_a[0, :93, :], trt_c[0, :93, :])
        max_diff_ab = np.abs(trt_a[0, :93, :] - trt_b[0, :93, :]).max()
        print(f"  TRT(real,len=743) vs TRT(real,len=500): cosine={cos_ab:.8f}, max_diff={max_diff_ab:.6f}")
        print(f"  TRT(real,len=743) vs TRT(real,len=100): cosine={cos_ac:.8f}")
        if cos_ab > 0.9999:
            print("  => mel_len has NO EFFECT on TRT output! Length masking may be broken in TRT.")
        else:
            print("  => mel_len DOES affect TRT output. Masking works in TRT.")


if __name__ == "__main__":
    main()
