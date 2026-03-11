#!/usr/bin/env python3
"""Quick cosine test for static TRT engine with test_audio.wav vs real_speech.wav.
Verifies if the converter accuracy issue is audio-content-dependent.
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
    if idx == -1:
        raise RuntimeError("Encoder blob not found")
    json_start = data.rfind(b'{"io_bindings"', max(0, idx - 5000), idx + 1)
    if json_start == -1:
        raise RuntimeError("JSON start not found")
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
    io_bindings = json.loads(json_str)
    hdr_start = data.rfind(b"TR01", max(0, json_start - 100), json_start)
    if hdr_start == -1:
        raise RuntimeError("TR01 header not found")
    total_meta = struct.unpack("<I", data[hdr_start+12:hdr_start+16])[0]
    engine_size = struct.unpack("<Q", data[hdr_start+16:hdr_start+24])[0]
    engine_start = hdr_start + total_meta
    engine_data = data[engine_start:engine_start + engine_size]
    return io_bindings, engine_data


def main():
    static_pte = "/home/gasoonjia/trt/executorch/parakeet_trt_static_fp32/model.pte"
    audio_files = [
        "/home/gasoonjia/trt/executorch/examples/models/parakeet/test_audio.wav",
        "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav",
    ]

    print("Loading model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()

    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.float()
    preprocessor_wrapper.eval()

    print("Loading static TRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    _, engine_data = extract_encoder_engine(static_pte)
    engine = runtime.deserialize_cuda_engine(engine_data)

    # List IO tensors once
    print(f"Engine has {engine.num_io_tensors} IO tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        is_shape = engine.is_shape_inference_io(name)
        print(f"  [{i}] {'IN' if mode == trt.TensorIOMode.INPUT else 'OUT'} "
              f"{'(shape) ' if is_shape else ''}{name}: {shape}")

    T_padded = 5000

    for audio_path in audio_files:
        fname = audio_path.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Testing: {fname}")
        print(f"{'='*60}")

        audio = load_audio_sf(audio_path)
        audio_1d = audio.squeeze(0)
        audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
        print(f"  Audio: {audio_1d.shape[0]} samples ({audio_1d.shape[0]/16000:.2f}s)")

        with torch.no_grad():
            mel, mel_len = preprocessor_wrapper(audio_1d, audio_len)
        T_actual = mel.shape[2]
        actual_mel_len = mel_len.item()
        print(f"  Mel: [1, 128, {T_actual}], mel_len={actual_mel_len}")

        # Pad
        mel_padded = torch.zeros(1, 128, T_padded, dtype=torch.float32)
        mel_padded[:, :, :T_actual] = mel
        mel_len_t = torch.tensor([actual_mel_len], dtype=torch.int64)

        # Eager
        with torch.no_grad():
            eager_f, eager_len = encoder_with_proj(
                audio_signal=mel_padded, length=mel_len_t
            )
        valid = eager_len.item()
        print(f"  Eager: f_proj={eager_f.shape}, enc_len={valid}")

        # TRT
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
                    t = mel_len_t.cuda().contiguous()
                    context.set_tensor_address(name, t.data_ptr())
                    gpu_tensors[name] = t
                else:
                    shape = engine.get_tensor_shape(name)
                    dtype_trt = engine.get_tensor_dtype(name)
                    td = torch.int32 if dtype_trt == trt.int32 else torch.float32
                    if dtype_trt == trt.int64:
                        td = torch.int64
                    t = torch.zeros([max(d,1) for d in shape], dtype=td, device="cuda")
                    context.set_tensor_address(name, t.data_ptr())
                    gpu_tensors[name] = t
            else:
                shape = context.get_tensor_shape(name)
                dtype_trt = engine.get_tensor_dtype(name)
                td = torch.float32
                if dtype_trt == trt.int64:
                    td = torch.int64
                elif dtype_trt == trt.int32:
                    td = torch.int32
                t = torch.zeros([max(d,1) for d in shape], dtype=td, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t

        stream = torch.cuda.current_stream()
        ok = context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        if not ok:
            print("  FAILED!")
            continue

        # Get f_proj output
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT and "view_copy" in name:
                shape = context.get_tensor_shape(name)
                t_np = gpu_tensors[name].cpu().numpy()
                actual_size = [d for d in shape]
                trt_f = t_np.flatten()[:int(np.prod(actual_size))].reshape(actual_size)

                trt_valid = trt_f[0, :valid, :]
                eager_valid = eager_f[0, :valid, :].numpy()

                overall = np.dot(eager_valid.flatten(), trt_valid.flatten()) / (
                    np.linalg.norm(eager_valid.flatten()) * np.linalg.norm(trt_valid.flatten()) + 1e-8
                )
                max_diff = np.abs(eager_valid - trt_valid).max()

                # Per-frame stats
                cosines = []
                for fr in range(valid):
                    e = eager_valid[fr]
                    t = trt_valid[fr]
                    cos = np.dot(e, t) / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-8)
                    cosines.append(cos)
                cosines = np.array(cosines)

                print(f"  Overall cosine: {overall:.8f}")
                print(f"  Per-frame: min={cosines.min():.6f}, max={cosines.max():.6f}, mean={cosines.mean():.6f}")
                print(f"  Max abs diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
