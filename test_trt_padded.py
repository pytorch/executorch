#!/usr/bin/env python3
"""Test TRT engine directly at PADDED (5000) mel shape vs actual (744) mel shape.
Determines if the cosine issue is shape-dependent or content-dependent.
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
    """Extract the encoder TRT engine from a PTE file."""
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
    json_size = struct.unpack("<I", data[hdr_start+8:hdr_start+12])[0]
    total_meta = struct.unpack("<I", data[hdr_start+12:hdr_start+16])[0]
    engine_size = struct.unpack("<Q", data[hdr_start+16:hdr_start+24])[0]
    engine_start = hdr_start + total_meta
    engine_data = data[engine_start:engine_start + engine_size]
    return io_bindings, engine_data


def run_trt_engine(engine, context, eager_mel, mel_len_val, T_mel):
    """Run the TRT engine with given mel shape and return outputs."""
    gpu_tensors = {}

    # Compute shape tensor values for given T_mel
    sym_size = T_mel
    add_1 = (T_mel - 1) // 2 + 1
    add_2 = ((T_mel - 1) // 2) // 2 + 1
    add_3 = (((T_mel - 1) // 2) // 2) // 2 + 1
    sub_val = 4999 - (T_mel - 1) // 8
    sub_1 = (T_mel - 1) // 8 + 5000
    add_5 = 2 * add_3 - 1

    shape_tensor_vals = {
        "sym_size": sym_size, "add_1": add_1, "add_2": add_2,
        "add_3": add_3, "sub_1": sub_1, "add_5": add_5,
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        is_shape = engine.is_shape_inference_io(name)

        if mode == trt.TensorIOMode.INPUT:
            if is_shape:
                val = shape_tensor_vals.get(name, 0)
                host_buf = np.array([val], dtype=np.int32)
                gpu_tensors[name + "_host"] = host_buf
                context.set_tensor_address(name, host_buf.ctypes.data)
                context.set_input_shape(name, (1,))
            elif name == "audio_signal":
                context.set_input_shape(name, (1, 128, T_mel))
                t = eager_mel.cuda().contiguous()
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
            elif name == "length":
                context.set_input_shape(name, (1,))
                t = torch.tensor([mel_len_val], dtype=torch.int64, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
            elif name == "sub":
                context.set_input_shape(name, (1,))
                t = torch.tensor([sub_val], dtype=torch.int32, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
        else:
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            torch_dtype = torch.float32
            if dtype_trt == trt.int64:
                torch_dtype = torch.int64
            elif dtype_trt == trt.int32:
                torch_dtype = torch.int32
            out_size = [max(d, 1) for d in shape]
            t = torch.zeros(out_size, dtype=torch_dtype, device="cuda")
            context.set_tensor_address(name, t.data_ptr())
            gpu_tensors[name] = t

    stream = torch.cuda.current_stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    if not success:
        return None

    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            t = gpu_tensors[name].cpu()
            actual_size = [d for d in shape]
            t_np = t.numpy().flatten()[:int(np.prod(actual_size))].reshape(actual_size)
            outputs[name] = t_np
    return outputs


def compare_outputs(trt_f_proj, eager_f_proj_np, valid_frames, label):
    """Compare TRT vs eager encoder outputs."""
    trt_valid = trt_f_proj[0, :valid_frames, :]
    eager_valid = eager_f_proj_np[0, :valid_frames, :]

    cosines = []
    for f in range(valid_frames):
        e = eager_valid[f]
        t = trt_valid[f]
        cos = np.dot(e, t) / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-8)
        cosines.append(cos)

    cosines = np.array(cosines)
    overall = np.dot(eager_valid.flatten(), trt_valid.flatten()) / (
        np.linalg.norm(eager_valid.flatten()) * np.linalg.norm(trt_valid.flatten()) + 1e-8
    )
    max_diff = np.abs(eager_valid - trt_valid).max()

    print(f"\n  [{label}]")
    print(f"    Overall cosine: {overall:.8f}")
    print(f"    Per-frame: min={cosines.min():.6f}, max={cosines.max():.6f}, mean={cosines.mean():.6f}")
    print(f"    Max abs diff: {max_diff:.6f}")
    return overall


def main():
    dynamic_pte = "/home/gasoonjia/trt/executorch/parakeet_tdt_exports/model.pte"
    static_pte = "/home/gasoonjia/trt/executorch/parakeet_trt_static_fp32/model.pte"
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

    # Load model
    print("Loading Parakeet model...")
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

    audio = load_audio_sf(audio_path)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)
    actual_mel_len = eager_mel_len.item()
    T_actual = eager_mel.shape[2]  # 744

    print(f"Mel: {eager_mel.shape}, mel_len={actual_mel_len}")

    # ========== Test 1: Dynamic TRT at actual shape (744) ==========
    print("\n" + "=" * 60)
    print("TEST 1: Dynamic TRT at actual shape (T_mel=744)")
    print("=" * 60)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    io_meta, engine_data = extract_encoder_engine(dynamic_pte)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Eager reference at actual shape
    mel_len_t = torch.tensor([actual_mel_len], dtype=torch.int64)
    with torch.no_grad():
        eager_f_proj_actual, eager_enc_len_actual = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_t
        )
    valid_actual = eager_enc_len_actual.item()
    print(f"Eager: f_proj={eager_f_proj_actual.shape}, enc_len={valid_actual}")

    outputs = run_trt_engine(engine, context, eager_mel, actual_mel_len, T_actual)
    if outputs:
        f_proj_key = [k for k in outputs if "view_copy" in k or "f_proj" in k]
        if f_proj_key:
            cos1 = compare_outputs(
                outputs[f_proj_key[0]],
                eager_f_proj_actual.numpy(), valid_actual,
                "Dynamic@744 vs Eager@744"
            )
        else:
            print(f"  Output keys: {list(outputs.keys())}")
    else:
        print("  FAILED to execute!")
        cos1 = 0

    # ========== Test 2: Dynamic TRT at padded shape (5000) ==========
    print("\n" + "=" * 60)
    print("TEST 2: Dynamic TRT at padded shape (T_mel=5000)")
    print("=" * 60)

    T_padded = 5000
    mel_padded = torch.zeros(1, 128, T_padded, dtype=torch.float32)
    mel_padded[:, :, :T_actual] = eager_mel

    # Eager reference at padded shape
    with torch.no_grad():
        eager_f_proj_padded, eager_enc_len_padded = encoder_with_proj(
            audio_signal=mel_padded, length=mel_len_t
        )
    valid_padded = eager_enc_len_padded.item()
    print(f"Eager: f_proj={eager_f_proj_padded.shape}, enc_len={valid_padded}")

    # Need fresh context for different shape
    context2 = engine.create_execution_context()
    outputs2 = run_trt_engine(engine, context2, mel_padded, actual_mel_len, T_padded)
    if outputs2:
        f_proj_key = [k for k in outputs2 if "view_copy" in k or "f_proj" in k]
        if f_proj_key:
            cos2 = compare_outputs(
                outputs2[f_proj_key[0]],
                eager_f_proj_padded.numpy(), valid_padded,
                "Dynamic@5000 vs Eager@5000"
            )
        else:
            print(f"  Output keys: {list(outputs2.keys())}")
    else:
        print("  FAILED to execute!")
        cos2 = 0

    # ========== Test 3: Static TRT at padded shape (5000) ==========
    print("\n" + "=" * 60)
    print("TEST 3: Static TRT at padded shape (T_mel=5000)")
    print("=" * 60)

    try:
        io_meta_s, engine_data_s = extract_encoder_engine(static_pte)
        engine_s = runtime.deserialize_cuda_engine(engine_data_s)
        context_s = engine_s.create_execution_context()

        # Static engine has no shape tensors - just set regular inputs
        gpu_tensors_s = {}
        for i in range(engine_s.num_io_tensors):
            name = engine_s.get_tensor_name(i)
            mode = engine_s.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if name == "audio_signal":
                    context_s.set_input_shape(name, (1, 128, T_padded))
                    t = mel_padded.cuda().contiguous()
                    context_s.set_tensor_address(name, t.data_ptr())
                    gpu_tensors_s[name] = t
                elif name == "length":
                    context_s.set_input_shape(name, (1,))
                    t = torch.tensor([actual_mel_len], dtype=torch.int64, device="cuda")
                    context_s.set_tensor_address(name, t.data_ptr())
                    gpu_tensors_s[name] = t
                else:
                    # Any other inputs
                    shape = engine_s.get_tensor_shape(name)
                    dtype_trt = engine_s.get_tensor_dtype(name)
                    torch_dtype = torch.int32 if dtype_trt == trt.int32 else torch.float32
                    if dtype_trt == trt.int64:
                        torch_dtype = torch.int64
                    t = torch.zeros([max(d,1) for d in shape], dtype=torch_dtype, device="cuda")
                    context_s.set_tensor_address(name, t.data_ptr())
                    gpu_tensors_s[name] = t
                    print(f"  Extra input '{name}': shape={shape}")
            else:
                shape = context_s.get_tensor_shape(name)
                dtype_trt = engine_s.get_tensor_dtype(name)
                torch_dtype = torch.float32
                if dtype_trt == trt.int64:
                    torch_dtype = torch.int64
                elif dtype_trt == trt.int32:
                    torch_dtype = torch.int32
                out_size = [max(d, 1) for d in shape]
                t = torch.zeros(out_size, dtype=torch_dtype, device="cuda")
                context_s.set_tensor_address(name, t.data_ptr())
                gpu_tensors_s[name] = t

        stream = torch.cuda.current_stream()
        success = context_s.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        print(f"Static engine execution: {'OK' if success else 'FAILED'}")

        if success:
            outputs_s = {}
            for i in range(engine_s.num_io_tensors):
                name = engine_s.get_tensor_name(i)
                mode = engine_s.get_tensor_mode(name)
                if mode == trt.TensorIOMode.OUTPUT:
                    shape = context_s.get_tensor_shape(name)
                    t = gpu_tensors_s[name].cpu()
                    actual_size = [d for d in shape]
                    t_np = t.numpy().flatten()[:int(np.prod(actual_size))].reshape(actual_size)
                    outputs_s[name] = t_np

            f_proj_key = [k for k in outputs_s if "view_copy" in k or "f_proj" in k]
            if f_proj_key:
                cos3 = compare_outputs(
                    outputs_s[f_proj_key[0]],
                    eager_f_proj_padded.numpy(), valid_padded,
                    "Static@5000 vs Eager@5000"
                )
            else:
                print(f"  Output keys: {list(outputs_s.keys())}")
                cos3 = 0
        else:
            cos3 = 0
    except Exception as e:
        print(f"  Static test failed: {e}")
        cos3 = 0

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dynamic@744  vs Eager@744:  cosine = {cos1:.6f}")
    print(f"  Dynamic@5000 vs Eager@5000: cosine = {cos2:.6f}")
    print(f"  Static@5000  vs Eager@5000: cosine = {cos3:.6f}")

    if cos2 > 0.99 and cos1 < 0.9:
        print("\n=> Issue is SHAPE-DEPENDENT: engine works at max shape but not at smaller shapes.")
        print("   Root cause likely in a converter that doesn't handle dynamic dims correctly.")
    elif cos2 < 0.9 and cos3 < 0.9:
        print("\n=> Issue affects BOTH static and dynamic engines. Pre-existing converter bug.")
    elif cos2 < 0.9 and cos3 > 0.99:
        print("\n=> Issue is specific to the DYNAMIC engine build. Dynamic export changes broke something.")
    else:
        print(f"\n=> Mixed results. Further investigation needed.")


if __name__ == "__main__":
    main()
