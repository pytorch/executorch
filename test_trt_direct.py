#!/usr/bin/env python3
"""Test TRT engine directly via Python API, bypassing ExecuTorch.
This isolates whether the accuracy issue is in the TRT engine itself
or in ExecuTorch's data handling.
"""
import sys
import struct
import json
import ctypes
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

    # Find TRT blob with audio_signal input (encoder)
    pattern = b'"audio_signal"'
    idx = data.find(pattern)
    if idx == -1:
        raise RuntimeError("Encoder blob not found")

    # Find the JSON start
    json_start = data.rfind(b'{"io_bindings"', max(0, idx - 5000), idx + 1)
    if json_start == -1:
        raise RuntimeError("JSON start not found")

    # Parse JSON
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

    # Find TR01 header
    hdr_start = data.rfind(b"TR01", max(0, json_start - 100), json_start)
    if hdr_start == -1:
        raise RuntimeError("TR01 header not found")

    # Parse: magic(4) + reserved(4) + json_size(4) + total_meta(4) + engine_size(8)
    json_size = struct.unpack("<I", data[hdr_start+8:hdr_start+12])[0]
    total_meta = struct.unpack("<I", data[hdr_start+12:hdr_start+16])[0]
    engine_size = struct.unpack("<Q", data[hdr_start+16:hdr_start+24])[0]

    engine_start = hdr_start + total_meta
    engine_data = data[engine_start:engine_start + engine_size]
    print(f"Engine: {len(engine_data)} bytes, json_size={json_size}, total_meta={total_meta}")

    return io_bindings, engine_data


def main():
    pte_path = "/home/gasoonjia/trt/executorch/parakeet_trt_dynamic_fp32/model.pte"
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

    print("Extracting TRT engine from PTE...")
    io_meta, engine_data = extract_encoder_engine(pte_path)

    # Deserialize engine
    print("Deserializing TRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        print("Failed to deserialize engine!")
        return
    context = engine.create_execution_context()
    print(f"Engine has {engine.num_io_tensors} IO tensors")

    # List all IO tensors
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        is_shape = engine.is_shape_inference_io(name)
        print(f"  [{i}] {'IN' if mode == trt.TensorIOMode.INPUT else 'OUT'} "
              f"{'(shape) ' if is_shape else ''}"
              f"{name}: shape={shape}, dtype={dtype}")

    # Load model and prepare inputs
    print("\nLoading Parakeet model...")
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
    mel_len_tensor = torch.tensor([eager_mel_len.item()], dtype=torch.int64)
    T_mel = eager_mel.shape[2]

    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_tensor
        )
    valid = eager_enc_len.item()
    print(f"\nMel: {eager_mel.shape}, mel_len={eager_mel_len.item()}")
    print(f"Eager: f_proj={eager_f_proj.shape}, enc_len={valid}")

    # Compute shape tensor values
    sym_size = T_mel
    add_1 = (T_mel - 1) // 2 + 1
    add_2 = ((T_mel - 1) // 2) // 2 + 1
    add_3 = (((T_mel - 1) // 2) // 2) // 2 + 1
    sub_val = 4999 - (T_mel - 1) // 8
    sub_1 = (T_mel - 1) // 8 + 5000
    add_5 = 2 * add_3 - 1

    shape_tensor_vals = {
        "sym_size": sym_size,
        "add_1": add_1,
        "add_2": add_2,
        "add_3": add_3,
        "sub_1": sub_1,
        "add_5": add_5,
    }

    print(f"\nSetting up TRT inputs:")

    # Use torch CUDA tensors for GPU memory management
    gpu_tensors = {}  # Keep references alive

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        is_shape = engine.is_shape_inference_io(name)

        if mode == trt.TensorIOMode.INPUT:
            if is_shape:
                val = shape_tensor_vals.get(name, 0)
                # Shape tensors: value is read from host memory at tensor address
                host_buf = np.array([val], dtype=np.int32)
                gpu_tensors[name + "_host"] = host_buf  # keep alive
                context.set_tensor_address(name, host_buf.ctypes.data)
                context.set_input_shape(name, (1,))
                print(f"  Shape tensor '{name}' = {val}")
            elif name == "audio_signal":
                context.set_input_shape(name, (1, 128, T_mel))
                t = eager_mel.cuda().contiguous()
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
                print(f"  Input '{name}' shape=(1, 128, {T_mel})")
            elif name == "length":
                context.set_input_shape(name, (1,))
                t = mel_len_tensor.cuda().contiguous()
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
                print(f"  Input '{name}' = {eager_mel_len.item()}")
            elif name == "sub":
                context.set_input_shape(name, (1,))
                t = torch.tensor([sub_val], dtype=torch.int32, device="cuda")
                context.set_tensor_address(name, t.data_ptr())
                gpu_tensors[name] = t
                print(f"  Input '{name}' = {sub_val} (device tensor)")
        else:
            # Output
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            print(f"  Output '{name}' inferred shape={shape}, dtype={dtype_trt}")

            torch_dtype = torch.float32
            if dtype_trt == trt.int64:
                torch_dtype = torch.int64
            elif dtype_trt == trt.int32:
                torch_dtype = torch.int32

            out_size = [max(d, 1) for d in shape]
            t = torch.zeros(out_size, dtype=torch_dtype, device="cuda")
            context.set_tensor_address(name, t.data_ptr())
            gpu_tensors[name] = t

    # Execute
    print("\nExecuting...")
    stream = torch.cuda.current_stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    print(f"Success: {success}")

    if not success:
        print("FAILED!")
        return

    # Get outputs
    print("\nOutputs:")
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            t = gpu_tensors[name].cpu()
            # Reshape to actual inferred shape
            actual_size = [d for d in shape]
            t_np = t.numpy().flatten()[:int(np.prod(actual_size))].reshape(actual_size)
            outputs[name] = t_np
            print(f"  {name}: shape={shape}, range=[{t_np.min():.4f}, {t_np.max():.4f}]")

    # Compare
    trt_f_proj = outputs.get("output_aten_view_copy_default_820")
    if trt_f_proj is None:
        print("Could not find f_proj output!")
        return

    print(f"\nTRT f_proj: {trt_f_proj.shape}")
    print(f"Eager f_proj: {eager_f_proj.shape}")

    trt_valid = trt_f_proj[0, :valid, :]
    eager_valid = eager_f_proj[0, :valid, :].numpy()

    # Per-frame cosine
    print(f"\nPer-frame cosine (first 5, last 3 of {valid}):")
    cosines = []
    for f in range(valid):
        e = eager_valid[f]
        t = trt_valid[f]
        cos = np.dot(e, t) / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-8)
        cosines.append(cos)
        if f < 5 or f >= valid - 3:
            diff = np.abs(e - t)
            print(f"  frame {f:3d}: cos={cos:.6f}, max_diff={diff.max():.4f}")
        elif f == 5:
            print("  ...")

    cosines = np.array(cosines)
    overall = np.dot(eager_valid.flatten(), trt_valid.flatten()) / (
        np.linalg.norm(eager_valid.flatten()) * np.linalg.norm(trt_valid.flatten()) + 1e-8
    )
    print(f"\nOverall cosine (TRT direct vs eager CPU): {overall:.8f}")
    print(f"Per-frame: min={cosines.min():.6f}, max={cosines.max():.6f}, mean={cosines.mean():.6f}")

    if overall > 0.99:
        print("\n=> TRT engine is accurate. Issue is in ExecuTorch data handling.")
    elif overall > 0.9:
        print("\n=> Moderate precision loss in TRT engine.")
    else:
        print("\n=> TRT engine itself produces wrong results. Issue in converters.")


if __name__ == "__main__":
    main()
