#!/usr/bin/env python3
"""Quick diagnostic: check encoder output ordering and data quality."""
import sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import EncoderWithProjection, PreprocessorWrapper

from executorch.runtime import Runtime


def load_audio_sf(audio_path, target_sr=16000):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return torch.from_numpy(data).unsqueeze(0)


def main():
    pte_path = "/home/gasoonjia/trt/executorch/parakeet_trt_dynamic_fp32/model.pte"
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

    print("Loading PTE...")
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)

    print("Loading eager model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    model.cpu()

    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.float()
    preprocessor_wrapper.eval()

    # Load audio and preprocess
    audio = load_audio_sf(audio_path)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)
    print(f"Mel: {eager_mel.shape}, mel_len={eager_mel_len.item()}")

    mel_len_tensor = torch.tensor([eager_mel_len.item()], dtype=torch.int64)

    # Eager encoder
    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_tensor
        )
    print(f"\nEager encoder:")
    print(f"  f_proj: shape={eager_f_proj.shape}, dtype={eager_f_proj.dtype}")
    print(f"  f_proj range: [{eager_f_proj.min():.4f}, {eager_f_proj.max():.4f}]")
    print(f"  enc_len: shape={eager_enc_len.shape}, value={eager_enc_len.item()}")

    # ET encoder
    encoder_method = program.load_method("encoder")
    et_enc_result = encoder_method.execute([eager_mel, mel_len_tensor])

    print(f"\nET encoder: {len(et_enc_result)} outputs")
    for idx, out in enumerate(et_enc_result):
        if isinstance(out, torch.Tensor):
            print(f"  output[{idx}]: shape={out.shape}, dtype={out.dtype}, "
                  f"range=[{out.float().min():.4f}, {out.float().max():.4f}], "
                  f"mean={out.float().mean():.4f}")
        else:
            print(f"  output[{idx}]: type={type(out)}, value={out}")

    # Check which output looks like f_proj vs enc_len
    for idx, out in enumerate(et_enc_result):
        if isinstance(out, torch.Tensor):
            if out.numel() == 1:
                print(f"\n  output[{idx}] is scalar-like: {out.item()} (likely enc_len)")
            elif out.dim() == 3:
                print(f"\n  output[{idx}] is 3D tensor: {out.shape} (likely f_proj)")

    # Compare with correct assignment
    et_f_proj = et_enc_result[0]
    et_enc_len = et_enc_result[1]

    print(f"\nAssignment check:")
    print(f"  et_f_proj = et_enc_result[0]: shape={et_f_proj.shape}")
    print(f"  et_enc_len = et_enc_result[1]: shape={et_enc_len.shape}")

    if et_f_proj.dim() == 3 and et_enc_len.numel() == 1:
        print("  Output ordering: CORRECT (f_proj first, enc_len second)")
    elif et_f_proj.numel() == 1 and et_enc_len.dim() == 3:
        print("  Output ordering: SWAPPED! (enc_len first, f_proj second)")
        # Swap
        et_f_proj, et_enc_len = et_enc_len, et_f_proj

    valid_frames = min(eager_enc_len.item(), et_enc_len.item()
                       if et_enc_len.numel() == 1 else int(et_enc_len.shape[1]))

    # Per-frame cosine analysis
    eager_np = eager_f_proj[0, :valid_frames, :].float().numpy()
    et_np = et_f_proj[0, :valid_frames, :].float().numpy()

    print(f"\nPer-frame cosine (first 10 and last 5 of {valid_frames} frames):")
    cosines = []
    for f in range(valid_frames):
        e = eager_np[f]
        t = et_np[f]
        cos = np.dot(e, t) / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-8)
        cosines.append(cos)
        if f < 10 or f >= valid_frames - 5:
            # Also check absolute diff stats
            diff = np.abs(e - t)
            print(f"  frame {f:3d}: cos={cos:.6f}, "
                  f"eager_norm={np.linalg.norm(e):.4f}, "
                  f"et_norm={np.linalg.norm(t):.4f}, "
                  f"max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
        elif f == 10:
            print("  ...")

    cosines = np.array(cosines)
    print(f"\nCosine summary over {valid_frames} frames:")
    print(f"  min={cosines.min():.6f}, max={cosines.max():.6f}, "
          f"mean={cosines.mean():.6f}, std={cosines.std():.6f}")

    # Overall cosine
    eager_flat = eager_np.flatten()
    et_flat = et_np.flatten()
    overall_cos = np.dot(eager_flat, et_flat) / (
        np.linalg.norm(eager_flat) * np.linalg.norm(et_flat) + 1e-8
    )
    print(f"  overall cosine: {overall_cos:.8f}")

    # Check if TRT is running on GPU while eager is on CPU
    print(f"\n  Eager mel device: {eager_mel.device}")
    print(f"  ET f_proj device: {et_f_proj.device}")

    # Try running eager on GPU to see if GPU matmul gives different results
    if torch.cuda.is_available():
        with torch.no_grad():
            eager_mel_gpu = eager_mel.cuda()
            mel_len_gpu = mel_len_tensor.cuda()
            encoder_gpu = encoder_with_proj.cuda()
            gpu_f_proj, gpu_enc_len = encoder_gpu(
                audio_signal=eager_mel_gpu, length=mel_len_gpu
            )
            gpu_f_proj = gpu_f_proj.cpu()

        gpu_np = gpu_f_proj[0, :valid_frames, :].float().numpy()
        gpu_cos = np.dot(eager_flat, gpu_np.flatten()) / (
            np.linalg.norm(eager_flat) * np.linalg.norm(gpu_np.flatten()) + 1e-8
        )
        trt_vs_gpu_cos = np.dot(et_flat, gpu_np.flatten()) / (
            np.linalg.norm(et_flat) * np.linalg.norm(gpu_np.flatten()) + 1e-8
        )
        print(f"\n  GPU eager vs CPU eager cosine: {gpu_cos:.8f}")
        print(f"  TRT vs GPU eager cosine: {trt_vs_gpu_cos:.8f}")
        print(f"  (If TRT≈GPU >> TRT≈CPU, the diff is GPU precision, not TRT)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
