#!/usr/bin/env python3
"""Compare encoder outputs: PyTorch eager vs torch_tensorrt (bypass ExecuTorch).
This tests whether the accuracy issue is in ExecuTorch's TRT converters or TRT itself.
"""
import sys
import numpy as np
import torch
import soundfile as sf

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


def main():
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

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

    # Load audio and preprocess
    audio = load_audio_sf(audio_path)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)

    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)
    mel_len_tensor = torch.tensor([eager_mel_len.item()], dtype=torch.int64)
    print(f"Mel: {eager_mel.shape}, mel_len={eager_mel_len.item()}")

    # CPU eager
    with torch.no_grad():
        cpu_f_proj, cpu_enc_len = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_tensor
        )
    print(f"\nCPU eager: f_proj={cpu_f_proj.shape}, enc_len={cpu_enc_len.item()}")

    # GPU eager — need to move the entire model, not just the wrapper
    model.cuda()
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint).cuda().eval()
    mel_gpu = eager_mel.cuda()
    mel_len_gpu = mel_len_tensor.cuda()
    with torch.no_grad():
        gpu_f_proj, gpu_enc_len = encoder_with_proj(
            audio_signal=mel_gpu, length=mel_len_gpu
        )
    gpu_f_proj = gpu_f_proj.cpu()
    print(f"GPU eager: f_proj={gpu_f_proj.shape}, enc_len={gpu_enc_len.item()}")

    valid = cpu_enc_len.item()

    # Compare CPU vs GPU
    cpu_np = cpu_f_proj[0, :valid, :].float().numpy()
    gpu_np = gpu_f_proj[0, :valid, :].float().numpy()
    cos_cpu_gpu = np.dot(cpu_np.flatten(), gpu_np.flatten()) / (
        np.linalg.norm(cpu_np.flatten()) * np.linalg.norm(gpu_np.flatten()) + 1e-8
    )
    print(f"\nCPU vs GPU eager cosine: {cos_cpu_gpu:.8f}")

    # Try torch.compile + TRT via torch_tensorrt
    try:
        import torch_tensorrt
        print(f"\ntorch_tensorrt version: {torch_tensorrt.__version__}")

        # Use torch_tensorrt.compile
        encoder_with_proj.cuda()
        encoder_with_proj.eval()

        # Export first
        print("Exporting encoder with torch.export...")
        exported = torch.export.export(
            encoder_with_proj,
            (mel_gpu, mel_len_gpu),
            dynamic_shapes={
                "audio_signal": {2: torch.export.Dim("T", min=161, max=5000)},
                "length": {},
            },
        )

        print("Compiling with torch_tensorrt...")
        trt_model = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=[1, 128, 161],
                    opt_shape=[1, 128, 5000],
                    max_shape=[1, 128, 5000],
                    dtype=torch.float32,
                ),
                torch_tensorrt.Input(
                    shape=[1],
                    dtype=torch.int64,
                ),
            ],
            enabled_precisions={torch.float32},
            truncate_double=True,
        )

        with torch.no_grad():
            trt_f_proj, trt_enc_len = trt_model(mel_gpu, mel_len_gpu)
        trt_f_proj = trt_f_proj.cpu()
        print(f"torch_tensorrt: f_proj={trt_f_proj.shape}, enc_len={trt_enc_len.item()}")

        trt_np = trt_f_proj[0, :valid, :].float().numpy()
        cos_cpu_trt = np.dot(cpu_np.flatten(), trt_np.flatten()) / (
            np.linalg.norm(cpu_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8
        )
        cos_gpu_trt = np.dot(gpu_np.flatten(), trt_np.flatten()) / (
            np.linalg.norm(gpu_np.flatten()) * np.linalg.norm(trt_np.flatten()) + 1e-8
        )
        print(f"\nCPU eager vs torch_tensorrt cosine: {cos_cpu_trt:.8f}")
        print(f"GPU eager vs torch_tensorrt cosine: {cos_gpu_trt:.8f}")
        print("(If torch_tensorrt has high cosine but ExecuTorch TRT has 0.66,")
        print(" the issue is in ExecuTorch's TRT converters)")

    except ImportError:
        print("\ntorch_tensorrt not available, skipping TRT comparison")
    except Exception as e:
        print(f"\ntorch_tensorrt compilation failed: {e}")

        # Fallback: try running the encoder through TRT with trtexec-like approach
        print("\nTrying alternative: PyTorch GPU with TF32 disabled...")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            gpu_notf32_f_proj, _ = encoder_with_proj(
                audio_signal=mel_gpu, length=mel_len_gpu
            )
        gpu_notf32_np = gpu_notf32_f_proj.cpu()[0, :valid, :].float().numpy()
        cos_notf32 = np.dot(cpu_np.flatten(), gpu_notf32_np.flatten()) / (
            np.linalg.norm(cpu_np.flatten()) * np.linalg.norm(gpu_notf32_np.flatten()) + 1e-8
        )
        print(f"CPU eager vs GPU (no TF32) cosine: {cos_notf32:.8f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
