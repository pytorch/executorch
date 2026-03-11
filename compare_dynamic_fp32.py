#!/usr/bin/env python3
"""
Compare Parakeet model outputs: Eager PyTorch vs ExecuTorch (dynamic fp32 TRT).
Tests with multiple audio files to verify dynamic shapes work correctly.
"""
import sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import (
    EncoderWithProjection,
    DecoderStep,
    JointWithArgmax,
    PreprocessorWrapper,
    greedy_decode_executorch,
    greedy_decode_eager,
)

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


def compare_tensors(name, eager_out, et_out, rtol=1e-3, atol=1e-3):
    eager_np = eager_out.detach().cpu().float().numpy()
    et_np = et_out.detach().cpu().float().numpy()

    abs_diff = np.abs(eager_np - et_np)
    rel_diff = abs_diff / (np.abs(eager_np) + 1e-8)

    print(f"\n  Comparing: {name}")
    print(f"    Shape: eager={eager_out.shape}, et={et_out.shape}")
    print(f"    Eager: [{eager_np.min():.6f}, {eager_np.max():.6f}], mean={eager_np.mean():.6f}")
    print(f"    ET:    [{et_np.min():.6f}, {et_np.max():.6f}], mean={et_np.mean():.6f}")
    print(f"    Abs diff: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")
    print(f"    Rel diff: max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")

    eager_flat = eager_np.flatten()
    et_flat = et_np.flatten()
    cos_sim = np.dot(eager_flat, et_flat) / (
        np.linalg.norm(eager_flat) * np.linalg.norm(et_flat) + 1e-8
    )
    print(f"    Cosine similarity: {cos_sim:.8f}")

    close = np.allclose(eager_np, et_np, rtol=rtol, atol=atol)
    if close:
        print(f"    PASS (rtol={rtol}, atol={atol})")
    else:
        mismatches = ~np.isclose(eager_np, et_np, rtol=rtol, atol=atol)
        print(f"    FAIL (rtol={rtol}, atol={atol}) - {mismatches.sum()}/{mismatches.size} mismatches")

    return close, cos_sim


def test_audio(audio_path, program, model, encoder_with_proj, preprocessor_wrapper):
    """Test a single audio file through the full pipeline."""
    print(f"\n{'='*70}")
    print(f"Testing: {audio_path}")
    print(f"{'='*70}")

    audio = load_audio_sf(audio_path)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
    print(f"  Audio: {audio_1d.shape} ({audio_1d.shape[0]/16000:.2f}s)")

    # Preprocessor
    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)
    print(f"  Mel: {eager_mel.shape}, mel_len={eager_mel_len.item()}")

    preproc_method = program.load_method("preprocessor")
    et_result = preproc_method.execute([audio_1d, audio_len])
    et_mel = et_result[0]
    et_mel_len = et_result[1]

    _, preproc_cos = compare_tensors("preprocessor", eager_mel, et_mel)

    # Encoder — dynamic: pass raw mel without padding
    mel_len_tensor = torch.tensor([eager_mel_len.item()], dtype=torch.int64)

    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(
            audio_signal=eager_mel, length=mel_len_tensor
        )
    print(f"  Eager encoder out: {eager_f_proj.shape}, enc_len={eager_enc_len.item()}")

    encoder_method = program.load_method("encoder")
    et_enc_result = encoder_method.execute([eager_mel, mel_len_tensor])
    et_f_proj = et_enc_result[0]
    et_enc_len = et_enc_result[1]
    print(f"  ET encoder out:    {et_f_proj.shape}, enc_len={et_enc_len.item()}")

    valid_frames = min(eager_enc_len.item(), et_enc_len.item())
    _, enc_cos = compare_tensors(
        "encoder (valid frames)",
        eager_f_proj[:, :valid_frames, :],
        et_f_proj[:, :valid_frames, :],
        rtol=0.05, atol=0.05,
    )

    # E2E greedy decode
    vocab_size = model.tokenizer.vocab_size
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden

    et_tokens = greedy_decode_executorch(
        et_f_proj,
        et_enc_len.item(),
        program,
        blank_id=vocab_size,
        num_rnn_layers=num_layers,
        pred_hidden=pred_hidden,
    )
    et_text = model.tokenizer.ids_to_text(et_tokens)
    print(f"\n  ET transcription:    {et_text}")

    with torch.no_grad():
        encoded, enc_len = model.encoder(audio_signal=eager_mel, length=mel_len_tensor)
        eager_tokens = greedy_decode_eager(encoded, enc_len, model)
        eager_text = model.tokenizer.ids_to_text(eager_tokens)
    print(f"  Eager transcription: {eager_text}")

    text_match = eager_text.strip() == et_text.strip()
    print(f"  Text match: {'PASS' if text_match else 'DIFFER'}")

    return {
        "preproc_cos": preproc_cos,
        "enc_cos": enc_cos,
        "et_text": et_text,
        "eager_text": eager_text,
        "text_match": text_match,
    }


def main():
    pte_path = "/home/gasoonjia/trt/executorch/parakeet_trt_dynamic_fp32/model.pte"
    audio_files = [
        "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav",
        "/home/gasoonjia/trt/executorch/examples/models/parakeet/test_audio.wav",
    ]

    print("=" * 70)
    print("Parakeet Eager vs ExecuTorch (Dynamic FP32 TRT) Comparison")
    print("=" * 70)

    print(f"\nLoading ExecuTorch module: {pte_path}")
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    print("  Loaded successfully")

    print("\nLoading Parakeet model (eager mode, CPU, fp32)...")
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

    all_results = {}
    for audio_path in audio_files:
        results = test_audio(audio_path, program, model, encoder_with_proj, preprocessor_wrapper)
        all_results[audio_path] = results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for path, r in all_results.items():
        name = path.split("/")[-1]
        print(f"\n  {name}:")
        print(f"    Preprocessor cosine: {r['preproc_cos']:.8f}")
        print(f"    Encoder cosine:      {r['enc_cos']:.8f}")
        print(f"    Text match:          {'PASS' if r['text_match'] else 'DIFFER'}")
        print(f"    ET:    {r['et_text']}")
        print(f"    Eager: {r['eager_text']}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
