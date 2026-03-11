#!/usr/bin/env python3
"""
Compare Parakeet model outputs: Eager PyTorch vs ExecuTorch (static fp32 TRT).
Uses the same wrapper classes as the canonical export_parakeet_tdt.py script.
"""
import sys
import numpy as np
import torch
import soundfile as sf

# Import wrapper classes from the canonical export script
sys.path.insert(0, "/home/gasoonjia/trt/executorch/examples/models/parakeet")
from export_parakeet_tdt import (
    EncoderWithProjection,
    DecoderStep,
    JointWithArgmax,
    PreprocessorWrapper,
    greedy_decode_executorch,
)

from executorch.runtime import Runtime


def load_audio_sf(audio_path, target_sr=16000):
    """Load audio using soundfile (avoids torchcodec/FFmpeg dependency)."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return torch.from_numpy(data).unsqueeze(0)  # [1, samples]


def compare_tensors(name, eager_out, et_out, rtol=1e-3, atol=1e-3):
    """Compare two tensors and report statistics."""
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

    # Cosine similarity
    eager_flat = eager_np.flatten()
    et_flat = et_np.flatten()
    cos_sim = np.dot(eager_flat, et_flat) / (np.linalg.norm(eager_flat) * np.linalg.norm(et_flat) + 1e-8)
    print(f"    Cosine similarity: {cos_sim:.8f}")

    close = np.allclose(eager_np, et_np, rtol=rtol, atol=atol)
    if close:
        print(f"    PASS (rtol={rtol}, atol={atol})")
    else:
        mismatches = ~np.isclose(eager_np, et_np, rtol=rtol, atol=atol)
        print(f"    FAIL (rtol={rtol}, atol={atol}) - {mismatches.sum()}/{mismatches.size} mismatches")

    return close, cos_sim


def main():
    pte_path = "/home/gasoonjia/trt/executorch/parakeet_trt_static_fp32/model.pte"
    audio_path = "/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav"

    print("=" * 70)
    print("Parakeet Eager vs ExecuTorch (Static FP32 TRT) Comparison")
    print("=" * 70)

    # Load ET module
    print(f"\nLoading ExecuTorch module: {pte_path}")
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    print("  Loaded successfully")

    # Load eager model
    print("\nLoading Parakeet model (eager mode, CPU, fp32)...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    model.cpu()

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio_sf(audio_path)
    audio_1d = audio.squeeze(0)
    audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
    print(f"  Audio: {audio_1d.shape} ({audio_1d.shape[0]/16000:.2f}s)")

    results = {}

    # ========== Test 1: Preprocessor ==========
    print("\n" + "=" * 70)
    print("TEST 1: Preprocessor")
    print("=" * 70)

    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.float()
    preprocessor_wrapper.eval()

    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio_1d, audio_len)
    print(f"  Eager mel: {eager_mel.shape}, len={eager_mel_len.item()}")

    preproc_method = program.load_method("preprocessor")
    et_result = preproc_method.execute([audio_1d, audio_len])
    et_mel = et_result[0]
    et_mel_len = et_result[1]
    print(f"  ET mel:    {et_mel.shape}, len={et_mel_len.item()}")

    ok, cos = compare_tensors("preprocessor mel", eager_mel, et_mel)
    results["preprocessor"] = (ok, cos)

    # ========== Test 2: Encoder ==========
    print("\n" + "=" * 70)
    print("TEST 2: Encoder (static shapes)")
    print("=" * 70)

    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()

    # For static encoder export, input must be padded to max_mel_frames (5000)
    max_mel_frames = model.encoder.max_audio_length  # 5000
    feat_in = getattr(model.encoder, "_feat_in", 128)
    actual_mel_len = eager_mel_len.item()

    # Pad mel to static size
    mel_for_encoder = torch.zeros(1, feat_in, max_mel_frames, dtype=torch.float32)
    mel_for_encoder[:, :, :eager_mel.shape[2]] = eager_mel
    mel_len_tensor = torch.tensor([actual_mel_len], dtype=torch.int64)

    print(f"  Encoder input: {mel_for_encoder.shape} (padded from {eager_mel.shape[2]} to {max_mel_frames})")

    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(
            audio_signal=mel_for_encoder, length=mel_len_tensor
        )
    print(f"  Eager f_proj: {eager_f_proj.shape}, enc_len={eager_enc_len.item()}")

    encoder_method = program.load_method("encoder")
    et_enc_result = encoder_method.execute([mel_for_encoder, mel_len_tensor])
    et_f_proj = et_enc_result[0]
    et_enc_len = et_enc_result[1]
    print(f"  ET f_proj:    {et_f_proj.shape}, enc_len={et_enc_len.item()}")

    # Compare only valid frames (not padding)
    valid_frames = eager_enc_len.item()
    print(f"  Comparing first {valid_frames} valid encoder frames...")
    ok, cos = compare_tensors(
        "encoder f_proj (valid frames)",
        eager_f_proj[:, :valid_frames, :],
        et_f_proj[:, :valid_frames, :],
        rtol=0.05, atol=0.05,
    )
    results["encoder"] = (ok, cos)

    # ========== Test 3: Decoder Step ==========
    print("\n" + "=" * 70)
    print("TEST 3: Decoder Step")
    print("=" * 70)

    decoder_step = DecoderStep(model.decoder, model.joint)
    decoder_step.eval()

    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    token = torch.tensor([[0]], dtype=torch.long)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=torch.float32)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=torch.float32)

    with torch.no_grad():
        eager_g, eager_h, eager_c = decoder_step(token, h, c)
    print(f"  Eager g: {eager_g.shape}")

    decoder_method = program.load_method("decoder_step")
    et_dec_result = decoder_method.execute([token, h, c])
    et_g = et_dec_result[0]
    print(f"  ET g:    {et_g.shape}")

    ok, cos = compare_tensors("decoder g_proj", eager_g, et_g, rtol=0.01, atol=0.01)
    results["decoder_step"] = (ok, cos)

    # ========== Test 4: Joint ==========
    print("\n" + "=" * 70)
    print("TEST 4: Joint Network")
    print("=" * 70)

    num_token_classes = model.tokenizer.vocab_size + 1
    joint_module = JointWithArgmax(model.joint, num_token_classes)
    joint_module.eval()

    # Use first valid encoder frame + decoder output
    f_input = eager_f_proj[:, 0:1, :]
    g_input = eager_g

    with torch.no_grad():
        eager_token_id, eager_dur_idx = joint_module(f_input, g_input)
    print(f"  Eager: token={eager_token_id.item()}, dur_idx={eager_dur_idx.item()}")

    joint_method = program.load_method("joint")
    et_joint_result = joint_method.execute([f_input, g_input])
    et_token_id = et_joint_result[0]
    et_dur_idx = et_joint_result[1]
    print(f"  ET:    token={et_token_id.item()}, dur_idx={et_dur_idx.item()}")

    joint_ok = (eager_token_id.item() == et_token_id.item())
    print(f"  Token match: {'PASS' if joint_ok else 'FAIL'}")
    results["joint"] = (joint_ok, 1.0 if joint_ok else 0.0)

    # ========== Test 5: End-to-end greedy decode ==========
    print("\n" + "=" * 70)
    print("TEST 5: End-to-end Greedy Decode")
    print("=" * 70)

    # Reload methods for greedy decode (they're stateful)
    vocab_size = model.tokenizer.vocab_size
    et_tokens = greedy_decode_executorch(
        et_f_proj,
        et_enc_len.item(),
        program,
        blank_id=vocab_size,
        num_rnn_layers=num_layers,
        pred_hidden=pred_hidden,
    )
    et_text = model.tokenizer.ids_to_text(et_tokens)
    print(f"  ET transcription: {et_text}")

    # Eager transcription using NeMo's built-in decoder
    with torch.no_grad():
        mel_for_nemo = mel_for_encoder
        encoded, enc_len = model.encoder(audio_signal=mel_for_nemo, length=mel_len_tensor)
        from export_parakeet_tdt import greedy_decode_eager
        eager_tokens = greedy_decode_eager(encoded, enc_len, model)
        eager_text = model.tokenizer.ids_to_text(eager_tokens)
    print(f"  Eager transcription: {eager_text}")

    text_match = (eager_text.strip() == et_text.strip())
    print(f"  Text match: {'PASS' if text_match else 'DIFFER'}")
    results["e2e_text"] = (text_match, 1.0 if text_match else 0.0)

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, (ok, cos) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:20s}: {status}  (cosine={cos:.8f})")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed - check details above.")

    return all_pass


if __name__ == "__main__":
    with torch.no_grad():
        success = main()
    sys.exit(0 if success else 1)
