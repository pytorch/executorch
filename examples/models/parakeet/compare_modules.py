#!/usr/bin/env python3
"""
Module-level numeric comparison between Eager and ExecuTorch exported Parakeet model.
Compares: preprocessor, encoder, decoder, joint outputs.
"""

import argparse
import os
import torch
import torchaudio
import numpy as np
from typing import Dict, Tuple, Optional

# Disable TF32 for numerical precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model(model_id: str = "nvidia/parakeet-tdt-0.6b-v3"):
    """Load the Parakeet model from HuggingFace."""
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()
    return model


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def compute_metrics(eager: torch.Tensor, et: torch.Tensor, name: str) -> Dict:
    """Compute comparison metrics between eager and ET outputs."""
    eager_np = eager.detach().cpu().float().numpy()
    et_np = et.detach().cpu().float().numpy()

    # Handle shape mismatches
    if eager_np.shape != et_np.shape:
        return {
            "name": name,
            "shape_match": False,
            "eager_shape": eager_np.shape,
            "et_shape": et_np.shape,
        }

    # Compute metrics
    abs_diff = np.abs(eager_np - et_np)
    rel_diff = abs_diff / (np.abs(eager_np) + 1e-8)

    # Cosine similarity (flatten for comparison)
    eager_flat = eager_np.flatten()
    et_flat = et_np.flatten()
    cosine_sim = np.dot(eager_flat, et_flat) / (
        np.linalg.norm(eager_flat) * np.linalg.norm(et_flat) + 1e-8
    )

    return {
        "name": name,
        "shape_match": True,
        "shape": eager_np.shape,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cosine_similarity": float(cosine_sim),
        "eager_range": (float(np.min(eager_np)), float(np.max(eager_np))),
        "et_range": (float(np.min(et_np)), float(np.max(et_np))),
    }


def print_metrics(metrics: Dict):
    """Pretty print comparison metrics."""
    name = metrics["name"]
    print(f"\n{'='*60}")
    print(f"Module: {name}")
    print('='*60)

    if not metrics.get("shape_match", True):
        print(f"  ❌ SHAPE MISMATCH!")
        print(f"     Eager shape: {metrics['eager_shape']}")
        print(f"     ET shape:    {metrics['et_shape']}")
        return

    print(f"  Shape: {metrics['shape']}")
    print(f"  Max Abs Diff:  {metrics['max_abs_diff']:.6e}")
    print(f"  Mean Abs Diff: {metrics['mean_abs_diff']:.6e}")
    print(f"  Max Rel Diff:  {metrics['max_rel_diff']:.6e}")
    print(f"  Mean Rel Diff: {metrics['mean_rel_diff']:.6e}")
    print(f"  Cosine Sim:    {metrics['cosine_similarity']:.8f}")
    print(f"  Eager range:   [{metrics['eager_range'][0]:.4f}, {metrics['eager_range'][1]:.4f}]")
    print(f"  ET range:      [{metrics['et_range'][0]:.4f}, {metrics['et_range'][1]:.4f}]")

    # Status
    if metrics['cosine_similarity'] > 0.999 and metrics['max_abs_diff'] < 0.01:
        print(f"  ✅ PASS: Outputs match closely")
    elif metrics['cosine_similarity'] > 0.99:
        print(f"  ⚠️  WARN: Small numerical differences")
    else:
        print(f"  ❌ FAIL: Significant differences detected")


class EncoderWithProjection(torch.nn.Module):
    """Wrapper matching export script's EncoderWithProjection."""
    def __init__(self, encoder, joint):
        super().__init__()
        self.encoder = encoder
        self.joint = joint

    def forward(self, audio_signal, length):
        encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
        f_proj = self.joint.enc(encoded)
        return f_proj, encoded_len


class DecoderStep(torch.nn.Module):
    """Wrapper matching export script's DecoderStep."""
    def __init__(self, decoder, joint):
        super().__init__()
        self.decoder = decoder
        self.joint = joint

    def forward(self, token, h, c):
        pred, (h_out, c_out) = self.decoder.predict(
            y=token, state=(h, c), add_sos=False, batch_size=None
        )
        g_proj = self.joint.pred(pred)
        return g_proj, h_out, c_out


class JointWithArgmax(torch.nn.Module):
    """Wrapper matching export script's JointWithArgmax."""
    def __init__(self, joint, num_classes):
        super().__init__()
        self.joint = joint
        self.num_classes = num_classes

    def forward(self, f, g):
        out = self.joint.joint_net(f + g)
        logits = out[..., :self.num_classes]
        durations = out[..., self.num_classes:]
        token_id = torch.argmax(logits, dim=-1)
        duration_idx = torch.argmax(durations, dim=-1)
        return token_id, duration_idx


def compare_modules(
    model,
    pte_path: str,
    audio_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict[str, Dict]:
    """Compare all modules between eager and ET."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    # (removed unused import)

    # Load ET model
    et_module = _load_for_executorch(pte_path)

    # Load audio
    waveform = load_audio(audio_path).to(device)

    results = {}

    # =========== PREPROCESSOR ===========
    print("\n[Testing Preprocessor]")
    with torch.no_grad():
        # Eager
        mel_eager, mel_len_eager = model.preprocessor(
            input_signal=waveform, length=torch.tensor([waveform.shape[-1]], device=device)
        )
        mel_eager = mel_eager.to(dtype)

        # ET
        et_inputs = [waveform.to(dtype).contiguous(), torch.tensor([waveform.shape[-1]], device=device)]
        et_outputs = et_module.run_method("preprocessor", tuple(et_inputs))
        mel_et = et_outputs[0]
        mel_len_et = et_outputs[1]

    results["preprocessor_mel"] = compute_metrics(mel_eager, mel_et, "preprocessor (mel)")
    print_metrics(results["preprocessor_mel"])

    # =========== ENCODER ===========
    print("\n[Testing Encoder]")
    encoder_wrapper = EncoderWithProjection(model.encoder, model.joint).to(device).to(dtype).eval()

    # Use same input shape as export (pad/truncate mel to max_mel_frames)
    max_mel_frames = 3072
    feat_in = mel_eager.shape[1]

    # Prepare encoder input (match export shape)
    mel_frames = mel_eager.shape[2]
    if mel_frames < max_mel_frames:
        mel_padded = torch.zeros(1, feat_in, max_mel_frames, dtype=dtype, device=device)
        mel_padded[:, :, :mel_frames] = mel_eager
        audio_signal = mel_padded
        length = torch.tensor([mel_frames], dtype=torch.int64, device=device)
    else:
        audio_signal = mel_eager[:, :, :max_mel_frames]
        length = torch.tensor([max_mel_frames], dtype=torch.int64, device=device)

    with torch.no_grad():
        # Eager
        f_proj_eager, enc_len_eager = encoder_wrapper(audio_signal=audio_signal, length=length)

        # ET
        et_inputs = [audio_signal.contiguous(), length]
        et_outputs = et_module.run_method("encoder", tuple(et_inputs))
        f_proj_et = et_outputs[0]
        enc_len_et = et_outputs[1]

    results["encoder_f_proj"] = compute_metrics(f_proj_eager, f_proj_et, "encoder (f_proj)")
    print_metrics(results["encoder_f_proj"])

    # Test individual frames
    print("\n[Testing Encoder Frame Consistency]")
    for frame_idx in [0, 10, 50]:
        if frame_idx < f_proj_eager.shape[1]:
            frame_metrics = compute_metrics(
                f_proj_eager[:, frame_idx:frame_idx+1, :],
                f_proj_et[:, frame_idx:frame_idx+1, :],
                f"encoder frame {frame_idx}"
            )
            print_metrics(frame_metrics)

    # =========== DECODER ===========
    print("\n[Testing Decoder]")
    decoder_wrapper = DecoderStep(model.decoder, model.joint).to(device).to(dtype).eval()

    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden

    token = torch.tensor([[0]], dtype=torch.long, device=device)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype, device=device)

    with torch.no_grad():
        # Eager
        g_proj_eager, h_eager, c_eager = decoder_wrapper(token, h, c)

        # ET
        et_inputs = [token, h.contiguous(), c.contiguous()]
        et_outputs = et_module.run_method("decoder_step", tuple(et_inputs))
        g_proj_et = et_outputs[0]
        h_et = et_outputs[1]
        c_et = et_outputs[2]

    results["decoder_g_proj"] = compute_metrics(g_proj_eager, g_proj_et, "decoder (g_proj)")
    print_metrics(results["decoder_g_proj"])

    results["decoder_h"] = compute_metrics(h_eager, h_et, "decoder (h state)")
    print_metrics(results["decoder_h"])

    # =========== JOINT ===========
    print("\n[Testing Joint]")
    num_token_classes = model.tokenizer.vocab_size + 1
    joint_wrapper = JointWithArgmax(model.joint, num_token_classes).to(device).to(dtype).eval()

    # Use actual f_proj and g_proj from encoder/decoder
    f_test = f_proj_eager[:, 0:1, :]  # First frame
    g_test = g_proj_eager

    with torch.no_grad():
        # Eager
        token_id_eager, dur_idx_eager = joint_wrapper(f_test, g_test)

        # ET
        et_inputs = [f_test.contiguous(), g_test.contiguous()]
        et_outputs = et_module.run_method("joint", tuple(et_inputs))
        token_id_et = et_outputs[0]
        dur_idx_et = et_outputs[1]

    # For joint, compare token IDs directly (they should match exactly)
    token_match = torch.all(token_id_eager == token_id_et).item()
    dur_match = torch.all(dur_idx_eager == dur_idx_et).item()

    print(f"\n{'='*60}")
    print(f"Module: joint")
    print('='*60)
    print(f"  Token ID match: {'✅ PASS' if token_match else '❌ FAIL'}")
    print(f"  Duration IDX match: {'✅ PASS' if dur_match else '❌ FAIL'}")
    print(f"  Eager token: {token_id_eager.item()}, ET token: {token_id_et.item()}")
    print(f"  Eager dur: {dur_idx_eager.item()}, ET dur: {dur_idx_et.item()}")

    results["joint"] = {
        "name": "joint",
        "token_match": token_match,
        "duration_match": dur_match,
        "eager_token": token_id_eager.item(),
        "et_token": token_id_et.item(),
    }

    # =========== SUMMARY ===========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = True
    for key, metrics in results.items():
        if "cosine_similarity" in metrics:
            status = "✅" if metrics["cosine_similarity"] > 0.99 else "❌"
            print(f"  {status} {metrics['name']}: cosine={metrics['cosine_similarity']:.6f}, max_diff={metrics['max_abs_diff']:.2e}")
            if metrics["cosine_similarity"] <= 0.99:
                all_pass = False
        elif "token_match" in metrics:
            status = "✅" if metrics["token_match"] and metrics["duration_match"] else "❌"
            print(f"  {status} {metrics['name']}: token_match={metrics['token_match']}, dur_match={metrics['duration_match']}")
            if not (metrics["token_match"] and metrics["duration_match"]):
                all_pass = False

    print("\n" + ("✅ ALL MODULES PASS" if all_pass else "❌ SOME MODULES FAILED"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Parakeet modules: Eager vs ExecuTorch")
    parser.add_argument("--pte-path", type=str, required=True, help="Path to exported .pte file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model-id", type=str, default="nvidia/parakeet-tdt-0.6b-v3",
                        help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="Data type")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Loading model: {args.model_id}")
    model = load_model(args.model_id)
    model = model.to(args.device).to(dtype).eval()

    print(f"Comparing with PTE: {args.pte_path}")
    print(f"Audio: {args.audio}")
    print(f"Device: {args.device}, dtype: {args.dtype}")

    results = compare_modules(model, args.pte_path, args.audio, args.device, dtype)

    return results


if __name__ == "__main__":
    main()
