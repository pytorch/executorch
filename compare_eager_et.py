#!/usr/bin/env python3
"""
Compare Parakeet model outputs: Eager PyTorch vs ExecuTorch module.
This validates that the TRT-exported model produces correct numerical results.
"""
import os
import sys
import torch
import numpy as np

from executorch.extension.pybindings.portable_lib import _load_for_executorch

# Re-define the wrapper classes here (avoid import issues)
class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio: torch.Tensor, length: torch.Tensor):
        mel, mel_len = self.preprocessor(
            input_signal=audio.unsqueeze(0),
            length=length,
        )
        return mel, mel_len


class EncoderWithProjection(torch.nn.Module):
    def __init__(self, encoder, joint):
        super().__init__()
        self.encoder = encoder
        self.enc_proj = joint.enc

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        f_proj = self.enc_proj(encoded)  # [B, T, joint_hidden]
        return f_proj, enc_len


class DecoderStep(torch.nn.Module):
    def __init__(self, decoder, joint):
        super().__init__()
        self.embedding = decoder.prediction.embed
        self.lstm = decoder.prediction.dec_rnn
        self.pred_proj = joint.pred

    def forward(self, token, h, c):
        embed = self.embedding(token)
        embed = embed.transpose(0, 1)  # [seq=1, batch=1, hidden]
        out, (h_new, c_new) = self.lstm(embed, (h, c))
        out = out.transpose(0, 1)  # back to [batch, seq, hidden]
        g_proj = self.pred_proj(out)  # [B, 1, joint_hidden]
        return g_proj, h_new, c_new


class JointWithArgmax(torch.nn.Module):
    def __init__(self, joint, num_classes):
        super().__init__()
        self.joint_net = joint.joint_net
        self.num_classes = num_classes

    def forward(self, f: torch.Tensor, g: torch.Tensor):
        joint_in = f + g
        logits = self.joint_net(joint_in)  # [B, 1, vocab+1]
        
        # TDT: first num_classes are tokens, rest are durations
        token_logits = logits[:, :, :self.num_classes]
        duration_logits = logits[:, :, self.num_classes:]
        
        token_id = torch.argmax(token_logits, dim=-1, keepdim=True)
        duration_idx = torch.argmax(duration_logits, dim=-1, keepdim=True)
        
        return torch.cat([token_id, duration_idx], dim=-1)


def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample if needed."""
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    return waveform

def compare_tensors(name, eager_out, et_out, rtol=1e-2, atol=1e-2):
    """Compare two tensors and report statistics."""
    eager_np = eager_out.detach().cpu().float().numpy()
    et_np = et_out.detach().cpu().float().numpy()
    
    # Basic stats
    abs_diff = np.abs(eager_np - et_np)
    rel_diff = abs_diff / (np.abs(eager_np) + 1e-8)
    
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")
    print(f"  Shape: eager={eager_out.shape}, et={et_out.shape}")
    print(f"  Eager range: [{eager_np.min():.6f}, {eager_np.max():.6f}], mean={eager_np.mean():.6f}")
    print(f"  ET range:    [{et_np.min():.6f}, {et_np.max():.6f}], mean={et_np.mean():.6f}")
    print(f"  Abs diff: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")
    print(f"  Rel diff: max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")
    
    # Check if close
    close = np.allclose(eager_np, et_np, rtol=rtol, atol=atol)
    if close:
        print(f"  ✅ PASS (rtol={rtol}, atol={atol})")
    else:
        print(f"  ❌ FAIL (rtol={rtol}, atol={atol})")
        # Show first few mismatches
        mismatches = ~np.isclose(eager_np, et_np, rtol=rtol, atol=atol)
        mismatch_indices = np.argwhere(mismatches)
        print(f"  Num mismatches: {mismatches.sum()} / {mismatches.size}")
        print(f"  First few mismatches:")
        for idx in mismatch_indices[:5]:
            idx_tuple = tuple(idx)
            print(f"    [{idx_tuple}]: eager={eager_np[idx_tuple]:.6f}, et={et_np[idx_tuple]:.6f}")
    
    return close

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pte", default="/home/gasoonjia/trt/executorch/parakeet_trt/model.pte")
    parser.add_argument("--audio", default="/home/gasoonjia/trt/executorch/examples/models/parakeet/real_speech.wav")
    args = parser.parse_args()
    
    print("="*70)
    print("Parakeet Eager vs ExecuTorch Comparison")
    print("="*70)
    
    pte_path = args.pte
    audio_path = args.audio
    
    print(f"\nLoading ExecuTorch module from: {pte_path}")
    et_module = _load_for_executorch(pte_path)
    
    # Print available methods
    print(f"Available methods: {et_module.method_names()}")
    
    print(f"\nLoading Parakeet model (eager mode)...")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    model.cuda()
    
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    audio_len = torch.tensor([len(audio)], dtype=torch.int64)
    print(f"  Audio shape: {audio.shape}, length: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    
    # ============================================================
    # Test 1: Preprocessor
    # ============================================================
    print("\n" + "="*70)
    print("TEST 1: Preprocessor")
    print("="*70)
    
    # Eager preprocessor
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()
    preprocessor_wrapper.float()
    
    with torch.no_grad():
        eager_mel, eager_mel_len = preprocessor_wrapper(audio, audio_len)
    print(f"  Eager mel shape: {eager_mel.shape}, len: {eager_mel_len}")
    
    # ET preprocessor
    et_result = et_module.run_method("preprocessor", (audio, audio_len))
    et_mel = et_result[0]
    et_mel_len = et_result[1]
    print(f"  ET mel shape: {et_mel.shape}, len: {et_mel_len}")
    
    preproc_ok = compare_tensors("preprocessor mel", eager_mel, et_mel)
    
    # ============================================================
    # Test 2: Encoder
    # ============================================================
    print("\n" + "="*70)
    print("TEST 2: Encoder")
    print("="*70)
    
    # Use eager mel output for encoder input
    mel_input = eager_mel.cuda().half()
    mel_len_input = eager_mel_len.cuda()
    
    # Eager encoder
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()
    encoder_with_proj.half()
    encoder_with_proj.cuda()
    
    with torch.no_grad():
        eager_f_proj, eager_enc_len = encoder_with_proj(audio_signal=mel_input, length=mel_len_input)
    print(f"  Eager f_proj shape: {eager_f_proj.shape}, enc_len: {eager_enc_len}")
    
    # ET encoder
    et_result = et_module.run_method("encoder", (mel_input.cpu(), mel_len_input.cpu()))
    et_f_proj = et_result[0]
    et_enc_len = et_result[1]
    print(f"  ET f_proj shape: {et_f_proj.shape}, enc_len: {et_enc_len}")
    
    encoder_ok = compare_tensors("encoder f_proj", eager_f_proj.cpu(), et_f_proj, rtol=0.1, atol=0.1)
    
    # Also do a frame-by-frame comparison for first few frames
    print("\n  Frame-by-frame comparison (first 5 frames):")
    for i in range(min(5, eager_f_proj.shape[1])):
        eager_frame = eager_f_proj[0, i, :].cpu()
        et_frame = et_f_proj[0, i, :]
        diff = (eager_frame - et_frame).abs()
        print(f"    Frame {i}: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
    
    # ============================================================
    # Test 3: Decoder Step
    # ============================================================
    print("\n" + "="*70)
    print("TEST 3: Decoder Step")
    print("="*70)
    
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    token = torch.tensor([[0]], dtype=torch.long, device="cuda")
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=torch.float16, device="cuda")
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=torch.float16, device="cuda")
    
    decoder_step = DecoderStep(model.decoder, model.joint)
    decoder_step.eval()
    decoder_step.half()
    decoder_step.cuda()
    
    with torch.no_grad():
        eager_g, eager_h, eager_c = decoder_step(token, h, c)
    print(f"  Eager g shape: {eager_g.shape}")
    
    # ET decoder
    et_result = et_module.run_method("decoder_step", (token.cpu(), h.cpu(), c.cpu()))
    et_g = et_result[0]
    print(f"  ET g shape: {et_g.shape}")
    
    decoder_ok = compare_tensors("decoder g_proj", eager_g.cpu(), et_g, rtol=0.05, atol=0.05)
    
    # ============================================================
    # Test 4: Joint network
    # ============================================================
    print("\n" + "="*70)
    print("TEST 4: Joint Network")
    print("="*70)
    
    # Get one encoder frame
    f_input = eager_f_proj[:, 0:1, :].cuda()  # [1, 1, joint_hidden]
    g_input = eager_g.cuda()  # from decoder
    
    joint_hidden = model.joint.joint_hidden
    num_token_classes = model.tokenizer.vocab_size + 1
    joint_module = JointWithArgmax(model.joint, num_token_classes)
    joint_module.eval()
    joint_module.half()
    joint_module.cuda()
    
    with torch.no_grad():
        eager_joint_out = joint_module(f_input, g_input)
    print(f"  Eager joint output: {eager_joint_out}")
    
    # ET joint
    et_result = et_module.run_method("joint", (f_input.cpu(), g_input.cpu()))
    et_joint_out = et_result[0]
    print(f"  ET joint output: {et_joint_out}")
    
    # Compare token predictions
    eager_token = eager_joint_out[0, 0, 0].item()
    et_token = et_joint_out[0, 0, 0].item()
    joint_ok = (eager_token == et_token)
    print(f"\n  Token comparison:")
    print(f"    Eager token: {int(eager_token)} (blank={num_token_classes-1})")
    print(f"    ET token:    {int(et_token)}")
    print(f"    {'✅ MATCH' if joint_ok else '❌ MISMATCH'}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Preprocessor: {'✅ PASS' if preproc_ok else '❌ FAIL'}")
    print(f"  Encoder:      {'✅ PASS' if encoder_ok else '❌ FAIL'}")
    print(f"  Decoder:      {'✅ PASS' if decoder_ok else '❌ FAIL'}")
    print(f"  Joint:        {'✅ PASS' if joint_ok else '❌ FAIL'}")
    
    all_pass = preproc_ok and encoder_ok and decoder_ok and joint_ok
    if all_pass:
        print("\n🎉 All tests passed! The ET module matches eager mode numerically.")
    else:
        print("\n⚠️ Some tests failed. Check the discrepancies above.")
    
    return all_pass

if __name__ == "__main__":
    with torch.no_grad():
        success = main()
    sys.exit(0 if success else 1)
