"""Validate sortformer .pte against NeMo reference diarize() on audio.

Usage:
    python validate_pte.py \
        --nemo-path /path/to/model.nemo \
        --pte-path ./sortformer_exports/sortformer.pte \
        --wav /path/to/audio.wav
"""

import argparse
import os
import tempfile

import numpy as np
import soundfile as sf
import torch

from executorch.runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, required=True)
    parser.add_argument("--pte-path", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    # --- Load audio ---
    audio_np, sr = sf.read(args.wav, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        target_len = int(len(audio_np) * 16000 / sr)
        audio_np = np.interp(
            np.linspace(0, len(audio_np) - 1, target_len),
            np.arange(len(audio_np)),
            audio_np,
        )
        sr = 16000
    # pre_encode static shape = 4000 mel frames ≈ 40s
    audio_np = audio_np[: sr * 40]
    audio = torch.from_numpy(audio_np).float()
    length = torch.tensor([audio.shape[0]], dtype=torch.int64)
    print(f"Audio: {audio.shape[0]/sr:.1f}s, {sr}Hz")

    # Save truncated audio so NeMo diarize() processes the same segment
    tmp_wav = os.path.join(tempfile.gettempdir(), "validate_pte_truncated.wav")
    sf.write(tmp_wav, audio_np, sr)

    # --- Run NeMo reference ---
    print("Loading NeMo model...")
    from nemo.collections.asr.models import SortformerEncLabelModel

    if args.nemo_path.endswith(".nemo"):
        model = SortformerEncLabelModel.restore_from(
            args.nemo_path, map_location="cpu", strict=False
        )
    else:
        model = SortformerEncLabelModel.from_pretrained(
            args.nemo_path, map_location="cpu"
        )
    model.eval()

    # Very high latency config — single chunk for ≤27s audio
    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_update_period = 300

    print("Running NeMo diarize()...")
    nemo_segments, nemo_probs = model.diarize(
        audio=[tmp_wav], batch_size=1, include_tensor_outputs=True
    )

    nemo_preds = nemo_probs[0]  # first audio from list
    if nemo_preds.dim() == 3:
        nemo_preds = nemo_preds.squeeze(0)  # remove batch dim
    print(f"  NeMo segments: {len(nemo_segments[0])}")
    print(f"  NeMo preds: {nemo_preds.shape}")

    # --- Run .pte ---
    print("Loading .pte...")
    runtime = Runtime.get()
    program = runtime.load_program(args.pte_path)
    pte_pre = program.load_method("preprocessor")
    pte_penc = program.load_method("pre_encode")
    pte_enc = program.load_method("encode")

    print("Running .pte...")
    mel, mel_len = pte_pre.execute([audio, length])

    mel_t = mel.transpose(1, 2).contiguous()
    valid_mel = min(mel_t.shape[1], 4000)
    if mel_t.shape[1] < 4000:
        padded = torch.zeros(1, 4000, 128)
        padded[:, : mel_t.shape[1], :] = mel_t
        mel_t = padded
    else:
        mel_t = mel_t[:, :4000, :].contiguous()
    chunk_len = torch.tensor([valid_mel], dtype=torch.int64)

    embs, emb_len = pte_penc.execute([mel_t, chunk_len])
    embs = embs[:, : emb_len.item(), :].contiguous()
    pte_preds = pte_enc.execute([embs, emb_len])[0]
    pte_preds = pte_preds.squeeze(0)  # (T, 4)
    print(f"  PTE preds: {pte_preds.shape}")

    # --- Compare ---
    min_t = min(nemo_preds.shape[0], pte_preds.shape[0])
    nemo_aligned = nemo_preds[:min_t]
    pte_aligned = pte_preds[:min_t]

    max_diff = torch.abs(nemo_aligned - pte_aligned).max().item()
    mean_diff = torch.abs(nemo_aligned - pte_aligned).mean().item()
    agree = ((nemo_aligned > 0.5) == (pte_aligned > 0.5)).float().mean().item()

    print(f"\nFrames compared:    {min_t}")
    print(f"Max abs diff:       {max_diff:.6f}")
    print(f"Mean abs diff:      {mean_diff:.6f}")
    print(f"Decision agreement: {agree*100:.1f}%")
    # NeMo diarize() uses streaming (chunked processing with speaker cache/FIFO),
    # while .pte runs single-pass, so raw probabilities differ at chunk boundaries.
    # Decision agreement is the meaningful metric.
    print(f"Result:             {'PASS' if agree > 0.95 else 'FAIL'}")

    # Show NeMo reference segments
    print("\nNeMo diarization segments:")
    for seg in nemo_segments[0]:
        print(f"  {seg}")

    # Cleanup
    os.remove(tmp_wav)


if __name__ == "__main__":
    main()
