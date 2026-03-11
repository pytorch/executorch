#!/usr/bin/env python3
"""Test the dynamic TRT export via pybind runtime with multiple audio files."""

import sys
import numpy as np
import torch
import soundfile as sf

# Import from the export script
sys.path.insert(0, "examples/models/parakeet")
from export_parakeet_tdt import (
    greedy_decode_executorch,
    load_model,
)


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio using soundfile (no FFmpeg dependency)."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    if sr != sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform

PTE_PATH = "/home/gasoonjia/trt/executorch/parakeet_tdt_exports/model.pte"
AUDIO_FILES = [
    "examples/models/parakeet/real_speech.wav",
    "examples/models/parakeet/test_49s.wav",
]


def main():
    print("Loading NeMo model (for tokenizer + config)...")
    model = load_model()

    print(f"Loading .pte from: {PTE_PATH}")
    with open(PTE_PATH, "rb") as f:
        pte_buffer = f.read()

    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_buffer)

    sample_rate = model.preprocessor._cfg.sample_rate
    vocab_size = model.tokenizer.vocab_size
    num_rnn_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden

    print(f"Model config: vocab_size={vocab_size}, num_rnn_layers={num_rnn_layers}, pred_hidden={pred_hidden}")

    for audio_path in AUDIO_FILES:
        print(f"\n{'='*60}")
        print(f"Testing: {audio_path}")
        print(f"{'='*60}")

        audio = load_audio(audio_path, sample_rate=sample_rate)
        print(f"  Audio shape: {audio.shape}, duration: {audio.shape[1]/sample_rate:.1f}s")

        with torch.no_grad():
            # Preprocessor
            preprocessor_method = program.load_method("preprocessor")
            audio_1d = audio.squeeze(0)
            audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
            proc_result = preprocessor_method.execute([audio_1d, audio_len])
            mel = proc_result[0]
            mel_len = proc_result[1].item()
            print(f"  Mel shape: {mel.shape}, mel_len: {mel_len}")

            # Encoder
            encoder_method = program.load_method("encoder")
            mel_len_tensor = torch.tensor([mel_len], dtype=torch.int64)
            enc_result = encoder_method.execute([mel, mel_len_tensor])
            f_proj = enc_result[0]
            encoded_len = enc_result[1].item()
            print(f"  Encoder output: {f_proj.shape}, encoded_len: {encoded_len}")

            # Decode
            tokens = greedy_decode_executorch(
                f_proj,
                encoded_len,
                program,
                blank_id=vocab_size,
                num_rnn_layers=num_rnn_layers,
                pred_hidden=pred_hidden,
            )

            text = model.tokenizer.ids_to_text(tokens)
            print(f"  Decoded {len(tokens)} tokens")
            print(f"  Transcription: {text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
