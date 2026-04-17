#!/usr/bin/env python3
"""Resample a WAV to 16 kHz and transcribe via the parakeet ExecuTorch runner.

Prints the transcript to stdout (matching the interface that
verify_xnnpack_transcript.py expects from the STT command).
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to input WAV (any sample rate)")
    parser.add_argument("--parakeet-runner", required=True)
    parser.add_argument("--parakeet-model", required=True)
    parser.add_argument("--parakeet-tokenizer", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found", flush=True)
        return 1

    with tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False) as tmp:
        tmp_path = tmp.name

    data, _ = librosa.load(str(audio_path), sr=16000)
    sf.write(tmp_path, data, 16000, subtype="PCM_16")

    result = subprocess.run(
        [
            args.parakeet_runner,
            "--model_path", args.parakeet_model,
            "--tokenizer_path", args.parakeet_tokenizer,
            "--audio_path", tmp_path,
        ],
        capture_output=True,
        text=True,
    )

    Path(tmp_path).unlink(missing_ok=True)

    transcript = ""
    for line in result.stdout.splitlines():
        m = re.match(r"Transcribed text:\s*(.*)", line)
        if m:
            transcript = m.group(1).strip()
            break

    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
