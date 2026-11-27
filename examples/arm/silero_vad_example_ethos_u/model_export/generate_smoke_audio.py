# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import struct
import wave


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
DEFAULT_DURATION_SECONDS = 2.5


def ensure_output_dir(output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def read_wav_i16(path: str) -> list[int]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    if compression != "NONE":
        raise ValueError(f"{path}: compressed WAV files are not supported")
    if channels != 1:
        raise ValueError(f"{path}: expected mono audio, got {channels} channels")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if sample_width != SAMPLE_WIDTH_BYTES:
        raise ValueError(f"{path}: expected 16-bit PCM WAV")
    if num_frames == 0:
        raise ValueError(f"{path}: audio file is empty")

    return list(struct.unpack(f"<{num_frames}h", frames))


def write_wav_i16(output_path: str, samples: list[int]) -> None:
    ensure_output_dir(output_path)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def generate_synthetic_samples(duration_seconds: float) -> list[int]:
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    samples = []

    for sample_index in range(num_samples):
        time_s = sample_index / sample_rate
        if time_s < 0.5 or time_s >= duration_seconds - 0.5:
            sample_value = 0.0
        else:
            active_duration = max(duration_seconds - 1.0, 0.001)
            envelope = 0.5 - 0.5 * math.cos(
                2.0 * math.pi * (time_s - 0.5) / active_duration
            )
            carrier = (
                0.6 * math.sin(2.0 * math.pi * 180.0 * time_s)
                + 0.3 * math.sin(2.0 * math.pi * 360.0 * time_s)
                + 0.1 * math.sin(2.0 * math.pi * 720.0 * time_s)
            )
            sample_value = 0.5 * envelope * carrier
        samples.append(max(-32768, min(32767, int(sample_value * 32767.0))))

    return samples


def trim_source_audio(
    source_wav: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    source_samples = read_wav_i16(source_wav)
    start_sample = int(start_seconds * SAMPLE_RATE)
    num_samples = int(duration_seconds * SAMPLE_RATE)
    end_sample = min(start_sample + num_samples, len(source_samples))
    samples = source_samples[start_sample:end_sample]
    if not samples:
        raise ValueError(
            f"{source_wav}: no samples available from {start_seconds:.3f}s"
        )

    write_wav_i16(output_path, samples)
    print(
        f"Trimmed '{source_wav}' to '{output_path}' "
        f"({len(samples)} samples from {start_seconds:.3f}s)"
    )


def generate_smoke_audio(output_path: str, duration_seconds: float) -> None:
    samples = generate_synthetic_samples(duration_seconds)
    write_wav_i16(output_path, samples)
    print(f"Generated '{output_path}' ({len(samples)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument(
        "--source-wav",
        help="Optional 16 kHz mono 16-bit PCM WAV to trim instead of synthesizing",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start offset when --source-wav is used",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Output duration in seconds",
    )
    args = parser.parse_args()

    if args.source_wav:
        trim_source_audio(
            args.source_wav,
            args.output,
            args.start_seconds,
            args.duration_seconds,
        )
    else:
        generate_smoke_audio(args.output, args.duration_seconds)
