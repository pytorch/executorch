# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import struct
import wave


SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SAMPLES = 40000


def read_wav(path, max_samples):
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    if num_frames == 0:
        raise ValueError(f"{path}: audio file is empty")
    if compression != "NONE":
        raise ValueError(f"{path}: compressed WAV files are not supported")
    if channels != 1:
        raise ValueError(f"{path}: expected mono audio, got {channels} channels")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got {8 * sample_width}-bit")
    if max_samples > 0 and num_frames > max_samples:
        raise ValueError(
            f"{path}: {num_frames} samples exceeds MAX_AUDIO_SAMPLES={max_samples}. "
            "Trim the WAV or raise MAX_AUDIO_SAMPLES only if the target memory map "
            "can fit the larger embedded audio array."
        )

    samples = struct.unpack(f"<{num_frames}h", frames)
    return [sample / float(1 << 15) for sample in samples]


def write_header(audio_path, output_path, max_samples):
    samples = read_wav(audio_path, max_samples)
    array_lines = []
    for idx in range(0, len(samples), 8):
        line = ", ".join(f"{value:.8f}f" for value in samples[idx : idx + 8])
        array_lines.append(f"    {line},")

    header = f"""#include <stddef.h>

const float audio_data[{len(samples)}] = {{
{os.linesep.join(array_lines)}
}};

const size_t audio_data_len = {len(samples)};
"""
    with open(output_path, "w") as output_file:
        output_file.write(header)

    print(f"Converted '{audio_path}' to '{output_path}' ({len(samples)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a 16 kHz mono WAV file")
    parser.add_argument("--output", required=True, help="Output C header path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_AUDIO_SAMPLES,
        help="Maximum samples to embed; use 0 to disable the size guard",
    )
    args = parser.parse_args()

    write_header(args.audio, args.output, args.max_samples)
