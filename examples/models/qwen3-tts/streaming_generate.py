"""Streaming TTS generation using ExecuTorch talker + decoder.

Interleaves talker code generation with decoder waveform synthesis,
emitting audio chunks as soon as enough codes are accumulated. This
mimics mlx-audio's streaming approach.

Usage:
    python streaming_generate.py \
        --talker-dir qwen3_tts_exports_talker_8da4w_s256 \
        --decoder-dir qwen3_tts_exports_8da4w_bucketed \
        --codes-path metal_test_codes.bin \
        --output-wav output_streaming.wav
"""

import argparse
import struct
import time
from pathlib import Path

import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def read_codes_binary(path: Path):
    with path.open("rb") as f:
        t_len, n_q = struct.unpack("<ii", f.read(8))
        values = struct.unpack(f"<{t_len * n_q}i", f.read(t_len * n_q * 4))
    return torch.tensor(values, dtype=torch.long).reshape(t_len, n_q)


def write_wav(path: str, waveform: list, sample_rate: int = 24000):
    import wave
    import array

    samples = []
    for s in waveform:
        clipped = max(-1.0, min(1.0, s))
        samples.append(int(clipped * 32767))

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(array.array("h", samples).tobytes())


def load_decoder_buckets(decoder_dir: Path):
    """Load all decoder bucket models from a directory."""
    import json

    manifest_path = decoder_dir / "export_manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    buckets = []
    for entry in manifest["buckets"]:
        codes_len = entry["codes_len"]
        pte_path = decoder_dir / entry["model_filename"]
        print(f"  Loading decoder bucket {codes_len}: {pte_path}")
        module = _load_for_executorch(str(pte_path))
        buckets.append((codes_len, module))

    buckets.sort(key=lambda x: x[0])
    return buckets


def select_decoder_bucket(buckets, codes_len):
    """Select smallest bucket >= codes_len."""
    for bucket_len, module in buckets:
        if bucket_len >= codes_len:
            return bucket_len, module
    return buckets[-1]


def decode_codes_chunk(decoder_module, codes, bucket_len):
    """Decode a chunk of codes using the decoder .pte.

    Args:
        decoder_module: ExecuTorch decoder module
        codes: [T, Q] tensor of codec codes
        bucket_len: padded length for this bucket

    Returns:
        waveform samples as list of floats
    """
    t_len, n_q = codes.shape

    # Pad to bucket length
    if t_len < bucket_len:
        padded = torch.full((bucket_len, n_q), -1, dtype=torch.long)
        padded[:t_len] = codes
        codes = padded

    # [1, bucket_len, n_q]
    codes_tensor = codes.unsqueeze(0)

    result = decoder_module.run_method("decode_codes", (codes_tensor,))
    wav_tensor = result[0]
    len_tensor = result[1]

    wav_len = int(len_tensor.item())
    wav_data = wav_tensor.squeeze()[:wav_len]
    return wav_data.tolist()


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS generation")
    parser.add_argument("--talker-dir", type=Path, required=True,
                        help="Directory with talker.pte and code_predictor.pte")
    parser.add_argument("--decoder-dir", type=Path, required=True,
                        help="Directory with bucketed decoder .pte files")
    parser.add_argument("--codes-path", type=Path, default=None,
                        help="Pre-generated codes file (skip talker, decode only)")
    parser.add_argument("--output-wav", type=str, default="output_streaming.wav")
    parser.add_argument("--chunk-size", type=int, default=25,
                        help="Number of codes per streaming decode chunk")
    args = parser.parse_args()

    # Load decoder buckets
    print("Loading decoder buckets...")
    decoder_buckets = load_decoder_buckets(args.decoder_dir)

    if args.codes_path is not None:
        # Decode-only mode: read pre-generated codes and decode in chunks
        print(f"Reading codes from: {args.codes_path}")
        codes = read_codes_binary(args.codes_path)
        t_len, n_q = codes.shape
        print(f"  codes_len={t_len}, num_quantizers={n_q}")

        all_samples = []
        total_start = time.time()
        first_audio_time = None

        n_chunks = (t_len + args.chunk_size - 1) // args.chunk_size
        print(f"\nStreaming decode: {n_chunks} chunks of {args.chunk_size} codes")

        for chunk_idx in range(n_chunks):
            chunk_start = time.time()
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, t_len)
            chunk_codes = codes[start:end]
            chunk_len = end - start

            # Select smallest bucket for this chunk
            bucket_len, decoder = select_decoder_bucket(decoder_buckets, chunk_len)

            # Decode
            samples = decode_codes_chunk(decoder, chunk_codes, bucket_len)
            all_samples.extend(samples)

            chunk_elapsed = time.time() - chunk_start
            chunk_audio_dur = len(samples) / 24000

            if first_audio_time is None:
                first_audio_time = time.time() - total_start
                print(f"  ** First audio at {first_audio_time:.2f}s **")

            print(f"  Chunk {chunk_idx + 1}/{n_chunks}: "
                  f"{chunk_len} codes -> {len(samples)} samples "
                  f"({chunk_audio_dur:.2f}s audio) in {chunk_elapsed:.2f}s "
                  f"(bucket={bucket_len})")

        total_elapsed = time.time() - total_start
        total_audio_dur = len(all_samples) / 24000

        print(f"\n=== Streaming Results ===")
        print(f"First audio:    {first_audio_time:.2f}s")
        print(f"Total time:     {total_elapsed:.2f}s")
        print(f"Audio duration: {total_audio_dur:.2f}s")
        print(f"RTF:            {total_audio_dur / total_elapsed:.2f}x realtime")

        # Write WAV
        write_wav(args.output_wav, all_samples)
        print(f"Wrote: {args.output_wav}")

    else:
        # Full pipeline: talker generation + streaming decode
        print("Loading talker models...")
        talker = _load_for_executorch(str(args.talker_dir / "talker.pte"))
        # code_predictor = _load_for_executorch(str(args.talker_dir / "code_predictor.pte"))
        print("  Loaded talker.pte")

        # TODO: Implement full talker generation loop
        # For now, this mode requires --codes-path
        print("ERROR: Full talker generation not yet implemented.")
        print("       Use --codes-path with pre-generated codes for now.")
        return


if __name__ == "__main__":
    main()
