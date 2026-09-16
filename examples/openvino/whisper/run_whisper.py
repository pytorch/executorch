# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-not-found"

"""
Run the three-way Whisper split (encoder.pte, cross_kv.pte, decoder.pte) through
the ExecuTorch runtime with the OpenVINO backend.

Usage:
    python run_whisper.py --model_dir ./whisper_ov --audio /path/to/audio.wav

    # Or use sample audio from HuggingFace datasets:
    python run_whisper.py --model_dir ./whisper_ov --use_sample_audio
"""

import argparse
import json
import logging
import os
import time

import torch
from executorch.runtime import Runtime, Verification
from transformers import AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_audio(audio_path, use_sample_audio, processor):
    if use_sample_audio:
        from datasets import load_dataset

        dataset = load_dataset(
            "distil-whisper/librispeech_long", "clean", split="validation"
        )
        sample = dataset[0]["audio"]
        audio_array = sample["array"]
        sampling_rate = sample["sampling_rate"]
    else:
        import soundfile as sf

        audio_array, sampling_rate = sf.read(audio_path)

    features = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features
    max_frames = 3000
    if features.shape[2] > max_frames:
        features = features[:, :, :max_frames].contiguous()
    return features


def main():
    parser = argparse.ArgumentParser(description="Run the OpenVINO Whisper split")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with encoder.pte, cross_kv.pte, decoder.pte, metadata.json",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace model ID for the processor/tokenizer "
        "(defaults to the model_id stored in metadata.json)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to an audio file (WAV, etc.) sampled at 16kHz",
    )
    parser.add_argument(
        "--use_sample_audio",
        action="store_true",
        help="Use sample audio from HuggingFace datasets",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.audio and not args.use_sample_audio:
        logger.warning("No audio specified; using --use_sample_audio")
        args.use_sample_audio = True

    with open(os.path.join(args.model_dir, "metadata.json")) as f:
        meta = json.load(f)
    model_id = args.model_id or meta["model_id"]
    start_token = meta["decoder_start_token_id"]
    eos_token = meta["eos_token_id"]
    num_layers = meta["num_decoder_layers"]
    max_cache = meta["max_cache_length"]

    logger.info(f"Loading processor/tokenizer from {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    features = load_audio(args.audio, args.use_sample_audio, processor)
    logger.info(f"Input features: {tuple(features.shape)} dtype={features.dtype}")

    rt = Runtime.get()
    logger.info("Loading encoder.pte ...")
    enc_method = rt.load_program(
        os.path.join(args.model_dir, "encoder.pte"), verification=Verification.Minimal
    ).load_method("forward")
    logger.info("Loading cross_kv.pte ...")
    ck_method = rt.load_program(
        os.path.join(args.model_dir, "cross_kv.pte"), verification=Verification.Minimal
    ).load_method("forward")
    logger.info("Loading decoder.pte ...")
    dec_method = rt.load_program(
        os.path.join(args.model_dir, "decoder.pte"), verification=Verification.Minimal
    ).load_method("forward")

    overall_start = time.perf_counter()

    t0 = time.perf_counter()
    encoder_hidden = enc_method.execute([features])[0]
    t_enc = time.perf_counter() - t0
    logger.info(f"Encoder: {t_enc*1000:.0f}ms, output {tuple(encoder_hidden.shape)}")

    t0 = time.perf_counter()
    ck_out = ck_method.execute([encoder_hidden])
    cross_k = tuple(ck_out[:num_layers])
    cross_v = tuple(ck_out[num_layers : 2 * num_layers])
    t_ck = time.perf_counter() - t0
    logger.info(f"Cross-KV: {t_ck*1000:.0f}ms, {num_layers} layers")

    # Additive mask: 0 for valid positions [0..step], -inf for future positions.
    base_mask = torch.full((1, 1, 1, max_cache), float("-inf"))
    tokens = [start_token]
    decode_start = time.perf_counter()
    for step in range(args.max_new_tokens):
        ids = torch.tensor([[tokens[-1]]], dtype=torch.long)
        pos = torch.tensor([step], dtype=torch.long)
        mask = base_mask.clone()
        mask[..., : step + 1] = 0.0
        logits = dec_method.execute([ids, pos, mask, *cross_k, *cross_v])[0]
        next_tok = int(torch.argmax(logits.view(-1)).item())
        tokens.append(next_tok)
        if next_tok == eos_token:
            break
    t_dec = time.perf_counter() - decode_start

    n_gen = len(tokens) - 1
    tok_s = n_gen / t_dec if t_dec > 0 else 0.0
    total = time.perf_counter() - overall_start
    transcript = tokenizer.decode(tokens, skip_special_tokens=True)

    print(f"\nEncoder time:  {t_enc:.3f}s")
    print(f"Cross-KV time: {t_ck:.3f}s")
    print(f"Decode time:   {t_dec:.3f}s ({n_gen} tokens, {tok_s:.1f} tok/s)")
    print(f"Total time:    {total:.3f}s")
    print("\n" + "=" * 60)
    print("Transcript:")
    print("=" * 60)
    print(transcript)
    print("=" * 60)


if __name__ == "__main__":
    main()
