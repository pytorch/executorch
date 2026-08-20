#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 E2B/E4B inference runner for ExecuTorch.

Supports audio transcription/translation, image understanding, and text-only
generation using a single PTE with 4 methods (speech_transform, audio_encoder,
vision_encoder, text_decoder).

Usage:
    # Audio transcription (E2B):
    buck2 run fbcode//executorch/examples/models/gemma4:run_gemma4 -- \
        --model_path /tmp/gemma4.pte \
        --audio_path /tmp/audio.wav \
        --tokenizer_path /tmp/tokenizer.model

    # Image understanding:
    buck2 run fbcode//executorch/examples/models/gemma4:run_gemma4 -- \
        --model_path /tmp/gemma4.pte \
        --image_path /tmp/photo.jpg \
        --tokenizer_path /tmp/tokenizer.model \
        --prompt "Describe this image in detail:"

    # Text-only:
    buck2 run fbcode//executorch/examples/models/gemma4:run_gemma4 -- \
        --model_path /tmp/gemma4.pte \
        --tokenizer_path /tmp/tokenizer.model \
        --prompt "What is the capital of France?"
"""

import argparse
import logging
import os
import time
import wave

import numpy as np
import sentencepiece
import torch
from executorch.runtime import Runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOS_TOKEN_ID = 2
TURN_START_ID = 105  # <|turn>
TURN_END_ID = 106  # <turn|>
AUDIO_TOKEN_ID = 258881
IMAGE_TOKEN_ID = 258880
BOI_TOKEN_ID = 255999  # <|boi|>
EOI_TOKEN_ID = 258882  # <|eoi|>
HIDDEN_SIZE = {"e2b": 1536, "e4b": 2560}
STOP_TOKENS = {1, 106}  # EOS, <turn|>


def pad_to_valid_frames(num_frames: int) -> int:
    k = (num_frames + 25 + 47) // 48
    return 48 * k - 25


def load_audio(path: str) -> torch.Tensor:
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected 16-bit PCM"
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.tensor(samples)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 E2B/E4B inference runner")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Single PTE with 4 methods (speech_transform, audio_encoder, vision_encoder, text_decoder)",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="WAV audio file (16kHz, 16-bit PCM). If omitted, runs text-only.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Image file (JPEG/PNG). If omitted, runs audio or text-only.",
    )
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Transcribe the following audio:")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument(
        "--variant",
        type=str,
        default="e2b",
        choices=["e2b", "e4b"],
        help="Model variant (determines hidden_size for text-only mode)",
    )
    return parser.parse_args()


def _run_audio_pipeline(prog, audio_path: str):
    """Run speech_transform + audio_encoder. Returns (audio_out, num_tokens, t_st_ms, t_ae_ms, audio_duration_s)."""
    st_method = prog.load_method("speech_transform")
    ae_method = prog.load_method("audio_encoder")

    waveform = load_audio(audio_path)
    audio_duration_s = len(waveform) / 16000
    logger.info(f"Audio: {len(waveform)} samples, {audio_duration_s:.2f}s")

    t0 = time.perf_counter()
    padded_len = ((len(waveform) + 127) // 128) * 128
    padded = torch.nn.functional.pad(waveform, (0, padded_len - len(waveform)))
    mel = st_method.execute([padded])[0]
    t_st_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Speech transform: {mel.shape} ({t_st_ms:.0f} ms)")

    t0 = time.perf_counter()
    num_frames = mel.shape[0]
    valid_frames = pad_to_valid_frames(num_frames)
    mel_padded = torch.nn.functional.pad(
        mel, (0, 0, 0, valid_frames - num_frames)
    ).unsqueeze(0)
    mel_mask = torch.ones(1, valid_frames, dtype=torch.bool)
    mel_mask[0, num_frames:] = False
    mel_padded = mel_padded * mel_mask.unsqueeze(-1).float()

    ae_results = ae_method.execute([mel_padded.float(), mel_mask])
    audio_out = ae_results[0]
    t_ae_ms = (time.perf_counter() - t0) * 1000

    encoder_tokens = audio_out.shape[1]
    if len(ae_results) > 1:
        num_tokens = int(ae_results[1].sum().item())
    else:
        # Fallback: compute from waveform (for older PTEs without mask output)
        padded_samples = len(waveform) + 160
        mel_from_waveform = (padded_samples - 321) // 160 + 1
        after_conv1 = (mel_from_waveform + 2 - 3) // 2 + 1
        after_conv2 = (after_conv1 + 2 - 3) // 2 + 1
        num_tokens = min(after_conv2, 750, encoder_tokens)
    logger.info(
        f"Audio encoder: {audio_out.shape}, {num_tokens} valid tokens "
        f"(stripped {encoder_tokens - num_tokens} padding) ({t_ae_ms:.0f} ms)"
    )
    return audio_out, num_tokens, t_st_ms, t_ae_ms, audio_duration_s


def _run_vision_pipeline(prog, image_path: str):
    """Run vision_encoder. Returns (vision_out, num_tokens, t_ve_ms)."""
    from executorch.examples.models.gemma4.image_utils import preprocess_image

    ve_method = prog.load_method("vision_encoder")

    t0 = time.perf_counter()
    pixel_values, pixel_position_ids, num_soft_tokens = preprocess_image(image_path)
    t_preprocess_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Image preprocessed: {pixel_values.shape}, {num_soft_tokens} soft tokens ({t_preprocess_ms:.0f} ms)"
    )

    t0 = time.perf_counter()
    ve_results = ve_method.execute([pixel_values, pixel_position_ids])
    vision_out = ve_results[0]
    t_ve_ms = (time.perf_counter() - t0) * 1000

    if len(ve_results) > 1:
        num_tokens = int(ve_results[1].sum().item())
    else:
        num_tokens = num_soft_tokens
    logger.info(
        f"Vision encoder: {vision_out.shape}, {num_tokens} valid tokens ({t_ve_ms:.0f} ms)"
    )
    return vision_out, num_tokens, t_ve_ms


def _build_prompt_ids(
    sp,
    prompt: str,
    has_audio: bool,
    has_image: bool,
    num_tokens: int,
) -> list:
    """Assemble the input token ID list for the model turn."""
    ids = [BOS_TOKEN_ID, TURN_START_ID]
    ids += sp.encode("user\n")
    if has_audio:
        ids += [AUDIO_TOKEN_ID] * num_tokens
    elif has_image:
        ids += [BOI_TOKEN_ID] + [IMAGE_TOKEN_ID] * num_tokens + [EOI_TOKEN_ID]
    ids += sp.encode(prompt)
    ids += [TURN_END_ID]
    ids += sp.encode("\n")
    ids += [TURN_START_ID]
    ids += sp.encode("model\n")
    return ids


def _build_inputs_embeds(
    ids: list,
    audio_out,
    vision_out,
    num_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Build the inputs_embeds tensor with audio/vision outputs scattered into placeholder positions."""
    seq_len = len(ids)
    inputs_embeds = torch.zeros(1, seq_len, hidden_size, dtype=torch.float32)
    if audio_out is not None:
        audio_idx = 0
        for i in range(seq_len):
            if ids[i] == AUDIO_TOKEN_ID and audio_idx < num_tokens:
                inputs_embeds[0, i] = audio_out[0, audio_idx]
                audio_idx += 1
    elif vision_out is not None:
        vision_idx = 0
        for i in range(seq_len):
            if ids[i] == IMAGE_TOKEN_ID and vision_idx < num_tokens:
                inputs_embeds[0, i] = vision_out[0, vision_idx]
                vision_idx += 1
    return inputs_embeds


def _log_perf_summary(
    pte_size_mb: float,
    seq_len: int,
    decode_tokens: int,
    t_st_ms: float,
    t_ae_ms: float,
    t_ve_ms: float,
    t_prefill_ms: float,
    t_decode_ms: float,
    total_ms: float,
    audio_duration_s: float,
    has_audio: bool,
) -> None:
    logger.info("=" * 50)
    logger.info("Performance Summary")
    logger.info("=" * 50)
    logger.info(f"Model size:       {pte_size_mb:.0f} MB")
    if t_st_ms > 0:
        logger.info(f"Speech transform: {t_st_ms:.0f} ms")
    if t_ae_ms > 0:
        logger.info(f"Audio encoder:    {t_ae_ms:.0f} ms")
    if t_ve_ms > 0:
        logger.info(f"Vision encoder:   {t_ve_ms:.0f} ms")
    logger.info(
        f"Prefill:          {t_prefill_ms:.0f} ms ({seq_len} tokens, {seq_len/t_prefill_ms*1000:.0f} tok/s)"
    )
    if decode_tokens > 0 and t_decode_ms > 0:
        logger.info(
            f"Decode:           {t_decode_ms:.0f} ms ({decode_tokens} tokens, {decode_tokens/t_decode_ms*1000:.1f} tok/s)"
        )
    ttft = t_st_ms + t_ae_ms + t_ve_ms + t_prefill_ms
    logger.info(f"TTFT:             {ttft:.0f} ms")
    logger.info(f"Total:            {total_ms:.0f} ms")
    if has_audio:
        rtf = total_ms / 1000 / audio_duration_s
        logger.info(f"Audio duration:   {audio_duration_s:.1f}s")
        logger.info(f"RTF:              {rtf:.2f}")


def main():
    args = _parse_args()

    rt = Runtime.get()
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)

    t_total_start = time.perf_counter()

    pte_size_mb = os.path.getsize(args.model_path) / (1024 * 1024)
    logger.info(f"Loading model: {args.model_path} ({pte_size_mb:.0f} MB)")
    prog = rt.load_program(args.model_path)
    td_method = prog.load_method("text_decoder")
    t_load = time.perf_counter()
    logger.info(f"Loaded in {(t_load - t_total_start)*1000:.0f} ms")

    if args.audio_path and args.image_path:
        raise ValueError(
            "Cannot specify both --audio_path and --image_path. Use one modality at a time."
        )

    audio_out = None
    vision_out = None
    num_tokens = 0
    t_st_ms = 0
    t_ae_ms = 0
    t_ve_ms = 0
    audio_duration_s = 0.0

    if args.audio_path:
        audio_out, num_tokens, t_st_ms, t_ae_ms, audio_duration_s = _run_audio_pipeline(
            prog, args.audio_path
        )
    elif args.image_path:
        vision_out, num_tokens, t_ve_ms = _run_vision_pipeline(prog, args.image_path)

    ids = _build_prompt_ids(
        sp, args.prompt, bool(args.audio_path), bool(args.image_path), num_tokens
    )
    seq_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long)
    if audio_out is not None:
        hidden_size = audio_out.shape[2]
    elif vision_out is not None:
        hidden_size = vision_out.shape[2]
    else:
        hidden_size = HIDDEN_SIZE[args.variant]
    inputs_embeds = _build_inputs_embeds(
        ids, audio_out, vision_out, num_tokens, hidden_size
    )

    t_prefill_start = time.perf_counter()
    logits = td_method.execute(
        [input_ids, torch.arange(seq_len, dtype=torch.long), inputs_embeds]
    )[0]
    t_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000

    next_token = torch.argmax(logits[0, -1, :]).item()
    generated = [next_token]

    t_decode_start = time.perf_counter()
    for step in range(args.max_new_tokens - 1):
        if next_token in STOP_TOKENS:
            break
        logits = td_method.execute(
            [
                torch.tensor([[next_token]], dtype=torch.long),
                torch.tensor([seq_len + step], dtype=torch.long),
                torch.zeros(1, 1, hidden_size, dtype=torch.float32),
            ]
        )[0]
        next_token = torch.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
    t_decode_ms = (time.perf_counter() - t_decode_start) * 1000
    total_ms = (time.perf_counter() - t_total_start) * 1000

    text = sp.decode(generated)
    decode_tokens = len(generated) - 1

    logger.info(f"\nOutput: {text}")
    _log_perf_summary(
        pte_size_mb=pte_size_mb,
        seq_len=seq_len,
        decode_tokens=decode_tokens,
        t_st_ms=t_st_ms,
        t_ae_ms=t_ae_ms,
        t_ve_ms=t_ve_ms,
        t_prefill_ms=t_prefill_ms,
        t_decode_ms=t_decode_ms,
        total_ms=total_ms,
        audio_duration_s=audio_duration_s,
        has_audio=bool(args.audio_path),
    )


if __name__ == "__main__":
    main()
