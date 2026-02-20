#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run exported Whisper model using ExecuTorch pybindings.

This script loads the three exported programs (encoder, cross_kv, decoder)
and performs speech-to-text transcription.

Usage:
    python -m executorch.backends.mlx.examples.whisper.run_whisper \
        --model-dir /tmp/whisper_mlx \
        --audio-file /path/to/audio.wav

    # Or use sample audio from HuggingFace:
    python -m executorch.backends.mlx.examples.whisper.run_whisper \
        --model-dir /tmp/whisper_mlx \
        --use-sample-audio

Requirements:
    pip install transformers soundfile datasets
"""

import argparse
import json
import logging
import os
import time
from typing import List, Optional

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def load_audio(
    audio_path: Optional[str],
    use_sample_audio: bool,
    processor,
) -> torch.Tensor:
    """
    Load and preprocess audio input.

    Returns:
        input_features: [1, n_mels, n_frames] tensor
    """
    if use_sample_audio:
        logger.info("Loading sample audio from HuggingFace datasets...")
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets not installed. Run: pip install datasets")
            raise

        dataset = load_dataset(
            "distil-whisper/librispeech_long",
            "clean",
            split="validation",
        )
        sample = dataset[0]["audio"]
        audio_array = sample["array"]
        sampling_rate = sample["sampling_rate"]
    else:
        if audio_path is None:
            raise ValueError(
                "Either --audio-file or --use-sample-audio must be provided"
            )

        logger.info(f"Loading audio from: {audio_path}")
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile not installed. Run: pip install soundfile")
            raise

        audio_array, sampling_rate = sf.read(audio_path)

    # Process audio to mel spectrogram
    input_features = processor(
        audio_array,
        return_tensors="pt",
        truncation=False,
        sampling_rate=sampling_rate,
    ).input_features

    # Truncate to 30 seconds (3000 frames at 100 frames/sec)
    max_frames = 3000
    if input_features.shape[2] > max_frames:
        logger.info(
            f"Truncating audio from {input_features.shape[2]} to {max_frames} frames"
        )
        input_features = input_features[:, :, :max_frames].contiguous()

    return input_features


def run_whisper_inference(  # noqa: C901
    model_dir: str,
    audio_path: Optional[str] = None,
    use_sample_audio: bool = False,
    max_new_tokens: int = 256,
    language: str = "en",
    task: str = "transcribe",
) -> str:
    """
    Run Whisper inference using exported ExecuTorch models.

    Args:
        model_dir: Directory containing encoder.pte, cross_kv.pte, decoder.pte
        audio_path: Path to audio file (WAV, MP3, etc.)
        use_sample_audio: If True, use sample audio from HuggingFace
        max_new_tokens: Maximum number of tokens to generate
        language: Language code for transcription
        task: "transcribe" or "translate"

    Returns:
        Transcribed text
    """
    from executorch.runtime import Runtime, Verification
    from transformers import AutoProcessor

    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Model: {metadata['model_id']}")

    num_layers = metadata["num_decoder_layers"]

    # Load processor
    processor_path = os.path.join(model_dir, "processor")
    logger.info(f"Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)

    # Load audio
    input_features = load_audio(audio_path, use_sample_audio, processor)
    logger.info(f"Input features shape: {input_features.shape}")

    # Cast to model dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_dtype = dtype_map.get(metadata["dtype"], torch.float32)
    input_features = input_features.to(model_dtype)
    logger.info(f"Input dtype: {input_features.dtype}")

    # Load ExecuTorch programs
    et_runtime = Runtime.get()

    logger.info("Loading encoder...")
    encoder_path = os.path.join(model_dir, "encoder.pte")
    encoder_program = et_runtime.load_program(
        encoder_path, verification=Verification.Minimal
    )
    encoder_forward = encoder_program.load_method("forward")

    logger.info("Loading cross_kv...")
    cross_kv_path = os.path.join(model_dir, "cross_kv.pte")
    cross_kv_program = et_runtime.load_program(
        cross_kv_path, verification=Verification.Minimal
    )
    cross_kv_forward = cross_kv_program.load_method("forward")

    logger.info("Loading decoder...")
    decoder_path = os.path.join(model_dir, "decoder.pte")
    decoder_program = et_runtime.load_program(
        decoder_path, verification=Verification.Minimal
    )
    decoder_forward = decoder_program.load_method("forward")

    # =========================================================================
    # Step 1: Run encoder
    # =========================================================================
    logger.info("Running encoder...")
    start_time = time.time()

    encoder_outputs = encoder_forward.execute([input_features])
    encoder_hidden_states = encoder_outputs[0]

    encoder_time = time.time() - start_time
    logger.info(f"Encoder time: {encoder_time:.3f}s")
    logger.info(f"Encoder output shape: {encoder_hidden_states.shape}")

    # =========================================================================
    # Step 2: Compute cross-attention K/V
    # =========================================================================
    logger.info("Computing cross-attention K/V...")
    start_time = time.time()

    cross_kv_outputs = cross_kv_forward.execute([encoder_hidden_states])
    # Output is (k_tuple, v_tuple) flattened: [k0, k1, ..., v0, v1, ...]
    # Each k_i, v_i has shape [B, H, T_enc, D]
    cross_k_tuple = tuple(cross_kv_outputs[:num_layers])
    cross_v_tuple = tuple(cross_kv_outputs[num_layers:])

    cross_kv_time = time.time() - start_time
    logger.info(f"Cross-KV time: {cross_kv_time:.3f}s")
    logger.info(f"Cross K/V: {num_layers} layers, each shape {cross_k_tuple[0].shape}")

    # =========================================================================
    # Step 3: Setup decoder generation
    # =========================================================================
    # Get forced decoder IDs for language/task
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
    )
    # Build forced tokens dict: position -> token_id
    forced_tokens_dict = {}
    if forced_decoder_ids is not None:
        for item in forced_decoder_ids:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pos, tok_id = item
                if tok_id is not None:
                    forced_tokens_dict[pos] = int(tok_id)

    # Start with decoder_start_token_id (start-of-transcript)
    # Get from processor.tokenizer if available, otherwise use common ID
    try:
        sot_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    except Exception:
        sot_id = 50258  # Common Whisper SOT token ID

    # Also get EOS token ID
    try:
        eos_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    except Exception:
        eos_id = 50257  # Common Whisper EOS token ID

    generated_tokens: List[int] = [sot_id]

    # =========================================================================
    # Step 4: Token-by-token decoder generation
    # =========================================================================
    logger.info(f"Generating up to {max_new_tokens} tokens...")
    decode_start = time.time()

    # Initial decoder input
    decoder_input_ids = torch.tensor([[sot_id]], dtype=torch.long)
    cache_position = torch.tensor([0], dtype=torch.long)

    # Prefill with initial token
    decoder_inputs = (
        [decoder_input_ids, cache_position] + list(cross_k_tuple) + list(cross_v_tuple)
    )
    decoder_outputs = decoder_forward.execute(decoder_inputs)
    logits = decoder_outputs[0]

    # Update cache position
    cache_position = cache_position + decoder_input_ids.shape[1]

    # Generation loop
    for _step in range(max_new_tokens):
        current_pos = cache_position.item()

        # Check for forced token at this position
        if current_pos in forced_tokens_dict:
            next_token_id = forced_tokens_dict[current_pos]
        else:
            next_token_id = torch.argmax(logits[0, -1, :]).item()

        generated_tokens.append(next_token_id)

        # Check for EOS
        if next_token_id == eos_id:
            break

        # Prepare next decoder input
        decoder_input_ids = torch.tensor([[next_token_id]], dtype=torch.long)

        # Run decoder
        decoder_inputs = (
            [decoder_input_ids, cache_position]
            + list(cross_k_tuple)
            + list(cross_v_tuple)
        )
        decoder_outputs = decoder_forward.execute(decoder_inputs)
        logits = decoder_outputs[0]

        # Update cache position
        cache_position = cache_position + 1

    decode_time = time.time() - decode_start
    tokens_generated = len(generated_tokens) - 1  # Exclude initial SOT
    tokens_per_sec = tokens_generated / decode_time if decode_time > 0 else 0

    logger.info(f"Decode time: {decode_time:.3f}s")
    logger.info(f"Tokens generated: {tokens_generated}")
    logger.info(f"Speed: {tokens_per_sec:.1f} tokens/sec")

    # Decode to text
    transcript = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return transcript


def main():
    parser = argparse.ArgumentParser(description="Run exported Whisper model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/tmp/whisper_mlx",
        help="Directory containing exported .pte files",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--use-sample-audio",
        action="store_true",
        help="Use sample audio from HuggingFace datasets",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for transcription",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: transcribe or translate",
    )

    args = parser.parse_args()

    if not args.audio_file and not args.use_sample_audio:
        logger.warning("No audio specified. Using --use-sample-audio")
        args.use_sample_audio = True

    transcript = run_whisper_inference(
        model_dir=args.model_dir,
        audio_path=args.audio_file,
        use_sample_audio=args.use_sample_audio,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        task=args.task,
    )

    print("\n" + "=" * 60)
    print("Transcript:")
    print("=" * 60)
    print(transcript)
    print("=" * 60)


if __name__ == "__main__":
    main()
