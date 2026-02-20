#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared argument definitions for Whisper export and run scripts.
"""

import argparse
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def add_export_args(parser: argparse.ArgumentParser) -> None:
    """Add common export arguments for Whisper scripts."""
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--max-decoder-seq-len",
        type=int,
        default=256,
        help="Maximum decoder sequence length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )
    from executorch.backends.mlx.examples.quantization import add_quantization_args

    add_quantization_args(parser)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add common runtime arguments for Whisper scripts."""
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace model ID (used to load processor)",
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
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Input dtype (must match the dtype used during export)",
    )


def load_audio(
    audio_path: Optional[str],
    use_sample_audio: bool,
    processor,
) -> torch.Tensor:
    """Load and preprocess audio input.

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
