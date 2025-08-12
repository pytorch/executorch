# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download


def download_and_convert_hf_checkpoint(
    repo_id: str, convert_weights: Callable[[str, str], None]
) -> str:
    """
    Downloads and converts to Meta format a HuggingFace checkpoint.

    Args:
        repo_id: Id of the HuggingFace repo, e.g. "Qwen/Qwen2.5-1.5B".
        convert_weights: Weight conversion function taking in path to the downloaded HuggingFace
            files and the desired output path.

    Returns:
        The output path of the Meta checkpoint converted from HuggingFace.
    """

    # Build cache path.
    cache_subdir = "meta_checkpoints"
    cache_dir = Path.home() / ".cache" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use repo name to name the converted file.
    model_name = repo_id.replace("/", "_")
    converted_path = cache_dir / f"{model_name}.pth"

    if converted_path.exists():
        print(f"✔ Using cached converted model: {str(converted_path)}")
        return str(converted_path)

    # 1. Download weights from Hugging Face.
    print("⬇ Downloading and converting checkpoint...")
    checkpoint_path = snapshot_download(
        repo_id=repo_id,
    )

    # 2. Convert weights to Meta format.
    convert_weights(checkpoint_path, str(converted_path))
    return str(converted_path)
