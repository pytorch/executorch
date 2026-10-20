# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import json
import os
import sys

from contextlib import redirect_stdout
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download


def download_hf_snapshot(
    repo_id: str,
    revision: str,
    expected_sha256: str | None = None,
    metadata_path: Path | None = None,
) -> str:
    """Download a pinned Safetensors checkpoint and optionally verify its hash."""
    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["*.safetensors", "*.json"],
        token=False,
    )
    if expected_sha256:
        checksum = hashlib.sha256()
        with (Path(snapshot) / "model.safetensors").open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                checksum.update(chunk)
        if metadata_path:
            metadata = (
                json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            )
            metadata.update(
                expected_checkpoint_sha256=expected_sha256,
                observed_checkpoint_sha256=checksum.hexdigest(),
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if checksum.hexdigest() != expected_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: {checksum.hexdigest()}")
    return snapshot


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
    home_dir = Path(os.environ.get("HF_HOME", Path.home()))
    cache_dir = home_dir / ".cache" / cache_subdir
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HuggingFace checkpoints, optionally converting to Meta format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Download a pinned snapshot."
    )
    snapshot_parser.add_argument("repo_id")
    snapshot_parser.add_argument("--revision", required=True)
    snapshot_parser.add_argument("--expected-sha256")
    snapshot_parser.add_argument("--metadata-path", type=Path)
    convert_parser = subparsers.add_parser(
        "download-and-convert", help="Download and cache a converted checkpoint."
    )
    convert_parser.add_argument("repo_id")
    convert_parser.add_argument(
        "--converter",
        required=True,
        metavar="MODULE:FUNCTION",
        help="Conversion callable accepting (input_dir, output_file).",
    )
    args = parser.parse_args()
    if args.command == "snapshot":
        checkpoint_path = download_hf_snapshot(
            args.repo_id, args.revision, args.expected_sha256, args.metadata_path
        )
    else:
        converter = EntryPoint(name="converter", value=args.converter, group="").load()
        with redirect_stdout(sys.stderr):
            checkpoint_path = download_and_convert_hf_checkpoint(
                args.repo_id, converter
            )
    print(checkpoint_path)
