# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Load generated NSS autoencoder calibration and verification data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


NSS_DATASET_REVISION = "main"
_TEST_CALIBRATION_DIR = "calibration/nss_v1_autoencoder_cpu_calibration"
_VERIFICATION_DIR = "evaluation/nss_v1_autoencoder_cpu_evaluation"


def nss_test_data_root() -> Path:
    if "NSS_GENERATED_DATASET_ROOT" in os.environ:
        return Path(os.environ["NSS_GENERATED_DATASET_ROOT"])
    else:
        return (
            Path(__file__).resolve().parents[1]
            / "test"
            / "models"
            / "nss_data"
            / NSS_DATASET_REVISION
        )


def nss_test_calibration_path(
    num_samples: int = 3663, spatial_size: tuple[int, int] = (128, 128)
) -> Path:
    """Return the preprocessed calibration dataset path used by NSS tests."""

    height, width = spatial_size
    return nss_test_data_root() / (
        f"{_TEST_CALIBRATION_DIR}_{num_samples}_{height}x{width}"
    )


def nss_test_verification_path() -> Path:
    """Return the preprocessed verification dataset path used by NSS tests."""

    return nss_test_data_root() / _VERIFICATION_DIR


def nss_input_shape(path: Path) -> tuple[int, int, int, int]:
    """Return the shape of the ``input`` tensor in a safetensors file."""

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "input" not in keys:
            raise KeyError(f"{path} does not contain an `input` tensor. Found {keys}.")

        shape = tuple(handle.get_slice("input").get_shape())

    if len(shape) != 4:
        raise ValueError(f"Expected NCHW `input`, got shape {shape} in {path}.")
    if shape[1] != 12:
        raise ValueError(f"Expected 12 NSS input channels, got shape {shape}.")
    return shape  # type: ignore[return-value]


def _load_input_slice(path: Path, start: int, stop: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)

    shape = nss_input_shape(path)
    if start < 0 or stop <= start or stop > shape[0]:
        raise ValueError(f"Invalid slice [{start}:{stop}] for shape {shape}.")

    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = (
            handle.get_slice("input")[start:stop].to(dtype=torch.float32).contiguous()
        )

    return tensor.to(memory_format=torch.channels_last)


def _safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if files:
            return files
    raise FileNotFoundError(f"No safetensors found at {path}")


def _load_sample(path: Path, start: int = 0) -> tuple[torch.Tensor]:
    if start < 0:
        raise ValueError("start must be non-negative.")

    skipped = 0
    for file_path in _safetensor_files(path):
        file_samples = nss_input_shape(file_path)[0]
        if start < skipped + file_samples:
            local_index = start - skipped
            return (_load_input_slice(file_path, local_index, local_index + 1),)
        skipped += file_samples

    raise ValueError(f"Sample {start} is outside the {skipped} samples in {path}.")


def iter_calibration_samples(
    path: Path,
    *,
    num_samples: int = 8,
) -> Iterator[tuple[torch.Tensor]]:
    """Stream NSS calibration samples without retaining them in memory."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    files = _safetensor_files(path)
    if num_samples > len(files):
        raise ValueError(
            f"Requested {num_samples} samples from {path}, but only found {len(files)}."
        )

    for file_path in files[:num_samples]:
        yield (_load_input_slice(file_path, 0, 1),)


def load_verification_inputs(
    path: Path | None = None,
    *,
    start: int = 0,
) -> tuple[torch.Tensor]:
    """Load one held-out verification sample in ``example_inputs`` format."""

    path = nss_test_verification_path() if path is None else path
    return _load_sample(path, start)
