# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fetch and load neural-graphics calibration and verification data."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Collection, Iterator

import torch
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session
from safetensors import safe_open


NSS_DATASET_REVISION = "main"
NSS_CALIBRATION_SAMPLES = 1024
NSS_VERIFICATION_SAMPLES = 1
NSS_CALIBRATION_SPATIAL_SIZE = (128, 128)
NFRU_DATASET_REVISION = "main"
NFRU_CALIBRATION_SAMPLES = 32
NFRU_VERIFICATION_SAMPLES = 1
NFRU_INPUT_SPATIAL_SIZE = (270, 480)
_NSS_INPUT_CHANNELS = 12
_NFRU_INPUT_CHANNELS = 16
_NSS_CALIBRATION_DIR = "calibration/nss_v1_autoencoder_cpu_calibration"
_NSS_VERIFICATION_DIR = "evaluation/nss_v1_autoencoder_cpu_evaluation"
_NFRU_CALIBRATION_DIR = "calibration/nfru_v1_autoencoder_cpu_calibration"
_NFRU_VERIFICATION_DIR = "evaluation/nfru_v1_autoencoder_cpu_evaluation"

_InputShape = tuple[int, int, int, int]
_InputShapeFn = Callable[[Path], _InputShape]
_ModelInputs = tuple[torch.Tensor]


def neural_graphics_test_data_root() -> Path:
    if "NEURAL_GRAPHICS_GENERATED_DATASET_ROOT" in os.environ:
        return Path(os.environ["NEURAL_GRAPHICS_GENERATED_DATASET_ROOT"])
    return (
        Path(__file__).resolve().parents[1] / "test" / "models" / "neural_graphics_data"
    )


def nss_test_data_root() -> Path:
    if "NSS_GENERATED_DATASET_ROOT" in os.environ:
        return Path(os.environ["NSS_GENERATED_DATASET_ROOT"])
    return neural_graphics_test_data_root() / "nss" / NSS_DATASET_REVISION


def nss_test_calibration_path(
    num_samples: int = NSS_CALIBRATION_SAMPLES,
    spatial_size: tuple[int, int] = NSS_CALIBRATION_SPATIAL_SIZE,
) -> Path:
    """Return the preprocessed calibration dataset path used by NSS tests."""

    height, width = spatial_size
    return nss_test_data_root() / (
        f"{_NSS_CALIBRATION_DIR}_{num_samples}_{height}x{width}"
    )


def nss_test_verification_path() -> Path:
    """Return the preprocessed verification dataset path used by NSS tests."""

    return nss_test_data_root() / _NSS_VERIFICATION_DIR


def nfru_test_data_root() -> Path:
    if "NFRU_GENERATED_DATASET_ROOT" in os.environ:
        return Path(os.environ["NFRU_GENERATED_DATASET_ROOT"])
    return neural_graphics_test_data_root() / "nfru" / NFRU_DATASET_REVISION


def nfru_test_calibration_path(
    num_samples: int = NFRU_CALIBRATION_SAMPLES,
) -> Path:
    """Return the preprocessed calibration dataset path used by NFRU tests."""

    height, width = NFRU_INPUT_SPATIAL_SIZE
    return nfru_test_data_root() / (
        f"{_NFRU_CALIBRATION_DIR}_{num_samples}_{height}x{width}"
    )


def nfru_test_verification_path() -> Path:
    """Return the preprocessed verification dataset path used by NFRU tests."""

    return nfru_test_data_root() / _NFRU_VERIFICATION_DIR


def _validate_input_shape(
    path: Path, expected_channels: int, model_name: str
) -> _InputShape:

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "input" not in keys:
            raise KeyError(f"{path} does not contain an `input` tensor. Found {keys}.")

        shape = tuple(handle.get_slice("input").get_shape())

    if len(shape) != 4:
        raise ValueError(f"Expected NCHW `input`, got shape {shape} in {path}.")
    if shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} {model_name} input channels, got shape "
            f"{shape}."
        )
    return shape  # type: ignore[return-value]


def nss_input_shape(path: Path) -> _InputShape:
    """Return the shape of an NSS ``input`` tensor."""

    return _validate_input_shape(path, _NSS_INPUT_CHANNELS, "NSS")


def nfru_input_shape(path: Path) -> _InputShape:
    """Return the shape of an NFRU ``input`` tensor."""

    return _validate_input_shape(path, _NFRU_INPUT_CHANNELS, "NFRU")


def _load_input_slice(
    path: Path,
    start: int,
    stop: int,
    input_shape_fn: _InputShapeFn,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)

    shape = input_shape_fn(path)
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


def _load_input_sample(
    path: Path,
    sample_index: int,
    input_shape_fn: _InputShapeFn,
) -> _ModelInputs:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")

    skipped = 0
    for file_path in _safetensor_files(path):
        file_samples = input_shape_fn(file_path)[0]
        if sample_index < skipped + file_samples:
            local_index = sample_index - skipped
            return (
                _load_input_slice(
                    file_path, local_index, local_index + 1, input_shape_fn
                ),
            )
        skipped += file_samples

    raise ValueError(
        f"Sample {sample_index} is outside the {skipped} samples in {path}."
    )


def _iter_calibration_samples(
    path: Path,
    *,
    num_samples: int,
    input_shape_fn: _InputShapeFn,
) -> Iterator[_ModelInputs]:
    """Stream calibration samples without retaining them in memory."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    files = _safetensor_files(path)
    if num_samples > len(files):
        raise ValueError(
            f"Requested {num_samples} samples from {path}, but only found {len(files)}."
        )

    for file_path in files[:num_samples]:
        yield (_load_input_slice(file_path, 0, 1, input_shape_fn),)


def load_nss_verification_inputs(
    path: Path | None = None,
    *,
    sample_index: int = 0,
) -> _ModelInputs:
    """Load one held-out NSS verification sample in ``example_inputs``
    format.
    """

    path = nss_test_verification_path() if path is None else path
    return _load_input_sample(path, sample_index, nss_input_shape)


def load_nfru_verification_inputs(
    path: Path | None = None,
    *,
    sample_index: int = 0,
) -> _ModelInputs:
    """Load one held-out NFRU verification sample in ``example_inputs``
    format.
    """

    path = nfru_test_verification_path() if path is None else path
    return _load_input_sample(path, sample_index, nfru_input_shape)


def _require_calibration_path(path: Path, model_name: str) -> Path:
    if not path.exists():
        raise RuntimeError(
            f"{model_name} calibration data is prepared by "
            "backends/arm/scripts/install_models_for_test.sh."
        )
    return path


def iter_nss_calibration_samples(
    path: Path,
    *,
    num_samples: int = 8,
) -> Iterator[_ModelInputs]:
    """Stream NSS calibration samples without retaining them in memory."""

    return _iter_calibration_samples(
        path, num_samples=num_samples, input_shape_fn=nss_input_shape
    )


def iter_nfru_calibration_samples(
    path: Path,
    *,
    num_samples: int = 8,
) -> Iterator[_ModelInputs]:
    """Stream NFRU calibration samples without retaining them in memory."""

    return _iter_calibration_samples(
        path, num_samples=num_samples, input_shape_fn=nfru_input_shape
    )


def iter_nss_test_calibration_samples(
    num_samples: int = NSS_CALIBRATION_SAMPLES,
) -> Iterator[_ModelInputs]:
    """Stream the generated NSS calibration dataset used by model tests."""

    path = _require_calibration_path(nss_test_calibration_path(), "NSS")
    return iter_nss_calibration_samples(path, num_samples=num_samples)


def iter_nfru_test_calibration_samples(
    num_samples: int = NFRU_CALIBRATION_SAMPLES,
) -> Iterator[_ModelInputs]:
    """Stream the generated NFRU calibration dataset used by model tests."""

    path = _require_calibration_path(nfru_test_calibration_path(), "NFRU")
    return iter_nfru_calibration_samples(path, num_samples=num_samples)


_HEADER_LENGTH_BYTES = 8
_COPY_CHUNK_SIZE = 1024 * 1024


def _download_range(url: str, start: int, stop: int, destination: BinaryIO) -> None:
    if stop <= start:
        return

    headers = build_hf_headers()
    headers.update(
        {
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{stop - 1}",
        }
    )
    with get_session().stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=None
    ) as response:
        if response.status_code != 206:
            raise RuntimeError(
                f"Expected an HTTP range response for {url}, got "
                f"status {response.status_code}."
            )
        written = 0
        for chunk in response.iter_bytes(_COPY_CHUNK_SIZE):
            destination.write(chunk)
            written += len(chunk)

    expected = stop - start
    if written != expected:
        raise RuntimeError(f"Expected {expected} bytes from {url}, received {written}.")


def _read_range(url: str, start: int, stop: int) -> bytes:
    destination = BytesIO()
    _download_range(url, start, stop, destination)
    return destination.getvalue()


def _local_prefix_matches(
    path: Path,
    num_frames: int,
    tensor_names: Collection[str],
    preserve_source_layout: bool,
) -> bool:
    try:
        with path.open("rb") as source:
            header_length = int.from_bytes(source.read(_HEADER_LENGTH_BYTES), "little")
            header = json.loads(source.read(header_length))
        metadata = header.get("__metadata__", {})
        expected_frames = (
            metadata.get("RangePrefixFrames")
            if preserve_source_layout
            else metadata.get("Length")
        )
        return int(expected_frames or -1) == num_frames and set(tensor_names).issubset(
            header
        )
    except (OSError, ValueError):
        return False


def _remote_header(url: str) -> tuple[dict, int]:
    header_length = int.from_bytes(_read_range(url, 0, _HEADER_LENGTH_BYTES), "little")
    if header_length <= 0:
        raise ValueError(f"Invalid safetensor header length {header_length} in {url}.")
    header = json.loads(
        _read_range(
            url,
            _HEADER_LENGTH_BYTES,
            _HEADER_LENGTH_BYTES + header_length,
        )
    )
    return header, _HEADER_LENGTH_BYTES + header_length


def _prefix_plan(
    header: dict,
    source_data_offset: int,
    num_frames: int,
    tensor_names: Collection[str],
    preserve_source_layout: bool = False,
) -> tuple[bytes, list[tuple[int, int, int]], int]:
    metadata = dict(header.get("__metadata__", {}))
    source_frames = int(metadata["Length"])
    if num_frames <= 0 or num_frames > source_frames:
        raise ValueError(
            f"Requested {num_frames} frames from a {source_frames}-frame safetensor."
        )

    missing = set(tensor_names).difference(header)
    if missing:
        raise KeyError(f"Missing tensors: {sorted(missing)}")

    output_metadata = dict(metadata)
    if preserve_source_layout:
        output_metadata["RangePrefixFrames"] = str(num_frames)
    else:
        output_metadata["Length"] = str(num_frames)
    output_header: dict = {"__metadata__": output_metadata}
    source_ranges: list[tuple[int, int, int]] = []
    output_offset = 0
    for name, descriptor in header.items():
        if name == "__metadata__" or name not in tensor_names:
            continue

        shape = list(descriptor["shape"])
        if not shape or shape[0] != source_frames:
            raise ValueError(
                f"Tensor {name} has shape {shape}; expected a leading frame "
                f"dimension of {source_frames}."
            )

        source_start, source_stop = descriptor["data_offsets"]
        source_bytes = source_stop - source_start
        if source_bytes % source_frames:
            raise ValueError(
                f"Tensor {name} byte size {source_bytes} is not divisible by "
                f"its {source_frames} frames."
            )

        downloaded_bytes = source_bytes // source_frames * num_frames
        output_bytes = source_bytes if preserve_source_layout else downloaded_bytes
        if not preserve_source_layout:
            shape[0] = num_frames
        output_header[name] = {
            "dtype": descriptor["dtype"],
            "shape": shape,
            "data_offsets": [output_offset, output_offset + output_bytes],
        }
        source_ranges.append(
            (
                source_data_offset + source_start,
                source_data_offset + source_start + downloaded_bytes,
                output_offset,
            )
        )
        output_offset += output_bytes

    encoded_header = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    return (
        len(encoded_header).to_bytes(_HEADER_LENGTH_BYTES, "little") + encoded_header,
        source_ranges,
        output_offset,
    )


def download_safetensors_prefix(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    destination: Path,
    num_frames: int,
    tensor_names: Collection[str],
    force_download: bool = False,
    preserve_source_layout: bool = False,
) -> Path:
    """Download selected tensors for the first ``num_frames`` frames."""

    if (
        destination.is_file()
        and not force_download
        and _local_prefix_matches(
            destination, num_frames, tensor_names, preserve_source_layout
        )
    ):
        return destination

    url = hf_hub_url(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    header, source_data_offset = _remote_header(url)
    output_header, source_ranges, output_size = _prefix_plan(
        header,
        source_data_offset,
        num_frames,
        tensor_names,
        preserve_source_layout,
    )
    if not source_ranges:
        raise ValueError("At least one tensor must be selected.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        with partial.open("wb") as output:
            output.write(output_header)
            output.truncate(len(output_header) + output_size)

        def download_range(source_range: tuple[int, int, int]) -> None:
            start, stop, output_offset = source_range
            with partial.open("r+b") as output:
                output.seek(len(output_header) + output_offset)
                _download_range(url, start, stop, output)

        with ThreadPoolExecutor(max_workers=min(4, len(source_ranges))) as executor:
            list(executor.map(download_range, source_ranges))
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)

    return destination
