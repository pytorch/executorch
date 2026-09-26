# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate NSS autoencoder calibration and verification data."""

from __future__ import annotations

import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    nss_input_shape,
    nss_test_calibration_path,
    nss_test_data_root,
    nss_test_verification_path,
)
from safetensors.torch import save_file

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from ng_model_gym.core.config.config_model import (  # type: ignore[import-not-found,import-untyped]
    ConfigModel,
)
from ng_model_gym.core.data.data_utils import (  # type: ignore[import-not-found,import-untyped]
    DataLoaderMode,
    DatasetType,
    tonemap_forward,
    ToneMapperMode,
)
from ng_model_gym.usecases.nss.data.dataset import (  # type: ignore[import-not-found,import-untyped]
    NSSDataset,
)


_DATASET_REPO_ID = "Arm/neural-graphics-dataset"
_CALIBRATION_SOURCE_ALLOW_PATTERNS = [
    "train/**/*.safetensors",
    "nss/train/**/*.safetensors",
]
_EVALUATION_SOURCE_ALLOW_PATTERNS = [
    "test/test_full_resolution_sample.safetensors",
    "nss/test/test_full_resolution_sample.safetensors",
]

EPS = 1e-7
NSS_V1_SPATIAL_MULTIPLE = 8


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)
    return torch.sum(rgb * weights, dim=1, keepdim=True)


def _motion_detector(
    motion_lr: torch.Tensor, render_size: torch.Tensor
) -> torch.Tensor:
    # render_size is stored as [height, width], matching the dataset writer.
    size = render_size.to(dtype=torch.float32).view(-1, 2, 1, 1)
    motion_norm = motion_lr.to(dtype=torch.float32) / torch.clamp(size, min=1.0)
    motion_length = torch.linalg.vector_norm(motion_norm, dim=1, keepdim=True)

    pix_min = torch.linalg.vector_norm(
        1.0 / torch.clamp(size, min=1.0), dim=1
    ).unsqueeze(1)
    pix_max = torch.linalg.vector_norm(
        200.0 / torch.clamp(size, min=1.0), dim=1
    ).unsqueeze(1)
    detector = (torch.clamp(motion_length, pix_min, pix_max) - pix_min) / torch.clamp(
        pix_max - pix_min, min=EPS
    )
    return torch.sqrt(torch.clamp(detector, min=0.0))


def _depth_edge(depth: torch.Tensor) -> torch.Tensor:
    dx = F.pad(torch.abs(depth[..., :, 1:] - depth[..., :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(depth[..., 1:, :] - depth[..., :-1, :]), (0, 0, 0, 1))
    return torch.clamp((dx + dy) * 100.0, 0.0, 1.0)


def _reflect_pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = NSS_V1_SPATIAL_MULTIPLE,
) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")


def _model_gym_dataset(src: Path) -> NSSDataset:
    config_path = files("ng_model_gym.usecases.nss.configs").joinpath(
        "nss_v1_template.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        config["dataset"]["path"][split] = str(src)
    config["dataset"].update(
        exposure=None,
        tonemapper=ToneMapperMode.KARIS.value,
        gt_augmentation=False,
    )
    params = ConfigModel.model_validate(config)
    return NSSDataset(params, DataLoaderMode.TEST, DatasetType.SAFETENSOR)


def _make_autoencoder_input(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    colour_tm = current["colour"]
    exposure = current["exposure"]
    _, _, h, w = colour_tm.shape
    same_sequence = previous is not None and torch.equal(
        current["seq"], previous["seq"]
    )

    history_linear = torch.zeros_like(current["colour_linear"])
    if previous is not None and same_sequence:
        history_linear = previous["ground_truth_linear"]
        if history_linear.shape[-2:] != (h, w):
            history_linear = F.interpolate(
                history_linear,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
    history_tm = tonemap_forward(history_linear * exposure, mode=ToneMapperMode.KARIS)

    motion_signal = _motion_detector(current["motion_lr"], current["render_size"])
    luma = _luminance(colour_tm)
    previous_luma = torch.zeros_like(luma)
    feedback = torch.zeros((1, 4, h, w), dtype=torch.float32)
    if previous is not None and same_sequence:
        previous_luma = _luminance(previous["colour"])
        previous_luma_derivative = torch.clamp(previous_luma, 0.0, 1.0)
        feedback[:, 0:1] = _motion_detector(
            previous["motion_lr"], previous["render_size"]
        )
        feedback[:, 1:2] = previous_luma_derivative
        feedback[:, 2:3] = previous_luma
        feedback[:, 3:4] = _depth_edge(previous["depth"])
    luma_derivative = torch.clamp(torch.abs(luma - previous_luma), 0.0, 1.0)

    autoencoder_input = torch.cat(
        [
            _reflect_pad_to_multiple(history_tm),
            _reflect_pad_to_multiple(colour_tm),
            _reflect_pad_to_multiple(motion_signal),
            _reflect_pad_to_multiple(feedback),
            _reflect_pad_to_multiple(luma_derivative),
        ],
        dim=1,
    )
    return autoencoder_input.to(torch.float16)


def _autoencoder_sample(dataset: NSSDataset, index: int) -> torch.Tensor:
    current = dataset[index][0]
    previous = dataset[index - 1][0] if index > 0 else None
    return _make_autoencoder_input(current, previous)


def _metadata(
    src: Path, tensor: torch.Tensor, shard: int | None = None
) -> dict[str, str]:
    metadata = {
        "format": "nss_v1_autoencoder_calibration",
        "source": str(src),
        "samples": str(tensor.shape[0]),
        "shape": json.dumps(list(tensor.shape)),
        "spatial_multiple": str(NSS_V1_SPATIAL_MULTIPLE),
        "preprocess": "cpu_approximation_of_nss_v1_slang_pre_process",
        "channels": json.dumps(
            [
                "history.r",
                "history.g",
                "history.b",
                "colour.r",
                "colour.g",
                "colour.b",
                "motion_detector",
                "feedback.r",
                "feedback.g",
                "feedback.b",
                "feedback.a",
                "luma_derivative",
            ]
        ),
    }
    if shard is not None:
        metadata["shard"] = str(shard)
    return metadata


def _write_tensor(
    src: Path, dst: Path, tensor: torch.Tensor, shard: int | None = None
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"input": tensor.contiguous()}, dst, metadata=_metadata(src, tensor, shard)
    )


def _remove_stale_shards(dst: Path) -> None:
    if not dst.exists():
        return
    if not dst.is_dir():
        raise NotADirectoryError(f"Expected shard output directory, got {dst}")
    for path in dst.glob("*.safetensors"):
        path.unlink()


def _generate_verification_dataset(
    src: Path,
    dst: Path,
    num_samples: int,
    shard_size: int,
) -> None:
    dataset = _model_gym_dataset(src)
    sample_limit = min(num_samples, len(dataset))
    _remove_stale_shards(dst)
    dst.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    tensors: list[torch.Tensor] = []
    sources: list[Path] = []
    for index in range(sample_limit):
        tensors.append(_autoencoder_sample(dataset, index))
        sources.append(dataset.frame_indexes[index][0])
        if len(tensors) < shard_size and index + 1 < sample_limit:
            continue
        shard = torch.cat(tensors, dim=0)
        _write_tensor(
            sources[0],
            dst / f"{shard_idx:04d}.safetensors",
            shard,
            shard_idx,
        )
        shard_idx += 1
        tensors.clear()
        sources.clear()


def _raw_source_path() -> Path:
    return nss_test_data_root() / "source"


def _raw_dataset_root(snapshot_path: Path) -> Path:
    if (snapshot_path / "nss" / "train").is_dir() or (
        snapshot_path / "nss" / "test"
    ).is_dir():
        return snapshot_path / "nss"
    return snapshot_path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _has_safetensors(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.glob("*.safetensors")))


def _download_raw_sources(
    allow_patterns: list[str], *, force_download: bool = False
) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=_DATASET_REPO_ID,
            repo_type="dataset",
            revision="5039ce015d7c877980fad44f87893fc5ac0927e2",
            allow_patterns=allow_patterns,
            local_dir=_raw_source_path(),
            force_download=force_download,
        )
    )
    return _raw_dataset_root(snapshot)


def _delete_raw_split(raw_root: Path, split: str, keep_env: str) -> None:
    if not os.environ.get(keep_env):
        shutil.rmtree(raw_root / split, ignore_errors=True)


def _delete_raw_sources() -> None:
    if os.environ.get("NSS_KEEP_RAW_TRAIN_DATA") or os.environ.get(
        "NSS_KEEP_RAW_EVALUATION_DATA"
    ):
        return
    shutil.rmtree(_raw_source_path(), ignore_errors=True)


def _has_test_calibration_samples(
    path: Path, num_samples: int, spatial_size: tuple[int, int]
) -> bool:
    if not path.is_dir() or len(list(path.glob("*.safetensors"))) != num_samples:
        return False
    return all(
        nss_input_shape(file_path)[1:] == (12, *spatial_size)
        for file_path in path.glob("*.safetensors")
    )


def ensure_generated_test_calibration_dataset(
    num_samples: int = 3663,
    spatial_size: tuple[int, int] = (128, 128),
    force_download: bool = False,
) -> Path:
    """Generate evenly distributed, test-ready NSS calibration samples."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    calibration_path = nss_test_calibration_path(num_samples, spatial_size)
    if _has_test_calibration_samples(calibration_path, num_samples, spatial_size):
        return calibration_path

    raw_root = _download_raw_sources(
        _CALIBRATION_SOURCE_ALLOW_PATTERNS, force_download=force_download
    )
    dataset = _model_gym_dataset(raw_root / "train")
    total_samples = len(dataset)
    if num_samples > total_samples:
        raise ValueError(
            f"Requested {num_samples} calibration samples, but only found "
            f"{total_samples}."
        )

    _remove_stale_shards(calibration_path)
    calibration_path.mkdir(parents=True, exist_ok=True)
    sample_indices = (
        [0]
        if num_samples == 1
        else [
            index * (total_samples - 1) // (num_samples - 1)
            for index in range(num_samples)
        ]
    )
    for output_index, sample_index in enumerate(sample_indices):
        tensor = _autoencoder_sample(dataset, sample_index)
        if tensor.shape[-2:] != spatial_size:
            tensor = F.interpolate(
                tensor.to(torch.float32),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ).to(torch.float16)
        _write_tensor(
            dataset.frame_indexes[sample_index][0],
            calibration_path / f"{output_index:04d}.safetensors",
            tensor,
            output_index,
        )

    _delete_raw_split(raw_root, "train", "NSS_KEEP_RAW_TRAIN_DATA")
    return calibration_path


def ensure_generated_verification_dataset(
    force_download: bool = False,
) -> Path:
    """Generate the held-out NSS verification input without calibration data."""

    verification_path = nss_test_verification_path()
    if _has_safetensors(verification_path):
        return verification_path

    raw_root = _download_raw_sources(
        _EVALUATION_SOURCE_ALLOW_PATTERNS, force_download=force_download
    )
    _generate_verification_dataset(
        raw_root / "test",
        verification_path,
        _env_int("NSS_GENERATED_EVALUATION_SAMPLES", 1),
        _env_int("NSS_GENERATED_EVALUATION_SHARD_SIZE", 10),
    )
    _delete_raw_split(raw_root, "test", "NSS_KEEP_RAW_EVALUATION_DATA")
    return verification_path


def ensure_generated_test_datasets(
    calibration_samples: int = 3663,
    spatial_size: tuple[int, int] = (128, 128),
) -> tuple[Path, Path]:
    """Generate the NSS artifacts consumed directly by ``test_nss.py``."""

    calibration_path = ensure_generated_test_calibration_dataset(
        calibration_samples, spatial_size
    )
    verification_path = ensure_generated_verification_dataset()
    _delete_raw_sources()
    return calibration_path, verification_path


def generate_test_datasets_from_scratch(
    calibration_samples: int = 3663,
    spatial_size: tuple[int, int] = (128, 128),
) -> tuple[Path, Path]:
    """Download and regenerate the NSS artifacts consumed by ``test_nss.py``."""

    calibration_path = nss_test_calibration_path(calibration_samples, spatial_size)
    verification_path = nss_test_verification_path()
    for path in (calibration_path, verification_path, _raw_source_path()):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    calibration_path = ensure_generated_test_calibration_dataset(
        calibration_samples,
        spatial_size,
        force_download=True,
    )
    verification_path = ensure_generated_verification_dataset(force_download=True)
    _delete_raw_sources()
    return calibration_path, verification_path
