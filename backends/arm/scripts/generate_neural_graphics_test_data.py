# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate NSS and NFRU autoencoder calibration and verification data."""

from __future__ import annotations

import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.arm.scripts.neural_graphics_test_data import (
    _NFRU_INPUT_CHANNELS,
    _NSS_INPUT_CHANNELS,
    download_safetensors_prefix,
    NFRU_CALIBRATION_SAMPLES,
    nfru_input_shape,
    NFRU_INPUT_SPATIAL_SIZE,
    nfru_test_calibration_path,
    nfru_test_data_root,
    nfru_test_verification_path,
    NFRU_VERIFICATION_SAMPLES,
    NSS_CALIBRATION_SAMPLES,
    NSS_CALIBRATION_SPATIAL_SIZE,
    nss_input_shape,
    nss_test_calibration_path,
    nss_test_data_root,
    nss_test_verification_path,
    NSS_VERIFICATION_SAMPLES,
)
from safetensors.torch import save_file

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ng_model_gym.core.config.config_model import (  # type: ignore[import-not-found,import-untyped]
    ConfigModel,
)
from ng_model_gym.core.data.data_utils import (  # type: ignore[import-not-found,import-untyped]
    DataLoaderMode,
    DatasetType,
    tonemap_forward,
    ToneMapperMode,
)
from ng_model_gym.usecases.nfru.data.dataset import (  # type: ignore[import-not-found,import-untyped]
    NFRUDataset,
)
from ng_model_gym.usecases.nfru.model.nfru_v1 import (  # type: ignore[import-not-found,import-untyped]
    NFRUv1,
)
from ng_model_gym.usecases.nss.data.dataset import (  # type: ignore[import-not-found,import-untyped]
    NSSDataset,
)


_DATASET_REPO_ID = "Arm/neural-graphics-dataset"
_NSS_DATASET_REVISION = "main"
_NSS_CAPTURE_FRAMES = 99
_NSS_TRAIN_CAPTURES = (
    *range(80, 90),
    *range(91, 107),
    *range(108, 119),
)
_NSS_SOURCE_TENSORS = {
    "colour_linear",
    "depth",
    "exposure",
    "ground_truth_linear",
    "motion_lr",
    "render_size",
}
_NSS_EVALUATION_SOURCE_FRAMES = {"nss/test/test_full_resolution_sample.safetensors": 1}

EPS = 1e-7
NSS_V1_SPATIAL_MULTIPLE = 8

_NFRU_DATASET_REVISION = "main"
_NFRU_CALIBRATION_SOURCE = "nfru/train/0002.safetensors"
_NFRU_EVALUATION_SOURCE = "nfru/test/0000.safetensors"
_NFRU_SOURCE_TENSORS = {
    "DepthParams",
    "FarPlane",
    "FovY",
    "NearPlane",
    "ViewProj",
    "depth",
    "exposure",
    "infinite_zFar",
    "mv_{}_f30_m1",
    "rgb_linear",
    "sy_{}_f30_m1",
    "sy_{}_f30_p1",
}


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


def _has_samples(path: Path, num_samples: int) -> bool:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob("*.safetensors"))
    else:
        return False
    return (
        bool(files) and sum(nss_input_shape(file)[0] for file in files) == num_samples
    )


def _calibration_source_frames(num_samples: int) -> dict[str, int]:
    max_samples = len(_NSS_TRAIN_CAPTURES) * _NSS_CAPTURE_FRAMES
    if num_samples > max_samples:
        raise ValueError(
            f"Requested {num_samples} calibration samples, but only "
            f"{max_samples} are available."
        )

    source_frames = {}
    remaining = num_samples
    for capture in _NSS_TRAIN_CAPTURES:
        if remaining == 0:
            break
        frames = min(remaining, _NSS_CAPTURE_FRAMES)
        source_frames[f"nss/train/bistro/0002/{capture:04d}/0002.safetensors"] = frames
        remaining -= frames
    return source_frames


def _download_raw_sources(
    source_frames: dict[str, int], *, force_download: bool = False
) -> Path:
    for source, num_frames in source_frames.items():
        download_safetensors_prefix(
            repo_id=_DATASET_REPO_ID,
            filename=source,
            revision=_NSS_DATASET_REVISION,
            destination=_raw_source_path() / source,
            num_frames=num_frames,
            tensor_names=_NSS_SOURCE_TENSORS,
            force_download=force_download,
            preserve_source_layout=True,
        )
    return _raw_dataset_root(_raw_source_path())


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
        nss_input_shape(file_path) == (1, _NSS_INPUT_CHANNELS, *spatial_size)
        for file_path in path.glob("*.safetensors")
    )


def ensure_generated_test_calibration_dataset(
    num_samples: int = NSS_CALIBRATION_SAMPLES,
    spatial_size: tuple[int, int] = NSS_CALIBRATION_SPATIAL_SIZE,
    force_download: bool = False,
) -> Path:
    """Generate evenly distributed, test-ready NSS calibration samples."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    calibration_path = nss_test_calibration_path(num_samples, spatial_size)
    if _has_test_calibration_samples(calibration_path, num_samples, spatial_size):
        return calibration_path

    raw_root = _download_raw_sources(
        _calibration_source_frames(num_samples), force_download=force_download
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
    sample_indices = range(num_samples)
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
    if _has_samples(verification_path, NSS_VERIFICATION_SAMPLES):
        return verification_path

    raw_root = _download_raw_sources(
        _NSS_EVALUATION_SOURCE_FRAMES, force_download=force_download
    )
    _generate_verification_dataset(
        raw_root / "test",
        verification_path,
        NSS_VERIFICATION_SAMPLES,
        NSS_VERIFICATION_SAMPLES,
    )
    _delete_raw_split(raw_root, "test", "NSS_KEEP_RAW_EVALUATION_DATA")
    return verification_path


def ensure_generated_test_datasets(
    calibration_samples: int = NSS_CALIBRATION_SAMPLES,
    spatial_size: tuple[int, int] = NSS_CALIBRATION_SPATIAL_SIZE,
) -> tuple[Path, Path]:
    """Generate the NSS artifacts consumed directly by ``test_nss.py``."""

    calibration_path = ensure_generated_test_calibration_dataset(
        calibration_samples, spatial_size
    )
    verification_path = ensure_generated_verification_dataset()
    _delete_raw_sources()
    return calibration_path, verification_path


def generate_test_datasets_from_scratch(
    calibration_samples: int = NSS_CALIBRATION_SAMPLES,
    spatial_size: tuple[int, int] = NSS_CALIBRATION_SPATIAL_SIZE,
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


class _NFRUAutoencoderInputCaptured(Exception):
    pass


def _nfru_model_gym_config(src: Path) -> ConfigModel:
    config_path = files("ng_model_gym.usecases.nfru.configs").joinpath(
        "nfru_template.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        config["dataset"]["path"][split] = str(src)
    config["dataset"].update(gt_augmentation=False, align_data=True)
    config["model"]["processing_backend"] = "torch"
    return ConfigModel.model_validate(config)


def _nfru_model_gym_dataset(src: Path, loader_mode: DataLoaderMode) -> NFRUDataset:
    return NFRUDataset(_nfru_model_gym_config(src), loader_mode, DatasetType.SAFETENSOR)


def _nfru_preprocessing_model(src: Path) -> NFRUv1:
    model = NFRUv1(_nfru_model_gym_config(src)).eval()
    model.on_evaluation_start()
    return model


def _capture_nfru_autoencoder_input(
    model: NFRUv1, inputs: dict[str, torch.Tensor], random_seed: int
) -> torch.Tensor:
    captured: torch.Tensor | None = None

    def capture(_module, args):
        nonlocal captured
        captured = args[0].detach().cpu()
        raise _NFRUAutoencoderInputCaptured

    hook = model.get_neural_network().register_forward_pre_hook(capture)
    try:
        try:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(random_seed)
                model({name: value.unsqueeze(0) for name, value in inputs.items()})
        except _NFRUAutoencoderInputCaptured:
            pass
    finally:
        hook.remove()

    if captured is None:
        raise RuntimeError("Failed to capture the NFRU autoencoder input.")
    return captured


def _nfru_autoencoder_sample(
    dataset: NFRUDataset, model: NFRUv1, index: int
) -> tuple[Path, torch.Tensor]:
    inputs, _ = dataset[index]
    source = dataset.frame_indexes[index][0]
    return source, _capture_nfru_autoencoder_input(model, inputs, index)


def _nfru_metadata(src: Path, tensor: torch.Tensor, sample: int) -> dict[str, str]:
    return {
        "format": "nfru_v1_autoencoder_calibration",
        "source": str(src),
        "sample": str(sample),
        "samples": str(tensor.shape[0]),
        "shape": json.dumps(list(tensor.shape)),
        "preprocess": "nfru_v1_torch_preprocess",
        "channels": json.dumps(
            [
                "rgb_m1_mv.r",
                "rgb_m1_mv.g",
                "rgb_m1_mv.b",
                "rgb_p1_mv.r",
                "rgb_p1_mv.g",
                "rgb_p1_mv.b",
                "rgb_m1_flow.r",
                "rgb_m1_flow.g",
                "rgb_m1_flow.b",
                "rgb_p1_flow.r",
                "rgb_p1_flow.g",
                "rgb_p1_flow.b",
                "depth_m1",
                "depth_p1",
                "disocclusion_m1",
                "disocclusion_p1",
            ]
        ),
    }


def _write_nfru_tensor(src: Path, dst: Path, tensor: torch.Tensor, sample: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"input": tensor.contiguous()},
        dst,
        metadata=_nfru_metadata(src, tensor, sample),
    )


def _nfru_raw_source_path() -> Path:
    return nfru_test_data_root() / "source"


def _download_nfru_raw_source(
    source: str, num_frames: int, *, force_download: bool = False
) -> Path:
    download_safetensors_prefix(
        repo_id=_DATASET_REPO_ID,
        filename=source,
        revision=_NFRU_DATASET_REVISION,
        destination=_nfru_raw_source_path() / source,
        num_frames=num_frames,
        tensor_names=_NFRU_SOURCE_TENSORS,
        force_download=force_download,
        preserve_source_layout=True,
    )
    return _nfru_raw_source_path() / "nfru"


def _has_nfru_samples(path: Path, num_samples: int) -> bool:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob("*.safetensors"))
    else:
        return False
    return (
        bool(files) and sum(nfru_input_shape(file)[0] for file in files) == num_samples
    )


def _has_nfru_calibration_samples(path: Path, num_samples: int) -> bool:
    if not path.is_dir() or len(list(path.glob("*.safetensors"))) != num_samples:
        return False
    return all(
        nfru_input_shape(file_path)
        == (1, _NFRU_INPUT_CHANNELS, *NFRU_INPUT_SPATIAL_SIZE)
        for file_path in path.glob("*.safetensors")
    )


def ensure_generated_nfru_calibration_dataset(
    num_samples: int = NFRU_CALIBRATION_SAMPLES,
    force_download: bool = False,
) -> Path:
    """Generate evenly distributed NFRU training inputs for calibration."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    calibration_path = nfru_test_calibration_path(num_samples)
    if _has_nfru_calibration_samples(calibration_path, num_samples):
        return calibration_path

    calibration_frames = 2 * num_samples + 3
    raw_root = _download_nfru_raw_source(
        _NFRU_CALIBRATION_SOURCE,
        calibration_frames,
        force_download=force_download,
    )
    dataset = _nfru_model_gym_dataset(raw_root / "train", DataLoaderMode.TRAIN)
    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} calibration samples, but only found "
            f"{len(dataset)}."
        )

    _remove_stale_shards(calibration_path)
    calibration_path.mkdir(parents=True, exist_ok=True)
    model = _nfru_preprocessing_model(raw_root / "train")
    sample_indices = range(num_samples)
    for output_index, sample_index in enumerate(sample_indices):
        source, tensor = _nfru_autoencoder_sample(dataset, model, sample_index)
        _write_nfru_tensor(
            source,
            calibration_path / f"{output_index:04d}.safetensors",
            tensor,
            output_index,
        )

    return calibration_path


def ensure_generated_nfru_verification_dataset(
    force_download: bool = False,
) -> Path:
    """Generate a held-out deployment-resolution NFRU autoencoder input."""

    verification_path = nfru_test_verification_path()
    if _has_nfru_samples(verification_path, NFRU_VERIFICATION_SAMPLES):
        return verification_path

    raw_root = _download_nfru_raw_source(
        _NFRU_EVALUATION_SOURCE,
        2 * NFRU_VERIFICATION_SAMPLES + 3,
        force_download=force_download,
    )
    dataset = _nfru_model_gym_dataset(raw_root / "test", DataLoaderMode.TEST)
    model = _nfru_preprocessing_model(raw_root / "test")
    source, tensor = _nfru_autoencoder_sample(dataset, model, 0)
    _remove_stale_shards(verification_path)
    _write_nfru_tensor(source, verification_path / "0000.safetensors", tensor, 0)
    return verification_path


def _delete_nfru_raw_sources() -> None:
    if not os.environ.get("NFRU_KEEP_RAW_DATA"):
        shutil.rmtree(_nfru_raw_source_path(), ignore_errors=True)


def ensure_generated_nfru_test_datasets(
    calibration_samples: int = NFRU_CALIBRATION_SAMPLES,
) -> tuple[Path, Path]:
    """Generate the NFRU artifacts consumed directly by ``test_nfru.py``."""

    calibration_path = ensure_generated_nfru_calibration_dataset(calibration_samples)
    verification_path = ensure_generated_nfru_verification_dataset()
    _delete_nfru_raw_sources()
    return calibration_path, verification_path


def generate_nfru_test_datasets_from_scratch(
    calibration_samples: int = NFRU_CALIBRATION_SAMPLES,
) -> tuple[Path, Path]:
    """Download and regenerate the NFRU artifacts consumed by
    ``test_nfru.py``.
    """

    paths = (
        nfru_test_calibration_path(calibration_samples),
        nfru_test_verification_path(),
        _nfru_raw_source_path(),
    )
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    calibration_path = ensure_generated_nfru_calibration_dataset(
        calibration_samples, force_download=True
    )
    verification_path = ensure_generated_nfru_verification_dataset(force_download=True)
    _delete_nfru_raw_sources()
    return calibration_path, verification_path
