# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import Swin2SRForImageSuperResolution

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SUPPORTED_MODELS = ("swin2sr",)


class Swin2SRWrapper(torch.nn.Module):
    def __init__(self, model: Swin2SRForImageSuperResolution):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values, return_dict=True).reconstruction


@dataclass(frozen=True)
class SuperResolutionModelBundle:
    model_name: str
    model: torch.nn.Module
    example_inputs: tuple[torch.Tensor]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    upscale: int
    window_size: int


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def list_image_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file type: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Image path does not exist: {path}")

    image_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {path}")
    return image_paths


def load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().clone()


def crop_input_tensor(
    image: torch.Tensor,
    input_height: int,
    input_width: int,
    crop_mode: str,
) -> torch.Tensor:
    height, width = image.shape[1:]
    if height < input_height or width < input_width:
        raise ValueError(
            "Image tensor is smaller than the requested crop size: "
            f"{tuple(image.shape)} vs {(input_height, input_width)}."
        )

    if height == input_height and width == input_width:
        return image

    max_top = height - input_height
    max_left = width - input_width
    if crop_mode == "random":
        top = int(torch.randint(max_top + 1, ()).item())
        left = int(torch.randint(max_left + 1, ()).item())
    elif crop_mode == "center":
        top = max_top // 2
        left = max_left // 2
    else:
        raise ValueError(f"Unsupported crop mode: {crop_mode}")

    return image[:, top : top + input_height, left : left + input_width].contiguous()


def load_calibration_inputs(
    image_path: str | Path,
    input_height: int,
    input_width: int,
    max_samples: int,
) -> list[tuple[torch.Tensor]]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    calibration_inputs = []
    for candidate in list_image_paths(image_path)[:max_samples]:
        image = load_image_tensor(candidate)
        image = crop_input_tensor(image, input_height, input_width, crop_mode="center")
        calibration_inputs.append((image.unsqueeze(0),))
    return calibration_inputs


def _collect_image_map(root: Path) -> dict[Path, Path]:
    image_map = {
        candidate.relative_to(root): candidate for candidate in list_image_paths(root)
    }
    if not image_map:
        raise ValueError(f"No supported images found in: {root}")
    return image_map


def paired_image_paths(
    lr_dir: str | Path,
    hr_dir: str | Path,
    max_samples: int | None = None,
) -> list[tuple[Path, Path]]:
    lr_root = Path(lr_dir)
    hr_root = Path(hr_dir)
    lr_map = _collect_image_map(lr_root)
    hr_map = _collect_image_map(hr_root)

    lr_keys = set(lr_map)
    hr_keys = set(hr_map)
    if lr_keys != hr_keys:
        missing_lr = sorted(str(key) for key in hr_keys - lr_keys)
        missing_hr = sorted(str(key) for key in lr_keys - hr_keys)
        details = []
        if missing_lr:
            details.append(f"Missing LR files: {missing_lr[:5]}")
        if missing_hr:
            details.append(f"Missing HR files: {missing_hr[:5]}")
        raise ValueError(
            "LR/HR directories do not contain matching files. " + " ".join(details)
        )

    pairs = [(lr_map[key], hr_map[key]) for key in sorted(lr_map)]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        pairs = pairs[:max_samples]
    if not pairs:
        raise ValueError("No paired super-resolution samples were found.")
    return pairs


def _validate_pair_shapes(
    lr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    upscale: int,
) -> None:
    if lr_tensor.shape[0] != 3 or hr_tensor.shape[0] != 3:
        raise ValueError("Only RGB LR/HR image pairs are supported.")

    expected_hr_shape = (
        hr_tensor.shape[0],
        lr_tensor.shape[1] * upscale,
        lr_tensor.shape[2] * upscale,
    )
    if tuple(hr_tensor.shape) != expected_hr_shape:
        raise ValueError(
            "HR image shape does not match the LR image and upscale factor: "
            f"LR {tuple(lr_tensor.shape)}, HR {tuple(hr_tensor.shape)}, upscale {upscale}."
        )


def crop_super_resolution_pair(
    lr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    input_height: int,
    input_width: int,
    upscale: int,
    crop_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_pair_shapes(lr_tensor, hr_tensor, upscale)

    lr_height, lr_width = lr_tensor.shape[1:]
    if lr_height < input_height or lr_width < input_width:
        raise ValueError(
            "LR image is smaller than the requested crop size: "
            f"{tuple(lr_tensor.shape)} vs {(input_height, input_width)}."
        )

    if lr_height == input_height and lr_width == input_width:
        return lr_tensor.contiguous(), hr_tensor.contiguous()

    max_top = lr_height - input_height
    max_left = lr_width - input_width
    if crop_mode == "random":
        top = int(torch.randint(max_top + 1, ()).item())
        left = int(torch.randint(max_left + 1, ()).item())
    elif crop_mode == "center":
        top = max_top // 2
        left = max_left // 2
    else:
        raise ValueError(f"Unsupported crop mode: {crop_mode}")

    hr_top = top * upscale
    hr_left = left * upscale
    return (
        lr_tensor[:, top : top + input_height, left : left + input_width].contiguous(),
        hr_tensor[
            :,
            hr_top : hr_top + input_height * upscale,
            hr_left : hr_left + input_width * upscale,
        ].contiguous(),
    )


class PairedSuperResolutionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        input_height: int,
        input_width: int,
        upscale: int,
        crop_mode: str,
        max_samples: int | None = None,
    ):
        self.pairs = paired_image_paths(lr_dir, hr_dir, max_samples=max_samples)
        self.input_height = input_height
        self.input_width = input_width
        self.upscale = upscale
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[idx]
        lr_tensor = load_image_tensor(lr_path)
        hr_tensor = load_image_tensor(hr_path)
        return crop_super_resolution_pair(
            lr_tensor,
            hr_tensor,
            self.input_height,
            self.input_width,
            self.upscale,
            self.crop_mode,
        )


def _load_checkpointed_swin2sr(
    checkpoint: str,
    checkpoint_revision: str | None,
    local_files_only: bool,
) -> Swin2SRWrapper:
    is_local_checkpoint = Path(checkpoint).expanduser().exists()
    if checkpoint_revision is None and not (local_files_only or is_local_checkpoint):
        raise ValueError(
            "--checkpoint-revision is required when --checkpoint is a remote Hugging Face model id."
        )
    if checkpoint_revision is None:
        model = Swin2SRForImageSuperResolution.from_pretrained(  # nosec B615
            checkpoint,
            local_files_only=local_files_only,
        ).eval()
    else:
        model = Swin2SRForImageSuperResolution.from_pretrained(  # nosec B615
            checkpoint,
            revision=checkpoint_revision,
            local_files_only=local_files_only,
        ).eval()
    return Swin2SRWrapper(model)


def create_model_bundle(
    model_name: str,
    input_height: int,
    input_width: int,
    checkpoint: str | None = None,
    checkpoint_revision: str | None = None,
    local_files_only: bool = False,
) -> SuperResolutionModelBundle:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS}"
        )
    if input_height <= 0 or input_width <= 0:
        raise ValueError("Input dimensions must be positive.")

    if checkpoint is None:
        raise ValueError("--checkpoint is required when --model-name=swin2sr.")
    model = _load_checkpointed_swin2sr(
        checkpoint,
        checkpoint_revision,
        local_files_only,
    )

    example_input = torch.rand((1, 3, input_height, input_width), dtype=torch.float32)
    with torch.no_grad():
        example_output = model(example_input)

    if example_output.dim() != 4:
        raise ValueError(
            f"Expected a 4D reconstruction tensor, got {tuple(example_output.shape)}."
        )

    model_impl = model.model
    return SuperResolutionModelBundle(
        model_name=model_name,
        model=model,
        example_inputs=(example_input,),
        input_shape=tuple(example_input.shape),
        output_shape=tuple(example_output.shape),
        input_dtype=str(example_input.dtype).replace("torch.", ""),
        output_dtype=str(example_output.dtype).replace("torch.", ""),
        upscale=int(model_impl.config.upscale),
        window_size=int(model_impl.config.window_size),
    )


def _gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    sigma = max(window_size / 6.0, 1e-3)

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[1]
    kernel = _gaussian_window(
        channels,
        prediction.device,
        prediction.dtype,
        prediction.shape[-2],
        prediction.shape[-1],
    )
    padding = kernel.shape[-1] // 2
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy
    )

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def _model_reconstruction(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    output = model(input_tensor)
    if hasattr(output, "reconstruction"):
        output = output.reconstruction
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    return output


def evaluate_super_resolution_model(
    model: torch.nn.Module,
    dataset: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    if hasattr(model, "eval"):
        try:
            model.eval()
        except NotImplementedError:
            pass

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_examples = 0

    with torch.no_grad():
        for lr_tensor, hr_tensor in dataset:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            prediction = _model_reconstruction(model, lr_tensor).clamp(0.0, 1.0)

            l1_values = F.l1_loss(prediction, hr_tensor, reduction="none").mean(
                dim=(1, 2, 3)
            )
            psnr_values = batch_psnr(prediction, hr_tensor)
            ssim_values = batch_ssim(prediction, hr_tensor)

            batch_size = lr_tensor.shape[0]
            total_l1 += l1_values.sum().item()
            total_psnr += psnr_values.sum().item()
            total_ssim += ssim_values.sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Evaluation dataset is empty.")

    return {
        "l1": total_l1 / total_examples,
        "psnr": total_psnr / total_examples,
        "ssim": total_ssim / total_examples,
        "num_samples": float(total_examples),
    }
