# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""Image preprocessing for Gemma 4 vision encoder.

Reimplements HF's Gemma4ImageProcessor logic: aspect-ratio-preserving resize,
patchification (16x16), and 2D position ID computation.
"""

import math

import numpy as np
import torch
from PIL import Image

PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
CELL_SIZE = PATCH_SIZE * POOLING_KERNEL_SIZE  # 48
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # 768
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def preprocess_image(
    image_path: str, max_soft_tokens: int = 280
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load and preprocess an image for the Gemma 4 vision encoder.

    Args:
        image_path: Path to image file (JPEG, PNG, etc.)
        max_soft_tokens: Max vision tokens after pooling. One of {70,140,280,560,1120}.

    Returns:
        pixel_values: (1, num_patches, 768) float32 in [0, 1]
        pixel_position_ids: (1, num_patches, 2) int64
        num_soft_tokens: actual number of vision tokens (after pooling)
    """
    if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
        raise ValueError(
            f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}"
        )

    max_patches = max_soft_tokens * POOLING_KERNEL_SIZE**2

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)  # (H, W, 3) uint8

    # Resize preserving aspect ratio
    h, w = img_array.shape[:2]
    target_h, target_w = _get_aspect_ratio_preserving_size(h, w, max_patches)

    if target_h != h or target_w != w:
        img = img.resize((target_w, target_h), Image.BICUBIC)
        img_array = np.array(img)

    # Rescale to [0, 1] and convert to CHW
    pixels = img_array.astype(np.float32) / 255.0
    pixels = np.transpose(pixels, (2, 0, 1))  # (3, H, W)

    # Patchify
    patches = _patchify(pixels)  # (num_patches, 768)

    # Position IDs
    h_patches = target_h // PATCH_SIZE
    w_patches = target_w // PATCH_SIZE
    position_ids = _compute_position_ids(h_patches, w_patches)  # (num_patches, 2)

    num_soft_tokens = patches.shape[0] // (POOLING_KERNEL_SIZE**2)

    # Pad to max_patches
    patches, position_ids = _pad_to_length(patches, position_ids, max_patches)

    pixel_values = torch.from_numpy(patches).unsqueeze(0)  # (1, max_patches, 768)
    pixel_position_ids = (
        torch.from_numpy(position_ids).unsqueeze(0).long()
    )  # (1, max_patches, 2)

    return pixel_values, pixel_position_ids, num_soft_tokens


def _get_aspect_ratio_preserving_size(
    height: int, width: int, max_patches: int
) -> tuple[int, int]:
    """Compute target size preserving aspect ratio, both dims divisible by 48."""
    target_px = max_patches * (PATCH_SIZE**2)
    factor = math.sqrt(target_px / (height * width))
    ideal_h = factor * height
    ideal_w = factor * width

    target_h = int(math.floor(ideal_h / CELL_SIZE)) * CELL_SIZE
    target_w = int(math.floor(ideal_w / CELL_SIZE)) * CELL_SIZE

    if target_h == 0 and target_w == 0:
        raise ValueError(
            f"Image {height}x{width} is too small for patch_size={PATCH_SIZE}, "
            f"pooling={POOLING_KERNEL_SIZE}"
        )

    max_side = (max_patches // POOLING_KERNEL_SIZE**2) * CELL_SIZE
    if target_h == 0:
        target_h = CELL_SIZE
        target_w = min(int(math.floor(width / height)) * CELL_SIZE, max_side)
    elif target_w == 0:
        target_w = CELL_SIZE
        target_h = min(int(math.floor(height / width)) * CELL_SIZE, max_side)

    return target_h, target_w


def _patchify(image: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) image to (num_patches, patch_dim) patches.

    Patches are extracted in row-major order (left-to-right, top-to-bottom).
    """
    c, h, w = image.shape
    ph = h // PATCH_SIZE
    pw = w // PATCH_SIZE
    # (C, ph, patch_size, pw, patch_size) -> (ph, pw, patch_size, patch_size, C) -> (ph*pw, patch_dim)
    patches = image.reshape(c, ph, PATCH_SIZE, pw, PATCH_SIZE)
    patches = patches.transpose(1, 3, 2, 4, 0)
    patches = patches.reshape(ph * pw, -1)
    return patches


def _compute_position_ids(h_patches: int, w_patches: int) -> np.ndarray:
    """Compute 2D (x, y) position IDs for patches. Returns (num_patches, 2)."""
    grid_x, grid_y = np.meshgrid(
        np.arange(w_patches), np.arange(h_patches), indexing="xy"
    )
    return np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)


def _pad_to_length(
    patches: np.ndarray, positions: np.ndarray, target_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad patches and positions to target_length along first dim."""
    current = patches.shape[0]
    if current >= target_length:
        return patches, positions
    pad_len = target_length - current
    patches = np.pad(
        patches, [(0, pad_len), (0, 0)], mode="constant", constant_values=0
    )
    positions = np.pad(
        positions, [(0, pad_len), (0, 0)], mode="constant", constant_values=-1
    )
    return patches, positions


def compute_vision_num_tokens(num_patches: int) -> int:
    """Compute output token count after pooling."""
    return num_patches // (POOLING_KERNEL_SIZE**2)
