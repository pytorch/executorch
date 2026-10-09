# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for Diffusers model loading and export."""

from dataclasses import dataclass, field

import torch
from executorch.exir._warnings import experimental

SUPPORTED_DTYPES = {torch.float16, torch.float32, torch.bfloat16}


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class ModelConfig:
    """Configure model inputs and component precision."""

    height: int
    width: int
    max_sequence_lengths: tuple[int, ...]
    text_encoder_dtype: torch.dtype = torch.float32
    denoiser_dtype: torch.dtype = torch.float32
    vae_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Validate model inputs and component precision."""
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
            or not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise ValueError("height and width must be positive integers")
        if not self.max_sequence_lengths or any(
            not isinstance(length, int) or isinstance(length, bool) or length <= 0
            for length in self.max_sequence_lengths
        ):
            raise ValueError("max_sequence_lengths must contain positive integers")
        for name in ("text_encoder_dtype", "denoiser_dtype", "vae_dtype"):
            if getattr(self, name) not in SUPPORTED_DTYPES:
                raise ValueError(f"{name} must be float16, float32, or bfloat16")


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class ExportConfig:
    """Configure dynamic-shape behavior during export."""

    dynamic_shapes: bool = False
    min_size: int = 256
    max_size: int = 1024

    def __post_init__(self) -> None:
        """Validate dynamic-shape bounds."""
        if (
            not isinstance(self.min_size, int)
            or isinstance(self.min_size, bool)
            or not isinstance(self.max_size, int)
            or isinstance(self.max_size, bool)
            or self.min_size <= 0
            or self.max_size < self.min_size
        ):
            raise ValueError("min_size and max_size must be positive, ordered integers")


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class DiffusionConfig:
    """Combine model and export configuration."""

    model: ModelConfig
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        """Validate constraints spanning model and export configuration."""
        if self.export.dynamic_shapes and self.export.min_size == self.export.max_size:
            raise ValueError("min_size must be less than max_size for dynamic export")
        if self.export.dynamic_shapes and not (
            self.export.min_size <= self.model.height <= self.export.max_size
            and self.export.min_size <= self.model.width <= self.export.max_size
        ):
            raise ValueError("height and width must be within dynamic-shape bounds")
