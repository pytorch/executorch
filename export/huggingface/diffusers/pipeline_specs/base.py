# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts and data types for Hugging Face diffusion export."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from executorch.exir._warnings import experimental

from executorch.extension.diffusers.pipeline_behaviors import DiffusionPipelineBehavior

from ..config import DiffusionConfig


@dataclass(frozen=True)
class DiffusionComponents:
    """Components loaded from one Diffusers pipeline."""

    models: Mapping[str, torch.nn.Module]

    def require(self, slot: str) -> torch.nn.Module:
        try:
            return self.models[slot]
        except KeyError as error:
            raise ValueError(
                f"missing required diffusion component {slot!r}"
            ) from error


class DiffusionPipelineExportSpec(ABC):
    """Hugging Face loading and graph-export contract for one pipeline."""

    pipeline_behavior: DiffusionPipelineBehavior
    default_max_sequence_lengths: tuple[int, ...]
    max_sequence_length_override_index: int | None
    default_size: tuple[int, int]

    @abstractmethod
    def wrap_components(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, torch.nn.Module]: ...

    @abstractmethod
    def example_inputs(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, tuple[torch.Tensor, ...]]: ...

    @abstractmethod
    def dynamic_shapes(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, Any | None]: ...

    @abstractmethod
    def resolve_model_info(
        self,
        components: DiffusionComponents,
        model_info: DiffusionModelInfo,
    ) -> DiffusionModelInfo: ...

    @abstractmethod
    def load_components(
        self,
        model_id: str,
        revision: str | None,
        config: DiffusionConfig,
    ) -> DiffusionComponents: ...


@dataclass(frozen=True)
class ResolvedDiffusionPipeline:
    """Bind export and runtime behavior to facts from one model."""

    export_spec: DiffusionPipelineExportSpec
    model_info: DiffusionModelInfo

    @property
    def behavior(self) -> DiffusionPipelineBehavior:
        return self.export_spec.pipeline_behavior


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class DiffusionModelInfo:
    """Facts discovered from a particular diffusion model."""

    scheduler_name: str
    max_sequence_lengths: tuple[int, ...]
    zero_negative_prompt: bool
    scheduler_prediction_type: str | None = None
    vae_scale_factor: int = 8
    dynamic_spatial_multiple: int = 1
    supports_cfg: bool = True
    accepts_guidance_input: bool = False
    guidance_input_dim: int = 0

    def __post_init__(self) -> None:
        """Validate discovered model facts."""
        if not self.scheduler_name:
            raise ValueError("scheduler_name must not be empty")
        if not self.max_sequence_lengths or min(self.max_sequence_lengths) <= 0:
            raise ValueError("max_sequence_lengths must contain positive values")
        if not isinstance(self.zero_negative_prompt, bool):
            raise ValueError("zero_negative_prompt must be a boolean")
        if self.vae_scale_factor <= 0:
            raise ValueError("vae_scale_factor must be positive")
        if self.dynamic_spatial_multiple <= 0:
            raise ValueError("dynamic_spatial_multiple must be positive")
        if self.guidance_input_dim < 0:
            raise ValueError("guidance_input_dim must not be negative")
