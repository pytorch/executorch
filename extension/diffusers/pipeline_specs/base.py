# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts and data types for diffusion pipeline specifications."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from executorch.exir._warnings import experimental

from ..config import DiffusionConfig


@experimental("This API is experimental and may change without notice.")
class DiffusionPipelineSpec(ABC):
    """Contract between diffusion pipelines, exporters and runners."""

    name: str
    supports_cfg: bool
    accepts_guidance_input: bool
    supports_negative_prompt: bool
    default_cfg_scale: float
    default_guidance_scale: float | None
    default_inference_steps: int
    default_zero_negative_prompt: bool
    tokenizer_slots: tuple[str, ...]
    text_encoder_slots: tuple[str, ...]
    text_encoder_input_names: tuple[tuple[str, ...], ...]
    default_max_sequence_lengths: tuple[int, ...]
    max_sequence_length_override_index: int | None
    default_size: tuple[int, int]

    @abstractmethod
    def wrap_components(
        self, components: "DiffusionComponents", config: DiffusionConfig
    ) -> dict[str, torch.nn.Module]:
        """Wrap loaded components as the methods exported into the PTE."""
        ...

    @abstractmethod
    def example_inputs(
        self, components: "DiffusionComponents", config: DiffusionConfig
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """Create example inputs for every exported method."""
        ...

    @abstractmethod
    def dynamic_shapes(
        self, components: "DiffusionComponents", config: DiffusionConfig
    ) -> dict[str, Any | None]:
        """Describe dynamic input dimensions for every exported method."""
        ...

    @abstractmethod
    def resolve_model_info(
        self,
        components: "DiffusionComponents",
        model_info: "DiffusionModelInfo",
    ) -> "DiffusionModelInfo":
        """Resolve model-specific runtime capabilities from components."""
        ...

    @abstractmethod
    def assemble_conditioning(
        self, encoder_outputs: tuple[tuple[torch.Tensor, ...], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Convert text encoder outputs into denoiser conditioning tensors."""
        ...

    def should_use_cfg(self, cfg_scale: float, negative_prompt: str | None) -> bool:
        """Return whether generation should run unconditional conditioning."""
        return cfg_scale > 1

    @abstractmethod
    def create_latents(
        self,
        exported_shape: tuple[int, ...],
        *,
        height: int,
        width: int,
        vae_scale_factor: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        scheduler: Any,
    ) -> torch.Tensor:
        """Create initial noise in the denoiser's latent layout."""
        ...

    @abstractmethod
    def prepare_encoded_latents(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample encoded VAE latents and convert to denoiser layout."""
        ...

    @abstractmethod
    def prepare_img2img_latents(
        self,
        scheduler: Any,
        image_latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        steps: int,
        strength: float,
        generator: torch.Generator,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select img2img timesteps and add noise to encoded latents."""
        ...

    def prepare_schedule(
        self,
        scheduler: Any,
        steps: int,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Configure the scheduler and return its inference timesteps."""
        scheduler.set_timesteps(steps)
        return scheduler.timesteps

    def prepare_model_input(
        self,
        scheduler: Any,
        latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scheduler-specific denoiser input scaling when available."""
        if hasattr(scheduler, "scale_model_input"):
            return scheduler.scale_model_input(latents, timestep)
        return latents

    @abstractmethod
    def prepare_denoiser_inputs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: tuple[torch.Tensor, ...],
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
        guidance_scale: float | None,
    ) -> tuple[torch.Tensor, ...]:
        """Assemble one invocation of the exported denoiser method."""
        ...

    def combine_predictions(
        self,
        predictions: tuple[torch.Tensor, ...],
        *,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Combine one prediction or an unconditional/conditional CFG pair."""
        if len(predictions) == 1:
            return predictions[0]
        if len(predictions) != 2:
            raise ValueError("CFG expects one or two predictions")
        unconditional, conditional = predictions
        return unconditional + cfg_scale * (conditional - unconditional)

    def prepare_decode_latents(
        self,
        latents: torch.Tensor,
        *,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        """Convert denoiser latents to the VAE decoder layout."""
        return latents

    @abstractmethod
    def load_components(
        self,
        model_id: str,
        revision: str | None = None,
        config: DiffusionConfig | None = None,
    ) -> "DiffusionComponents":
        """Load model components required by this pipeline specification."""
        ...


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class DiffusionComponents:
    """Components loaded for a diffusion pipeline."""

    models: Mapping[str, torch.nn.Module]

    def require(self, slot: str) -> torch.nn.Module:
        """Return a required component or raise a descriptive error."""
        try:
            return self.models[slot]
        except KeyError as error:
            raise ValueError(
                f"missing required diffusion component {slot!r}"
            ) from error


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


@experimental("This API is experimental and may change without notice.")
@dataclass(frozen=True)
class ResolvedDiffusionPipeline:
    """Bind pipeline behavior to facts from a particular model."""

    spec: DiffusionPipelineSpec
    model_info: DiffusionModelInfo
