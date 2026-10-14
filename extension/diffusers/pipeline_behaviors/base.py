# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts and data types for diffusion pipeline specifications."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from executorch.exir._warnings import experimental


@experimental("This API is experimental and may change without notice.")
class DiffusionPipelineBehavior(ABC):
    """Runtime behavior for one diffusion pipeline."""

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
