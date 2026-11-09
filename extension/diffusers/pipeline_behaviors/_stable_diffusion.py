# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared runtime behavior for Stable Diffusion pipelines."""

from typing import Any

import torch

from .base import DiffusionPipelineBehavior


class StableDiffusionPipelineBehavior(DiffusionPipelineBehavior):
    """Shared UNet/VAE behavior for Stable Diffusion pipelines."""

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
        """Create NCHW latent noise for a Stable Diffusion UNet."""
        latents = torch.randn(
            (
                exported_shape[0],
                exported_shape[1],
                height // vae_scale_factor,
                width // vae_scale_factor,
            ),
            generator=generator,
            dtype=dtype,
        )
        if hasattr(scheduler, "init_noise_sigma"):
            latents *= scheduler.init_noise_sigma
        return latents

    def prepare_encoded_latents(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample an encoded Gaussian VAE posterior."""
        return mean + std * torch.randn(
            mean.shape, generator=generator, dtype=mean.dtype
        )

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
        """Apply Stable Diffusion img2img timestep slicing and noise."""
        start = max(steps - min(int(steps * strength), steps), 0) * getattr(
            scheduler, "order", 1
        )
        timesteps = timesteps[start:]
        if len(timesteps) == 0:
            raise ValueError(f"strength {strength} leaves no denoising steps")
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start)
        noise = torch.randn(image_latents.shape, generator=generator, dtype=dtype)
        return scheduler.add_noise(image_latents, noise, timesteps[:1]), timesteps
