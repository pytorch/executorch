# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared implementation for Stable Diffusion pipeline specifications."""

from dataclasses import replace
from typing import Any

import torch

from ..config import DiffusionConfig
from ..wrappers import vae_scale_factor
from .base import DiffusionComponents, DiffusionModelInfo, DiffusionPipelineSpec


class StableDiffusionPipelineSpec(DiffusionPipelineSpec):
    """Shared UNet/VAE behavior for Stable Diffusion pipelines."""

    def _image_spatial_multiple(
        self, components: DiffusionComponents, scale: int
    ) -> int:
        down_block_types = getattr(
            components.require("denoiser").config,
            "down_block_types",
            None,
        )
        if not isinstance(down_block_types, (list, tuple)) or not down_block_types:
            raise ValueError("denoiser config must define down_block_types")
        return scale * 2 ** (len(down_block_types) - 1)

    def _dynamic_spatial_dims(
        self,
        components: DiffusionComponents,
        config: DiffusionConfig,
        scale: int,
    ) -> tuple[Any, Any]:
        spatial_multiple = self._image_spatial_multiple(components, scale)
        if (
            config.export.min_size % spatial_multiple
            or config.export.max_size % spatial_multiple
        ):
            raise ValueError(
                f"dynamic size bounds must be multiples of {spatial_multiple}"
            )
        height_units = torch.export.Dim(
            "height_units",
            min=config.export.min_size // spatial_multiple,
            max=config.export.max_size // spatial_multiple,
        )
        width_units = torch.export.Dim(
            "width_units",
            min=config.export.min_size // spatial_multiple,
            max=config.export.max_size // spatial_multiple,
        )
        latent_multiple = spatial_multiple // scale
        return latent_multiple * height_units, latent_multiple * width_units

    def resolve_model_info(
        self,
        components: DiffusionComponents,
        model_info: DiffusionModelInfo,
    ) -> DiffusionModelInfo:
        """Resolve guidance and spatial behavior from the UNet and VAE."""
        guidance_input_dim = getattr(
            components.require("denoiser").config,
            "time_cond_proj_dim",
            None,
        )
        scale = vae_scale_factor(components.require("vae"))
        return replace(
            model_info,
            vae_scale_factor=scale,
            dynamic_spatial_multiple=self._image_spatial_multiple(components, scale),
            supports_cfg=self.supports_cfg and guidance_input_dim is None,
            accepts_guidance_input=(
                self.accepts_guidance_input or guidance_input_dim is not None
            ),
            guidance_input_dim=guidance_input_dim or 0,
        )

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
