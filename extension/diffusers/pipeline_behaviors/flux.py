# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FLUX text-to-image runtime behavior."""

from typing import Any

import torch
from executorch.exir._warnings import experimental

from .base import DiffusionPipelineBehavior


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack 2x2 spatial neighborhoods into the channel dimension."""
    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("FLUX latent height and width must be even")
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // 4, channels * 4)
    )


def unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
) -> torch.Tensor:
    """Restore packed FLUX latents to the VAE's NCHW layout."""
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    batch, _, packed_channels = latents.shape
    if packed_channels % 4:
        raise ValueError("FLUX packed latent channels must be divisible by four")
    return (
        latents.reshape(
            batch, latent_height // 2, latent_width // 2, packed_channels // 4, 2, 2
        )
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, packed_channels // 4, latent_height, latent_width)
    )


def image_position_ids(height: int, width: int, device=None) -> torch.Tensor:
    """Create FLUX (type, row, column) image position IDs."""
    rows = torch.arange(height, device=device).reshape(height, 1).expand(height, width)
    columns = torch.arange(width, device=device).reshape(1, width).expand(height, width)
    return torch.stack((torch.zeros_like(rows), rows, columns), dim=-1).reshape(-1, 3)


@experimental("This API is experimental and may change without notice.")
class FluxPipelineBehavior(DiffusionPipelineBehavior):
    """Runtime behavior shared by FLUX.1-schnell and FLUX.1-dev."""

    name = "flux"
    supports_cfg = True
    accepts_guidance_input = True
    supports_negative_prompt = True
    default_cfg_scale = 1.0
    default_guidance_scale = 3.5
    default_inference_steps = 28
    default_zero_negative_prompt = False
    tokenizer_slots = ("tokenizer", "tokenizer_2")
    text_encoder_slots = ("text_encoder", "text_encoder_2")
    text_encoder_input_names = (("input_ids",), ("input_ids",))

    def assemble_conditioning(self, encoder_outputs):
        (pooled,), (prompt,) = encoder_outputs
        return prompt, pooled

    def should_use_cfg(self, cfg_scale: float, negative_prompt: str | None) -> bool:
        return cfg_scale > 1 and negative_prompt is not None

    def create_latents(
        self,
        exported_shape,
        *,
        height,
        width,
        vae_scale_factor,
        generator,
        dtype,
        scheduler,
    ):
        channels = exported_shape[-1] // 4
        noise = torch.randn(
            (1, channels, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            dtype=dtype,
        )
        return pack_latents(noise)

    def prepare_encoded_latents(self, mean, std, *, height, width, generator):
        raise ValueError("FLUX img2img is not supported")

    def prepare_img2img_latents(self, scheduler, image_latents, timesteps, **kwargs):
        raise ValueError("FLUX img2img is not supported")

    def prepare_schedule(
        self,
        scheduler: Any,
        steps: int,
        *,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        image_sequence_length = (height // vae_scale_factor // 2) * (
            width // vae_scale_factor // 2
        )
        config = scheduler.config
        base_length = getattr(config, "base_image_seq_len", 256)
        max_length = getattr(config, "max_image_seq_len", 4096)
        base_shift = getattr(config, "base_shift", 0.5)
        max_shift = getattr(config, "max_shift", 1.15)
        mu = base_shift + (image_sequence_length - base_length) * (
            max_shift - base_shift
        ) / (max_length - base_length)
        sigmas = (
            None
            if getattr(config, "use_flow_sigmas", False)
            else torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64).tolist()
        )
        scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
        return scheduler.timesteps

    def prepare_denoiser_inputs(
        self,
        latents,
        timestep,
        conditioning,
        *,
        height,
        width,
        vae_scale_factor,
        dtype,
        guidance_scale,
    ):
        prompt, pooled = conditioning
        text_ids = torch.zeros(prompt.shape[1], 3, dtype=torch.int64)
        image_ids = image_position_ids(
            height // vae_scale_factor // 2, width // vae_scale_factor // 2
        )
        inputs = (
            latents,
            timestep.to(dtype) / 1000,
            prompt,
            pooled,
            text_ids,
            image_ids,
        )
        if guidance_scale is not None:
            return (*inputs, torch.tensor([guidance_scale], dtype=dtype))
        return inputs

    def prepare_decode_latents(self, latents, *, height, width, vae_scale_factor):
        return unpack_latents(latents, height, width, vae_scale_factor)
