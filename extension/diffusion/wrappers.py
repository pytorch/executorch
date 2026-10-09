# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
import torch.nn.functional as F


def vae_scale_factor(vae: torch.nn.Module) -> int:
    """Derive the image-to-latent spatial scale from the VAE configuration."""
    return 2 ** (len(vae.config.block_out_channels) - 1)


def _guidance_scale_embedding(
    guidance_scale: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    """Match Diffusers' sinusoidal embedding of guidance_scale - 1."""
    scale = (guidance_scale.float() - 1) * 1000
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=scale.device)
        * -(math.log(10000.0) / (half_dim - 1))
    )
    embedding = scale[:, None] * frequencies[None, :]
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class CLIPPenultimateTextEncoder(torch.nn.Module):
    """Return CLIP's penultimate hidden state and optional projected pool."""

    def __init__(self, model: torch.nn.Module, *, return_pooled: bool) -> None:
        super().__init__()
        self.model = model
        self.return_pooled = return_pooled

    def forward(self, input_ids: torch.Tensor):
        output = self.model(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[-2]
        if not self.return_pooled:
            return hidden
        pooled = getattr(output, "text_embeds", None)
        if pooled is None:
            pooled = output.pooler_output
        return hidden, pooled


class CLIPLastHiddenStateTextEncoder(torch.nn.Module):
    """Return the final CLIP hidden state used by Stable Diffusion 1.x."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


class SD15Denoiser(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.guidance_input_dim = getattr(model.config, "time_cond_proj_dim", None)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        guidance_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timestep_cond = (
            _guidance_scale_embedding(guidance_scale, self.guidance_input_dim).to(
                latents.dtype
            )
            if guidance_scale is not None
            else None
        )
        return self.model(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeddings,
            timestep_cond=timestep_cond,
        ).sample


class SDXLDenoiser(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.guidance_input_dim = getattr(model.config, "time_cond_proj_dim", None)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        pooled_embedding: torch.Tensor,
        time_ids: torch.Tensor,
        guidance_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timestep_cond = (
            _guidance_scale_embedding(guidance_scale, self.guidance_input_dim).to(
                latents.dtype
            )
            if guidance_scale is not None
            else None
        )
        return self.model(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeddings,
            added_cond_kwargs={
                "text_embeds": pooled_embedding,
                "time_ids": time_ids,
            },
            timestep_cond=timestep_cond,
        ).sample


class VAEDecoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self.scaling_factor = model.config.scaling_factor
        self.shift_factor = getattr(model.config, "shift_factor", None) or 0.0
        if compute_dtype == torch.float16 and getattr(
            model.config, "force_upcast", False
        ):
            warnings.warn(
                "The VAE declares force_upcast=True and may overflow or produce "
                "NaNs with vae_dtype=float16.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (
            latents.to(self.compute_dtype) / self.scaling_factor + self.shift_factor
        )
        return (self.model.decode(latents).sample / 2 + 0.5).clamp(0, 1)


class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        compute_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self.scaling_factor = model.config.scaling_factor
        self.shift_factor = getattr(model.config, "shift_factor", None) or 0.0

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.model.encode(
            image.to(self.compute_dtype) * 2 - 1
        ).latent_dist
        mean = (distribution.mean - self.shift_factor) * self.scaling_factor
        std = distribution.std * self.scaling_factor
        return mean.to(self.output_dtype), std.to(self.output_dtype)
