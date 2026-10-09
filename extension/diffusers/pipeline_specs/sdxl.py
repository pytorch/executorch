# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable Diffusion XL pipeline specification."""

from typing import Any

import torch
from executorch.exir._warnings import experimental

from ..config import DiffusionConfig
from ..wrappers import (
    CLIPPenultimateTextEncoder,
    SDXLDenoiser,
    vae_scale_factor,
    VAEDecoder,
    VAEEncoder,
)
from ._stable_diffusion import StableDiffusionPipelineSpec
from .base import DiffusionComponents


@experimental("This API is experimental and may change without notice.")
class SDXLPipelineSpec(StableDiffusionPipelineSpec):
    """Export and runtime behavior for Stable Diffusion XL."""

    name = "sdxl"
    supports_cfg = True
    accepts_guidance_input = False
    supports_negative_prompt = True
    default_cfg_scale = 5.0
    default_guidance_scale = 5.0
    default_inference_steps = 50
    default_zero_negative_prompt = True
    tokenizer_slots = ("tokenizer", "tokenizer_2")
    text_encoder_slots = ("text_encoder", "text_encoder_2")
    text_encoder_input_names = (("input_ids",), ("input_ids",))
    default_max_sequence_lengths = (77, 77)
    max_sequence_length_override_index = None
    default_size = (1024, 1024)

    def _validate(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> int:
        scale = vae_scale_factor(components.require("vae"))
        if config.model.height <= 0 or config.model.width <= 0:
            raise ValueError("height and width must be positive")
        if config.model.height % scale or config.model.width % scale:
            raise ValueError(f"height and width must be multiples of {scale}")
        if config.model.max_sequence_lengths != self.default_max_sequence_lengths:
            raise ValueError("SDXL requires sequence lengths of (77, 77)")
        return scale

    def wrap_components(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, torch.nn.Module]:
        self._validate(components, config)
        text_encoder_0 = components.require(self.text_encoder_slots[0])
        text_encoder_1 = components.require(self.text_encoder_slots[1])
        denoiser = components.require("denoiser")
        vae = components.require("vae")
        return {
            "text_encoder_0": CLIPPenultimateTextEncoder(
                text_encoder_0, return_pooled=False
            ),
            "text_encoder_1": CLIPPenultimateTextEncoder(
                text_encoder_1, return_pooled=True
            ),
            "denoise": SDXLDenoiser(denoiser),
            "decode": VAEDecoder(vae, config.model.vae_dtype),
            "encode": VAEEncoder(
                vae, config.model.vae_dtype, config.model.denoiser_dtype
            ),
        }

    def example_inputs(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        scale = self._validate(components, config)
        ids_0 = torch.zeros(1, config.model.max_sequence_lengths[0], dtype=torch.long)
        ids_1 = torch.zeros(1, config.model.max_sequence_lengths[1], dtype=torch.long)
        text_encoder_0 = components.require(self.text_encoder_slots[0])
        text_encoder_1 = components.require(self.text_encoder_slots[1])
        denoiser = components.require("denoiser")
        with torch.no_grad():
            hidden_0 = CLIPPenultimateTextEncoder(text_encoder_0, return_pooled=False)(
                ids_0
            )
            hidden_1, pooled = CLIPPenultimateTextEncoder(
                text_encoder_1, return_pooled=True
            )(ids_1)
        prompt = torch.cat((hidden_0, hidden_1), dim=-1).to(config.model.denoiser_dtype)
        pooled = pooled.to(config.model.denoiser_dtype)
        latents = torch.zeros(
            1,
            int(denoiser.config.in_channels),
            config.model.height // scale,
            config.model.width // scale,
            dtype=config.model.denoiser_dtype,
        )
        return {
            "text_encoder_0": (ids_0,),
            "text_encoder_1": (ids_1,),
            "denoise": self.prepare_denoiser_inputs(
                latents,
                torch.zeros(1, dtype=torch.float32),
                (prompt, pooled),
                height=config.model.height,
                width=config.model.width,
                dtype=config.model.denoiser_dtype,
                guidance_scale=(
                    self.default_guidance_scale
                    if getattr(denoiser.config, "time_cond_proj_dim", None) is not None
                    else None
                ),
            ),
            "decode": (latents,),
            "encode": (
                torch.zeros(
                    1,
                    3,
                    config.model.height,
                    config.model.width,
                    dtype=config.model.vae_dtype,
                ),
            ),
        }

    def assemble_conditioning(
        self, encoder_outputs: tuple[tuple[torch.Tensor, ...], ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (hidden_0,), (hidden_1, pooled) = encoder_outputs
        return torch.cat((hidden_0, hidden_1), dim=-1), pooled

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
        # SDXL time_ids: original size, crop coordinates, and target size.
        inputs = (
            latents,
            timestep,
            *conditioning,
            torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype),
        )
        if guidance_scale is None:
            return inputs
        return (*inputs, torch.tensor([guidance_scale], dtype=torch.float32))

    def dynamic_shapes(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, Any | None]:
        if not config.export.dynamic_shapes:
            return {name: None for name in self.wrap_components(components, config)}
        scale = self._validate(components, config)
        latent_height, latent_width = self._dynamic_spatial_dims(
            components, config, scale
        )
        denoise_shapes = (
            {2: latent_height, 3: latent_width},
            None,
            None,
            None,
            None,
        )
        if (
            getattr(
                components.require("denoiser").config,
                "time_cond_proj_dim",
                None,
            )
            is not None
        ):
            denoise_shapes = (*denoise_shapes, None)
        return {
            "text_encoder_0": None,
            "text_encoder_1": None,
            "denoise": denoise_shapes,
            "decode": ({2: latent_height, 3: latent_width},),
            "encode": ({2: scale * latent_height, 3: scale * latent_width},),
        }

    @staticmethod
    def load_components(
        model_id: str,
        revision: str | None = None,
        config: DiffusionConfig | None = None,
    ) -> DiffusionComponents:
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTextModelWithProjection
        except ImportError as error:
            raise ImportError(
                "Loading SDXL requires diffusers and transformers"
            ) from error
        if config is None:
            raise ValueError("export config is required to load SDXL components")
        common: dict[str, Any] = {"revision": revision}
        return DiffusionComponents(
            {
                "text_encoder": CLIPTextModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    torch_dtype=config.model.text_encoder_dtype,
                    **common,
                ).eval(),
                "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
                    model_id,
                    subfolder="text_encoder_2",
                    torch_dtype=config.model.text_encoder_dtype,
                    **common,
                ).eval(),
                "denoiser": UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    torch_dtype=config.model.denoiser_dtype,
                    **common,
                ).eval(),
                "vae": AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                    torch_dtype=config.model.vae_dtype,
                    **common,
                ).eval(),
            }
        )
