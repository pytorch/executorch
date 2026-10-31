# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face export specification for FLUX.1."""

from dataclasses import replace
from typing import Any

import torch

from executorch.extension.diffusers.pipeline_behaviors import FluxPipelineBehavior

from ..config import DiffusionConfig
from ..wrappers import (
    CLIPPooledTextEncoder,
    FluxDenoiser,
    freeze_flux_pos_embed,
    T5TextEncoder,
    vae_scale_factor,
    VAEDecoder,
)
from .base import DiffusionComponents, DiffusionModelInfo, DiffusionPipelineExportSpec


class FluxPipelineExportSpec(DiffusionPipelineExportSpec):
    """Compose FLUX text encoders, transformer, and VAE for export."""

    pipeline_behavior = FluxPipelineBehavior()
    default_max_text_sequence_lengths = (77, 512)
    max_sequence_length_override_index = 1
    default_size = (1024, 1024)

    def _validate(self, components, config) -> int:
        if (
            len(config.model.max_sequence_lengths) != 2
            or config.model.max_sequence_lengths[0] != 77
        ):
            raise ValueError("FLUX requires CLIP length 77 and one T5 sequence length")
        scale = vae_scale_factor(components.require("vae"))
        spatial_multiple = scale * 2
        if (
            config.model.height % spatial_multiple
            or config.model.width % spatial_multiple
        ):
            raise ValueError(
                f"FLUX height and width must be multiples of {spatial_multiple}"
            )
        if config.export.dynamic_shapes and (
            config.export.min_size % spatial_multiple
            or config.export.max_size % spatial_multiple
        ):
            raise ValueError(
                f"FLUX dynamic size bounds must be multiples of {spatial_multiple}"
            )
        return scale

    def wrap_components(self, components, config):
        scale = self._validate(components, config)
        transformer = components.require("denoiser")
        vae = components.require("vae")
        freeze_flux_pos_embed(
            transformer,
            max_positions=(
                config.export.max_size
                if config.export.dynamic_shapes
                else max(config.model.height, config.model.width)
            )
            // scale
            // 2,
        )
        return {
            "text_encoder_0": CLIPPooledTextEncoder(components.require("text_encoder")),
            "text_encoder_1": T5TextEncoder(components.require("text_encoder_2")),
            "denoise": FluxDenoiser(
                transformer, bool(getattr(transformer.config, "guidance_embeds", False))
            ),
            "decode": VAEDecoder(vae, config.model.vae_dtype),
        }

    def example_inputs(self, components, config):
        scale = self._validate(components, config)
        clip_ids = torch.zeros(1, 77, dtype=torch.long)
        t5_ids = torch.zeros(1, config.model.max_sequence_lengths[1], dtype=torch.long)
        with torch.no_grad():
            pooled = CLIPPooledTextEncoder(components.require("text_encoder"))(
                clip_ids
            ).to(config.model.denoiser_dtype)
            prompt = T5TextEncoder(components.require("text_encoder_2"))(t5_ids).to(
                config.model.denoiser_dtype
            )
        transformer = components.require("denoiser")
        packed_channels = int(transformer.config.in_channels)
        sequence_length = (config.model.height // scale // 2) * (
            config.model.width // scale // 2
        )
        latents = torch.zeros(
            1, sequence_length, packed_channels, dtype=config.model.denoiser_dtype
        )
        inputs = self.pipeline_behavior.prepare_denoiser_inputs(
            latents,
            torch.zeros(1),
            (prompt, pooled),
            height=config.model.height,
            width=config.model.width,
            vae_scale_factor=scale,
            dtype=config.model.denoiser_dtype,
            guidance_scale=(
                self.pipeline_behavior.default_guidance_scale
                if getattr(transformer.config, "guidance_embeds", False)
                else None
            ),
        )
        vae_channels = packed_channels // 4
        return {
            "text_encoder_0": (clip_ids,),
            "text_encoder_1": (t5_ids,),
            "denoise": inputs,
            "decode": (
                torch.zeros(
                    1,
                    vae_channels,
                    config.model.height // scale,
                    config.model.width // scale,
                    dtype=config.model.denoiser_dtype,
                ),
            ),
        }

    def dynamic_shapes(self, components, config):
        scale = self._validate(components, config)
        modules = self.wrap_components(components, config)
        if not config.export.dynamic_shapes:
            return {name: None for name in modules}
        packed_multiple = scale * 2
        image_tokens = torch.export.Dim(
            "image_tokens",
            min=(config.export.min_size // packed_multiple) ** 2,
            max=(config.export.max_size // packed_multiple) ** 2,
        )
        latent_height = torch.export.Dim(
            "latent_height",
            min=config.export.min_size // scale,
            max=config.export.max_size // scale,
        )
        latent_width = torch.export.Dim(
            "latent_width",
            min=config.export.min_size // scale,
            max=config.export.max_size // scale,
        )
        denoise_shapes = (
            {1: image_tokens},
            None,
            None,
            None,
            None,
            {0: image_tokens},
        )
        if getattr(components.require("denoiser").config, "guidance_embeds", False):
            denoise_shapes = (*denoise_shapes, None)
        return {
            "text_encoder_0": None,
            "text_encoder_1": None,
            "denoise": denoise_shapes,
            "decode": ({2: latent_height, 3: latent_width},),
        }

    def resolve_model_info(self, components, model_info):
        scale = vae_scale_factor(components.require("vae"))
        guidance_embeds = bool(
            getattr(components.require("denoiser").config, "guidance_embeds", False)
        )
        return replace(
            model_info,
            vae_scale_factor=scale,
            dynamic_spatial_multiple=scale * 2,
            supports_cfg=True,
            accepts_guidance_input=guidance_embeds,
            guidance_input_dim=1 if guidance_embeds else 0,
            default_inference_steps=28 if guidance_embeds else 4,
            default_guidance_scale=3.5 if guidance_embeds else 0.0,
        )

    def load_components(self, model_id, revision, config):
        try:
            from diffusers import AutoencoderKL, FluxTransformer2DModel
            from transformers import CLIPTextModel, T5EncoderModel
        except ImportError as error:
            raise ImportError(
                "Loading FLUX requires diffusers and transformers"
            ) from error
        common: dict[str, Any] = {"revision": revision}
        return DiffusionComponents(
            {
                "text_encoder": CLIPTextModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    torch_dtype=config.model.text_encoder_dtype,
                    **common,
                ).eval(),
                "text_encoder_2": T5EncoderModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder_2",
                    torch_dtype=config.model.text_encoder_dtype,
                    **common,
                ).eval(),
                "denoiser": FluxTransformer2DModel.from_pretrained(
                    model_id,
                    subfolder="transformer",
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
