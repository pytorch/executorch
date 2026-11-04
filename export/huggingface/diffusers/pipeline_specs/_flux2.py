# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared export composition for FLUX.2-dev and Klein."""

from abc import abstractmethod
from dataclasses import replace

import torch

from ..wrappers import (
    Flux2Denoiser,
    Flux2TextEncoder,
    Flux2VAEDecoder,
    freeze_flux_pos_embed,
    vae_scale_factor,
)
from .base import DiffusionComponents, DiffusionPipelineExportSpec


class _Flux2PipelineExportSpec(DiffusionPipelineExportSpec):
    """Share transformer, VAE, input shapes, and checkpoint loading."""

    default_max_text_sequence_lengths = (512,)
    max_sequence_length_override_index = 0
    default_size = (1024, 1024)
    text_encoder_layers: tuple[int, ...]

    def _validate(self, components, config):
        if len(config.model.max_sequence_lengths) != 1:
            raise ValueError("FLUX.2 requires one text sequence length")
        vae = components.require("vae")
        if tuple(vae.config.patch_size) != (2, 2):
            raise ValueError("FLUX.2 requires VAE patch_size (2, 2)")
        packed_channels = int(components.require("denoiser").config.in_channels)
        if packed_channels != vae.config.latent_channels * 4:
            raise ValueError("FLUX.2 transformer and VAE latent channels must match")
        scale = vae_scale_factor(vae)
        multiple = scale * 2
        if config.model.height % multiple or config.model.width % multiple:
            raise ValueError(f"FLUX.2 height and width must be multiples of {multiple}")
        if config.export.dynamic_shapes and (
            config.export.min_size % multiple or config.export.max_size % multiple
        ):
            raise ValueError(
                f"FLUX.2 dynamic size bounds must be multiples of {multiple}"
            )
        return scale

    def wrap_components(self, components, config):
        scale = self._validate(components, config)
        transformer = components.require("denoiser")
        text_length = config.model.max_sequence_lengths[0]
        max_size = (
            config.export.max_size
            if config.export.dynamic_shapes
            else max(config.model.height, config.model.width)
        )
        # FLUX.2 invokes pos_embed separately for text and image coordinates.
        freeze_flux_pos_embed(
            transformer,
            max_positions=max(text_length, max_size // scale // 2),
        )
        return {
            "text_encoder_0": Flux2TextEncoder(
                components.require("text_encoder"), self.text_encoder_layers
            ),
            "denoise": Flux2Denoiser(transformer, transformer.config.guidance_embeds),
            "decode": Flux2VAEDecoder(
                components.require("vae"), config.model.vae_dtype
            ),
        }

    def example_inputs(self, components, config):
        scale = self._validate(components, config)
        ids = torch.zeros(1, config.model.max_sequence_lengths[0], dtype=torch.long)
        mask = torch.ones_like(ids)
        transformer = components.require("denoiser")
        prompt = torch.zeros(
            *ids.shape,
            int(transformer.config.joint_attention_dim),
            dtype=config.model.denoiser_dtype,
        )
        packed_height = config.model.height // scale // 2
        packed_width = config.model.width // scale // 2
        channels = int(transformer.config.in_channels)
        latents = torch.zeros(
            1, packed_height * packed_width, channels, dtype=config.model.denoiser_dtype
        )
        return {
            "text_encoder_0": (ids, mask),
            "denoise": self.pipeline_behavior.prepare_denoiser_inputs(
                latents,
                torch.zeros(1),
                (prompt,),
                height=config.model.height,
                width=config.model.width,
                vae_scale_factor=scale,
                dtype=config.model.denoiser_dtype,
                guidance_scale=(
                    self.pipeline_behavior.default_guidance_scale
                    if transformer.config.guidance_embeds
                    else None
                ),
            ),
            "decode": (
                torch.zeros(
                    1,
                    channels,
                    packed_height,
                    packed_width,
                    dtype=config.model.denoiser_dtype,
                ),
            ),
        }

    def dynamic_shapes(self, components, config):
        scale = self._validate(components, config)
        if not config.export.dynamic_shapes:
            return {"text_encoder_0": None, "denoise": None, "decode": None}
        multiple = scale * 2
        image_tokens = torch.export.Dim(
            "image_tokens",
            min=(config.export.min_size // multiple) ** 2,
            max=(config.export.max_size // multiple) ** 2,
        )
        packed_height = torch.export.Dim(
            "packed_height",
            min=config.export.min_size // multiple,
            max=config.export.max_size // multiple,
        )
        packed_width = torch.export.Dim(
            "packed_width",
            min=config.export.min_size // multiple,
            max=config.export.max_size // multiple,
        )
        denoise_shapes = ({1: image_tokens}, None, None, None, {1: image_tokens})
        if components.require("denoiser").config.guidance_embeds:
            denoise_shapes = (*denoise_shapes, None)
        return {
            "text_encoder_0": None,
            "denoise": denoise_shapes,
            "decode": ({2: packed_height, 3: packed_width},),
        }

    def resolve_model_info(self, components, model_info):
        guidance_embeds = bool(components.require("denoiser").config.guidance_embeds)
        scale = vae_scale_factor(components.require("vae"))
        return replace(
            model_info,
            vae_scale_factor=scale,
            dynamic_spatial_multiple=scale * 2,
            supports_cfg=self.pipeline_behavior.supports_cfg,
            accepts_guidance_input=guidance_embeds,
            guidance_input_dim=1 if guidance_embeds else 0,
            default_inference_steps=self.pipeline_behavior.default_inference_steps,
            default_guidance_scale=(
                self.pipeline_behavior.default_guidance_scale
                if guidance_embeds
                else None
            ),
        )

    @abstractmethod
    def _load_text_encoder(self, model_id, revision, dtype): ...

    def load_components(self, model_id, revision, config):
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel

        return DiffusionComponents(
            {
                "text_encoder": self._load_text_encoder(
                    model_id, revision, config.model.text_encoder_dtype
                ),
                "denoiser": Flux2Transformer2DModel.from_pretrained(
                    model_id,
                    subfolder="transformer",
                    revision=revision,
                    torch_dtype=config.model.denoiser_dtype,
                ).eval(),
                "vae": AutoencoderKLFlux2.from_pretrained(
                    model_id,
                    subfolder="vae",
                    revision=revision,
                    torch_dtype=config.model.vae_dtype,
                ).eval(),
            }
        )
