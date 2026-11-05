# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face export specification for Stable Diffusion 1.5."""

from typing import Any

import torch

from executorch.extension.diffusers.pipeline_behaviors import SD15PipelineBehavior

from ..config import DiffusionConfig

from ..wrappers import (
    CLIPLastHiddenStateTextEncoder,
    SD15Denoiser,
    vae_scale_factor,
    VAEDecoder,
    VAEEncoder,
)
from ._stable_diffusion import StableDiffusionPipelineExportSpec
from .base import DiffusionComponents


class SD15PipelineExportSpec(StableDiffusionPipelineExportSpec):
    """Compose SD1.5 components into exported PTE methods."""

    pipeline_behavior = SD15PipelineBehavior()
    default_max_sequence_lengths = (77,)
    max_sequence_length_override_index = None
    default_size = (512, 512)

    def _validate(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> int:
        scale = vae_scale_factor(components.require("vae"))
        if config.model.height % scale or config.model.width % scale:
            raise ValueError(f"height and width must be multiples of {scale}")
        if config.model.max_sequence_lengths != self.default_max_sequence_lengths:
            raise ValueError("SD 1.5 requires a sequence length of 77")
        return scale

    def wrap_components(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, torch.nn.Module]:
        self._validate(components, config)
        vae = components.require("vae")
        return {
            "text_encoder_0": CLIPLastHiddenStateTextEncoder(
                components.require("text_encoder")
            ),
            "denoise": SD15Denoiser(components.require("denoiser")),
            "decode": VAEDecoder(vae, config.model.vae_dtype),
            "encode": VAEEncoder(
                vae, config.model.vae_dtype, config.model.denoiser_dtype
            ),
        }

    def example_inputs(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        scale = self._validate(components, config)
        input_ids = torch.zeros(
            1, config.model.max_sequence_lengths[0], dtype=torch.long
        )
        with torch.no_grad():
            prompt = CLIPLastHiddenStateTextEncoder(components.require("text_encoder"))(
                input_ids
            ).to(config.model.denoiser_dtype)
        denoiser = components.require("denoiser")
        latents = torch.zeros(
            1,
            int(denoiser.config.in_channels),
            config.model.height // scale,
            config.model.width // scale,
            dtype=config.model.denoiser_dtype,
        )
        return {
            "text_encoder_0": (input_ids,),
            "denoise": self.pipeline_behavior.prepare_denoiser_inputs(
                latents,
                torch.zeros(1, dtype=torch.float32),
                (prompt,),
                height=config.model.height,
                width=config.model.width,
                dtype=config.model.denoiser_dtype,
                guidance_scale=(
                    self.pipeline_behavior.default_guidance_scale
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

    def dynamic_shapes(
        self, components: DiffusionComponents, config: DiffusionConfig
    ) -> dict[str, Any | None]:
        if not config.export.dynamic_shapes:
            return {name: None for name in self.wrap_components(components, config)}
        scale = self._validate(components, config)
        latent_height, latent_width = self._dynamic_spatial_dims(
            components, config, scale
        )
        denoise_shapes = ({2: latent_height, 3: latent_width}, None, None)
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
            "denoise": denoise_shapes,
            "decode": ({2: latent_height, 3: latent_width},),
            "encode": ({2: scale * latent_height, 3: scale * latent_width},),
        }

    def load_components(
        self,
        model_id: str,
        revision: str | None,
        config: DiffusionConfig,
    ) -> DiffusionComponents:
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel
        except ImportError as error:
            raise ImportError(
                "Loading SD 1.5 requires diffusers and transformers"
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
