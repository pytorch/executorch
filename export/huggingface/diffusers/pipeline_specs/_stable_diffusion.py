# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared export behavior for Stable Diffusion pipelines."""

from dataclasses import replace
from typing import Any

import torch

from ..config import DiffusionConfig

from ..wrappers import vae_scale_factor
from .base import DiffusionComponents, DiffusionModelInfo, DiffusionPipelineExportSpec


class StableDiffusionPipelineExportSpec(DiffusionPipelineExportSpec):
    """Shared UNet and VAE export behavior."""

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
            supports_cfg=(
                self.pipeline_behavior.supports_cfg and guidance_input_dim is None
            ),
            accepts_guidance_input=(
                self.pipeline_behavior.accepts_guidance_input
                or guidance_input_dim is not None
            ),
            guidance_input_dim=guidance_input_dim or 0,
        )
