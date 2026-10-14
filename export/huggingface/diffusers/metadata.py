# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build runtime metadata while exporting Diffusers pipelines."""

from executorch.extension.diffusers.metadata import DiffusionMetadata
from executorch.extension.diffusers.pipeline_behaviors import DiffusionPipelineBehavior

from .config import DiffusionConfig
from .pipeline_specs import DiffusionModelInfo


def build_diffusion_metadata(
    behavior: DiffusionPipelineBehavior,
    model_info: DiffusionModelInfo,
    config: DiffusionConfig,
) -> DiffusionMetadata:
    """Build the runtime contract from pipeline, model, and export data."""
    return DiffusionMetadata(
        pipeline=behavior.name,
        scheduler=model_info.scheduler_name,
        scheduler_prediction_type=model_info.scheduler_prediction_type,
        supports_cfg=model_info.supports_cfg,
        accepts_guidance_input=model_info.accepts_guidance_input,
        supports_negative_prompt=behavior.supports_negative_prompt,
        default_cfg_scale=(
            behavior.default_cfg_scale if model_info.supports_cfg else 1.0
        ),
        default_guidance_scale=(
            behavior.default_guidance_scale
            if model_info.accepts_guidance_input
            else None
        ),
        guidance_input_dim=model_info.guidance_input_dim,
        default_inference_steps=behavior.default_inference_steps,
        text_encoder_dtype=str(config.model.text_encoder_dtype).removeprefix("torch."),
        denoiser_dtype=str(config.model.denoiser_dtype).removeprefix("torch."),
        vae_dtype=str(config.model.vae_dtype).removeprefix("torch."),
        zero_negative_prompt=model_info.zero_negative_prompt,
        height=config.model.height,
        width=config.model.width,
        max_sequence_lengths=config.model.max_sequence_lengths,
        dynamic_shapes=config.export.dynamic_shapes,
        min_size=config.export.min_size,
        max_size=config.export.max_size,
        dynamic_spatial_multiple=model_info.dynamic_spatial_multiple,
        vae_scale_factor=model_info.vae_scale_factor,
    )
