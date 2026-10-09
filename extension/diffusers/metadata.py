# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constant-method metadata for experimental Diffusers programs."""

import math
from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

import torch

from .config import DiffusionConfig

if TYPE_CHECKING:
    from .pipeline_specs import DiffusionModelInfo, DiffusionPipelineSpec

METADATA_METHODS = {
    "pipeline": "get_diffusion_pipeline",
    "scheduler": "get_diffusion_scheduler",
    "scheduler_prediction_type": "get_diffusion_scheduler_prediction_type",
    "supports_cfg": "get_diffusion_supports_cfg",
    "accepts_guidance_input": "get_diffusion_accepts_guidance_input",
    "supports_negative_prompt": "get_diffusion_supports_negative_prompt",
    "default_cfg_scale": "get_diffusion_default_cfg_scale",
    "default_guidance_scale": "get_diffusion_default_guidance_scale",
    "guidance_input_dim": "get_diffusion_guidance_input_dim",
    "default_inference_steps": "get_diffusion_default_inference_steps",
    "text_encoder_dtype": "get_diffusion_text_encoder_dtype",
    "denoiser_dtype": "get_diffusion_denoiser_dtype",
    "vae_dtype": "get_diffusion_vae_dtype",
    "zero_negative_prompt": "get_diffusion_zero_negative_prompt",
    "max_sequence_lengths": "get_diffusion_max_sequence_lengths",
    "height": "get_diffusion_height",
    "width": "get_diffusion_width",
    "dynamic_shapes": "get_diffusion_dynamic_shapes",
    "min_size": "get_diffusion_min_size",
    "max_size": "get_diffusion_max_size",
    "dynamic_spatial_multiple": "get_diffusion_dynamic_spatial_multiple",
    "vae_scale_factor": "get_diffusion_vae_scale_factor",
}


@dataclass(frozen=True)
class DiffusionMetadata:
    """Runtime contract embedded in an exported diffusion program."""

    pipeline: str
    scheduler: str
    scheduler_prediction_type: str | None
    supports_cfg: bool
    accepts_guidance_input: bool
    supports_negative_prompt: bool
    default_cfg_scale: float
    default_guidance_scale: float | None
    guidance_input_dim: int
    default_inference_steps: int
    text_encoder_dtype: str
    denoiser_dtype: str
    vae_dtype: str
    zero_negative_prompt: bool
    height: int
    width: int
    max_sequence_lengths: tuple[int, ...]
    dynamic_shapes: bool = False
    min_size: int = 0
    max_size: int = 0
    dynamic_spatial_multiple: int = 1
    vae_scale_factor: int = 8

    def to_constant_methods(self) -> dict[str, object]:
        """Return metadata encoded as ExecuTorch constant methods."""
        values: dict[str, object] = {
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "scheduler_prediction_type": self.scheduler_prediction_type or "",
            "supports_cfg": int(self.supports_cfg),
            "accepts_guidance_input": int(self.accepts_guidance_input),
            "supports_negative_prompt": int(self.supports_negative_prompt),
            "default_cfg_scale": self.default_cfg_scale,
            "default_guidance_scale": (
                self.default_guidance_scale
                if self.default_guidance_scale is not None
                else math.nan
            ),
            "guidance_input_dim": self.guidance_input_dim,
            "default_inference_steps": self.default_inference_steps,
            "text_encoder_dtype": self.text_encoder_dtype,
            "denoiser_dtype": self.denoiser_dtype,
            "vae_dtype": self.vae_dtype,
            "zero_negative_prompt": int(self.zero_negative_prompt),
            "max_sequence_lengths": torch.tensor(
                self.max_sequence_lengths, dtype=torch.int32
            ),
            "height": self.height,
            "width": self.width,
            "dynamic_shapes": int(self.dynamic_shapes),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "dynamic_spatial_multiple": self.dynamic_spatial_multiple,
            "vae_scale_factor": self.vae_scale_factor,
        }
        return {METADATA_METHODS[key]: value for key, value in values.items()}

    @classmethod
    def from_constant_method_values(
        cls, values: Mapping[str, object]
    ) -> "DiffusionMetadata":
        """Construct metadata from values read from constant methods."""
        max_sequence_lengths = values["max_sequence_lengths"]
        if (
            not isinstance(max_sequence_lengths, torch.Tensor)
            or max_sequence_lengths.ndim != 1
        ):
            raise ValueError("max_sequence_lengths metadata must be a rank-one tensor")
        lengths = tuple(int(value) for value in max_sequence_lengths.tolist())
        if not lengths or min(lengths) <= 0:
            raise ValueError(
                "max_sequence_lengths metadata must contain positive values"
            )
        return cls(
            pipeline=str(values["pipeline"]),
            scheduler=str(values["scheduler"]),
            scheduler_prediction_type=(
                str(values["scheduler_prediction_type"]) or None
            ),
            supports_cfg=bool(int(values["supports_cfg"])),
            accepts_guidance_input=bool(int(values["accepts_guidance_input"])),
            supports_negative_prompt=bool(int(values["supports_negative_prompt"])),
            default_cfg_scale=float(values["default_cfg_scale"]),
            default_guidance_scale=(
                None
                if math.isnan(float(values["default_guidance_scale"]))
                else float(values["default_guidance_scale"])
            ),
            guidance_input_dim=int(values["guidance_input_dim"]),
            default_inference_steps=int(values["default_inference_steps"]),
            text_encoder_dtype=str(values["text_encoder_dtype"]),
            denoiser_dtype=str(values["denoiser_dtype"]),
            vae_dtype=str(values["vae_dtype"]),
            zero_negative_prompt=bool(int(values["zero_negative_prompt"])),
            height=int(values["height"]),
            width=int(values["width"]),
            dynamic_shapes=bool(int(values["dynamic_shapes"])),
            min_size=int(values["min_size"]),
            max_size=int(values["max_size"]),
            dynamic_spatial_multiple=int(values["dynamic_spatial_multiple"]),
            vae_scale_factor=int(values["vae_scale_factor"]),
            max_sequence_lengths=lengths,
        )


def build_diffusion_metadata(
    spec: "DiffusionPipelineSpec",
    model_info: "DiffusionModelInfo",
    config: DiffusionConfig,
) -> DiffusionMetadata:
    """Build the runtime contract from pipeline, model, and export data."""
    return DiffusionMetadata(
        pipeline=spec.name,
        scheduler=model_info.scheduler_name,
        scheduler_prediction_type=model_info.scheduler_prediction_type,
        supports_cfg=model_info.supports_cfg,
        accepts_guidance_input=model_info.accepts_guidance_input,
        supports_negative_prompt=spec.supports_negative_prompt,
        default_cfg_scale=(spec.default_cfg_scale if model_info.supports_cfg else 1.0),
        default_guidance_scale=(
            spec.default_guidance_scale if model_info.accepts_guidance_input else None
        ),
        guidance_input_dim=model_info.guidance_input_dim,
        default_inference_steps=spec.default_inference_steps,
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
