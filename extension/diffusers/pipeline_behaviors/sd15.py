# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable Diffusion 1.5 runtime pipeline specification."""

import torch
from executorch.exir._warnings import experimental

from ._stable_diffusion import StableDiffusionPipelineBehavior


@experimental("This API is experimental and may change without notice.")
class SD15PipelineBehavior(StableDiffusionPipelineBehavior):
    """Runtime behavior for Stable Diffusion 1.5."""

    name = "sd15"
    supports_cfg = True
    accepts_guidance_input = False
    supports_negative_prompt = True
    default_cfg_scale = 7.5
    default_guidance_scale = 7.5
    default_inference_steps = 50
    default_zero_negative_prompt = False
    tokenizer_slots = ("tokenizer",)
    text_encoder_slots = ("text_encoder",)
    text_encoder_input_names = (("input_ids",),)

    def assemble_conditioning(
        self, encoder_outputs: tuple[tuple[torch.Tensor, ...], ...]
    ) -> tuple[torch.Tensor, ...]:
        return (encoder_outputs[0][0],)

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
        inputs = (latents, timestep, *conditioning)
        if guidance_scale is None:
            return inputs
        return (*inputs, torch.tensor([guidance_scale], dtype=torch.float32))
