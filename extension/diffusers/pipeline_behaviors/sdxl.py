# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stable Diffusion XL runtime pipeline specification."""

import torch
from executorch.exir._warnings import experimental

from ._stable_diffusion import StableDiffusionPipelineBehavior


@experimental("This API is experimental and may change without notice.")
class SDXLPipelineBehavior(StableDiffusionPipelineBehavior):
    """Runtime behavior for Stable Diffusion XL."""

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
