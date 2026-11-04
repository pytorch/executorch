# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared FLUX.2 text-to-image runtime behavior."""

import torch

from .base import DiffusionPipelineBehavior
from .flux import image_position_ids


def flux2_text_position_ids(length: int) -> torch.Tensor:
    positions = torch.arange(length)
    zeros = torch.zeros_like(positions)
    return torch.stack((zeros, zeros, zeros, positions), dim=-1).unsqueeze(0)


def flux2_image_position_ids(height: int, width: int) -> torch.Tensor:
    ids = image_position_ids(height, width)
    return torch.cat((ids, torch.zeros_like(ids[:, :1])), dim=-1).unsqueeze(0)


def compute_empirical_mu(image_sequence_length: int, steps: int) -> float:
    # Match the empirical shift coefficients in Diffusers' FLUX.2 schedule.
    shift_200 = 0.00016927 * image_sequence_length + 0.45666666
    if image_sequence_length > 4300:
        return shift_200
    shift_10 = 8.73809524e-05 * image_sequence_length + 1.89833333
    slope = (shift_200 - shift_10) / 190.0
    return shift_200 + slope * (steps - 200)


class _Flux2PipelineBehavior(DiffusionPipelineBehavior):
    """Shared FLUX.2 latent handling and denoiser inputs."""

    supports_cfg = False
    accepts_guidance_input = False
    supports_negative_prompt = False
    default_cfg_scale = 1.0
    default_guidance_scale = None
    default_inference_steps = 50
    default_zero_negative_prompt = False
    tokenizer_slots = ("tokenizer",)
    text_encoder_slots = ("text_encoder",)
    text_encoder_input_names = (("input_ids", "attention_mask"),)

    def assemble_conditioning(self, encoder_outputs):
        (prompt,) = encoder_outputs[0]
        return (prompt,)

    def create_latents(
        self,
        exported_shape,
        *,
        height,
        width,
        vae_scale_factor,
        generator,
        dtype,
        scheduler,
    ):
        noise = torch.randn(
            (
                1,
                exported_shape[-1],
                height // vae_scale_factor // 2,
                width // vae_scale_factor // 2,
            ),
            generator=generator,
            dtype=dtype,
        )
        return noise.flatten(2).transpose(1, 2).contiguous()

    def prepare_encoded_latents(self, mean, std, *, height, width, generator):
        raise ValueError("FLUX.2 img2img is not supported")

    def prepare_img2img_latents(self, scheduler, image_latents, timesteps, **kwargs):
        raise ValueError("FLUX.2 img2img is not supported")

    def prepare_schedule(self, scheduler, steps, *, height, width, vae_scale_factor):
        image_tokens = (height // vae_scale_factor // 2) * (
            width // vae_scale_factor // 2
        )
        sigmas = (
            None
            if getattr(scheduler.config, "use_flow_sigmas", False)
            else torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64).tolist()
        )
        scheduler.set_timesteps(
            steps, sigmas=sigmas, mu=compute_empirical_mu(image_tokens, steps)
        )
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        return scheduler.timesteps

    def prepare_denoiser_inputs(
        self,
        latents,
        timestep,
        conditioning,
        *,
        height,
        width,
        vae_scale_factor,
        dtype,
        guidance_scale,
    ):
        (prompt,) = conditioning
        inputs = (
            latents.contiguous(),
            timestep.to(dtype) / 1000,
            prompt,
            flux2_text_position_ids(prompt.shape[1]),
            flux2_image_position_ids(
                height // vae_scale_factor // 2, width // vae_scale_factor // 2
            ),
        )
        if guidance_scale is not None:
            return (*inputs, torch.tensor([guidance_scale], dtype=dtype))
        return inputs

    def prepare_decode_latents(self, latents, *, height, width, vae_scale_factor):
        return (
            latents.transpose(1, 2)
            .reshape(
                latents.shape[0],
                latents.shape[-1],
                height // vae_scale_factor // 2,
                width // vae_scale_factor // 2,
            )
            .contiguous()
        )


class Flux2PipelineBehavior(_Flux2PipelineBehavior):
    """FLUX.2-dev conditioning with a Mistral processor."""

    name = "flux2"
    accepts_guidance_input = True
    default_guidance_scale = 4.0

    def load_tokenizers(self, model_id, revision=None):
        from transformers import AutoProcessor

        return [
            AutoProcessor.from_pretrained(
                model_id, subfolder="tokenizer", revision=revision
            )
        ]

    def tokenize_prompt(self, tokenizer, prompt, *, max_length):
        from diffusers.pipelines.flux2.pipeline_flux2 import format_input

        return tokenizer.apply_chat_template(
            format_input([prompt]),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
