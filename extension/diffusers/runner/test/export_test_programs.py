# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the program fixtures of the C++ runner tests.

Metadata-only programs for the metadata reader, and tiny SD1.5 and FLUX.1
programs with the tokenizer and scheduler files of a Hugging Face snapshot for
the end-to-end runner test. generate_goldens.py runs the Python runner on the same fixtures.
"""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from executorch.exir import to_edge
from executorch.export.huggingface.diffusers.config import (
    DiffusionConfig,
    ExportConfig,
    ModelConfig,
)
from executorch.export.huggingface.diffusers.export import (
    export_programs,
    lower_programs,
    write_pte,
)
from executorch.export.huggingface.diffusers.pipeline_specs import (
    DiffusionComponents,
    DiffusionModelInfo,
    FluxPipelineExportSpec,
    ResolvedDiffusionPipeline,
    SD15PipelineExportSpec,
)
from executorch.extension.diffusers.metadata import DiffusionMetadata

PROGRAMS = {
    "flux": DiffusionMetadata(
        pipeline="flux",
        scheduler="FlowMatchEulerDiscreteScheduler",
        scheduler_prediction_type=None,
        supports_cfg=True,
        accepts_guidance_input=True,
        supports_negative_prompt=True,
        default_cfg_scale=1.0,
        default_guidance_scale=3.5,
        guidance_input_dim=0,
        default_inference_steps=28,
        text_encoder_dtype="bfloat16",
        denoiser_dtype="bfloat16",
        vae_dtype="float32",
        zero_negative_prompt=False,
        height=1024,
        width=768,
        max_sequence_lengths=(77, 512),
        dynamic_shapes=True,
        min_size=256,
        max_size=1024,
        dynamic_spatial_multiple=16,
        vae_scale_factor=8,
    ),
    "sd15": DiffusionMetadata(
        pipeline="sd15",
        scheduler="PNDMScheduler",
        scheduler_prediction_type="epsilon",
        supports_cfg=True,
        accepts_guidance_input=False,
        supports_negative_prompt=True,
        default_cfg_scale=7.5,
        default_guidance_scale=None,
        guidance_input_dim=0,
        default_inference_steps=50,
        text_encoder_dtype="float32",
        denoiser_dtype="float16",
        vae_dtype="float16",
        zero_negative_prompt=False,
        height=512,
        width=512,
        max_sequence_lengths=(77,),
    ),
}


class Identity(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def write_metadata_programs(outdir: Path) -> None:
    exported = torch.export.export(Identity(), (torch.ones(1),), strict=True)
    for name, metadata in PROGRAMS.items():
        program = to_edge(
            exported, constant_methods=metadata.to_constant_methods()
        ).to_executorch()
        (outdir / f"DiffusionMetadata_{name}.pte").write_bytes(program.buffer)
    bare = to_edge(exported).to_executorch()
    (outdir / "DiffusionMetadata_bare.pte").write_bytes(bare.buffer)


# Toy SD1.5 and FLUX.1 components whose image depends on the first prompt
# tokens, the latents, the timestep and the guidance input, strongly enough that
# a CFG, scheduler or conditioning mistake shows in the pixels.
class TextEncoder(torch.nn.Module):
    def forward(self, input_ids):
        hidden = input_ids.float().unsqueeze(-1) * 0.1 + torch.arange(8) * 0.001
        return SimpleNamespace(
            last_hidden_state=hidden, pooler_output=hidden[:, :6].sum(dim=1)
        )


class UNet(torch.nn.Module):
    config = SimpleNamespace(
        in_channels=4, down_block_types=("DownBlock", "DownBlock", "DownBlock")
    )

    def forward(self, latents, timestep, encoder_hidden_states, **kwargs):
        sample = (
            latents * 0.5 + encoder_hidden_states[:, :6].sum() * 0.05 + timestep * 0.001
        )
        return SimpleNamespace(sample=sample)


class FluxTransformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=16, guidance_embeds=True)

    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        pooled_projections,
        encoder_hidden_states,
        img_ids,
        **kwargs,
    ):
        sample = (
            hidden_states * 0.5
            + pooled_projections.sum() * 0.05
            + encoder_hidden_states[:, :6].sum() * 0.02
            + timestep.view(-1, 1, 1) * 0.5
            + guidance.view(-1, 1, 1) * 0.1
            + img_ids[:, 1:2].unsqueeze(0) * 0.1
        )
        return (sample,)


class VAE(torch.nn.Module):
    config = SimpleNamespace(
        block_out_channels=(1, 1, 1, 1), scaling_factor=0.5, shift_factor=None
    )

    def encode(self, image):
        mean = torch.nn.functional.avg_pool2d(image[:, :1], 8).repeat(1, 4, 1, 1)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mean=mean, std=torch.full_like(mean, 0.1))
        )

    def decode(self, latents):
        sample = torch.nn.functional.interpolate(latents[:, :3] * 0.1, scale_factor=8)
        return SimpleNamespace(sample=sample)


PNDM_CONFIG = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
}
FLOW_MATCH_CONFIG = {
    "base_image_seq_len": 1,
    "base_shift": 0.5,
    "max_image_seq_len": 16,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def _write_runner_program(
    outdir: Path,
    name: str,
    export_spec,
    components: dict,
    model_info: DiffusionModelInfo,
    size: int,
    scheduler_config: dict,
) -> None:
    components = DiffusionComponents(components)
    lengths = model_info.max_sequence_lengths
    config = DiffusionConfig(
        model=ModelConfig(height=size, width=size, max_sequence_lengths=lengths),
        export=ExportConfig(),
    )
    pipeline = ResolvedDiffusionPipeline(
        export_spec, export_spec.resolve_model_info(components, model_info)
    )
    programs = export_programs(components, export_spec, config)
    write_pte(lower_programs(programs, pipeline, config), outdir / f"{name}.pte")

    snapshot = outdir / name
    for slot in export_spec.pipeline_behavior.tokenizer_slots:
        (snapshot / slot).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            Path(__file__).with_name("test_tokenizer.json"),
            snapshot / slot / "tokenizer.json",
        )
        (snapshot / slot / "tokenizer_config.json").write_text(
            json.dumps(
                {"tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "<pad>"}
            )
        )
    (snapshot / "scheduler").mkdir(exist_ok=True)
    (snapshot / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(scheduler_config)
    )


def write_runner_programs(outdir: Path) -> None:
    """Writes tiny_sd15.pte and tiny_flux.pte with snapshot directories."""
    _write_runner_program(
        outdir,
        "tiny_sd15",
        SD15PipelineExportSpec(),
        {"text_encoder": TextEncoder(), "denoiser": UNet(), "vae": VAE()},
        DiffusionModelInfo(
            scheduler_name="PNDMScheduler",
            scheduler_prediction_type="epsilon",
            max_sequence_lengths=(77,),
            zero_negative_prompt=False,
        ),
        16,
        PNDM_CONFIG,
    )
    _write_runner_program(
        outdir,
        "tiny_flux",
        FluxPipelineExportSpec(),
        {
            "text_encoder": TextEncoder(),
            "text_encoder_2": TextEncoder(),
            "denoiser": FluxTransformer(),
            "vae": VAE(),
        },
        DiffusionModelInfo(
            scheduler_name="FlowMatchEulerDiscreteScheduler",
            max_sequence_lengths=(77, 8),
            zero_negative_prompt=False,
        ),
        32,
        FLOW_MATCH_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_metadata_programs(args.outdir)
    write_runner_programs(args.outdir)


if __name__ == "__main__":
    main()
