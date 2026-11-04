# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Hugging Face Diffusers export."""

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
from executorch.export.huggingface.diffusers.config import (
    DiffusionConfig,
    ExportConfig,
    ModelConfig,
)
from executorch.export.huggingface.diffusers.export import (
    _resolve_max_sequence_lengths,
    _validate_text_encoder_lengths,
    build_diffusion_metadata,
    export_programs,
    lower_programs,
    main,
    write_pte,
)
from executorch.export.huggingface.diffusers.pipeline_discovery import (
    pipeline_spec_from_model_index,
    resolve_huggingface_pipeline,
)
from executorch.export.huggingface.diffusers.pipeline_specs import (
    DiffusionComponents,
    DiffusionModelInfo,
    Flux2KleinPipelineExportSpec,
    Flux2PipelineExportSpec,
    FluxPipelineExportSpec,
    ResolvedDiffusionPipeline,
    SD15PipelineExportSpec,
    SDXLPipelineExportSpec,
)
from executorch.export.huggingface.diffusers.wrappers import (
    _guidance_scale_embedding,
    BoundedFluxPosEmbed,
    Flux2TextEncoder,
    Flux2VAEDecoder,
    freeze_flux_pos_embed,
)

from executorch.extension.diffusers.pipeline_behaviors import SD15PipelineBehavior


class TextEncoder(torch.nn.Module):
    def __init__(self, width: int, pooled_width: int = 0) -> None:
        super().__init__()
        self.width = width
        self.pooled_width = pooled_width

    def forward(self, input_ids, output_hidden_states=False):
        shape = (*input_ids.shape, self.width)
        hidden = torch.ones(shape)
        output = SimpleNamespace(hidden_states=(torch.zeros(shape), hidden))
        output.last_hidden_state = hidden
        output.pooler_output = torch.ones(
            input_ids.shape[0], self.pooled_width or self.width
        )
        if self.pooled_width:
            output.text_embeds = torch.ones(input_ids.shape[0], self.pooled_width)
        return output


class UNet(torch.nn.Module):
    config = SimpleNamespace(
        in_channels=4,
        down_block_types=("DownBlock", "DownBlock", "DownBlock"),
    )

    def forward(self, latents, timestep, **kwargs):
        return SimpleNamespace(sample=latents)


class VAE(torch.nn.Module):
    config = SimpleNamespace(
        block_out_channels=(1, 1, 1, 1), scaling_factor=0.5, shift_factor=None
    )

    def encode(self, image):
        distribution = SimpleNamespace(
            mean=image[:, :1].repeat(1, 4, 1, 1),
            std=torch.ones(
                image.shape[0], 4, image.shape[2], image.shape[3], dtype=image.dtype
            ),
        )
        return SimpleNamespace(latent_dist=distribution)

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :3])


class FluxTransformer(torch.nn.Module):
    def __init__(self, guidance_embeds: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=16, guidance_embeds=guidance_embeds)

    def forward(self, hidden_states, **kwargs):
        image_ids = kwargs["img_ids"][:, :1].unsqueeze(0)
        return (hidden_states + image_ids * 0,)


def config(
    height: int,
    width: int,
    max_sequence_lengths: tuple[int, ...],
    *,
    text_encoder_dtype: torch.dtype = torch.float32,
    denoiser_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float32,
    dynamic_shapes: bool = False,
    min_size: int = 256,
    max_size: int = 1024,
) -> DiffusionConfig:
    return DiffusionConfig(
        model=ModelConfig(
            height=height,
            width=width,
            max_sequence_lengths=max_sequence_lengths,
            text_encoder_dtype=text_encoder_dtype,
            denoiser_dtype=denoiser_dtype,
            vae_dtype=vae_dtype,
        ),
        export=ExportConfig(
            dynamic_shapes=dynamic_shapes,
            min_size=min_size,
            max_size=max_size,
        ),
    )


def resolved_pipeline(
    export_spec=None,
    *,
    scheduler_name: str = "FakeScheduler",
    max_sequence_lengths: tuple[int, ...] = (77, 77),
    zero_negative_prompt: bool = True,
) -> ResolvedDiffusionPipeline:
    return ResolvedDiffusionPipeline(
        export_spec=export_spec or SDXLPipelineExportSpec(),
        model_info=DiffusionModelInfo(
            scheduler_name=scheduler_name,
            max_sequence_lengths=max_sequence_lengths,
            zero_negative_prompt=zero_negative_prompt,
        ),
    )


class DiffusersExportTest(unittest.TestCase):
    def setUp(self):
        self.components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "text_encoder_2": TextEncoder(5, 7),
                "denoiser": UNet(),
                "vae": VAE(),
            }
        )

    def test_sequence_length_override_changes_only_supported_encoder(self):
        spec = SimpleNamespace(
            name="sd3",
            default_max_text_sequence_lengths=(77, 77, 256),
            max_sequence_length_override_index=2,
        )
        self.assertEqual(
            _resolve_max_sequence_lengths(spec, 128),
            (77, 77, 128),
        )

    def test_text_encoder_position_limit_is_validated(self):
        encoder = TextEncoder(3)
        encoder.config = SimpleNamespace(max_position_embeddings=64)
        components = DiffusionComponents({"text_encoder": encoder})
        with self.assertRaisesRegex(ValueError, "supports 64 positions"):
            _validate_text_encoder_lengths(components, SD15PipelineBehavior(), (77,))

    def test_guidance_embedding_computes_in_float32(self):
        embedding = _guidance_scale_embedding(
            torch.tensor([70.0], dtype=torch.float16), 256
        )
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(embedding).all()))

    def test_sdxl_component_examples_match_pipeline_contract(self):
        inputs = SDXLPipelineExportSpec().example_inputs(
            self.components, config(64, 80, (77, 77))
        )
        self.assertEqual(inputs["denoise"][0].shape, (1, 4, 8, 10))
        self.assertEqual(inputs["denoise"][2].shape, (1, 77, 8))
        self.assertEqual(inputs["denoise"][3].shape, (1, 7))
        torch.testing.assert_close(
            inputs["denoise"][4],
            torch.tensor([[64, 80, 0, 0, 64, 80]], dtype=torch.float32),
        )

    def test_vae_runs_in_float32_at_fp16_boundaries(self):
        export_config = config(
            64,
            80,
            (77, 77),
            text_encoder_dtype=torch.float16,
            denoiser_dtype=torch.float16,
            vae_dtype=torch.float32,
        )
        modules = SDXLPipelineExportSpec().wrap_components(
            self.components, export_config
        )
        mean, std = modules["encode"](torch.zeros(1, 3, 8, 8, dtype=torch.float16))
        self.assertEqual(mean.dtype, torch.float16)
        self.assertEqual(std.dtype, torch.float16)
        image = modules["decode"](torch.zeros(1, 4, 8, 8, dtype=torch.float16))
        self.assertEqual(image.dtype, torch.float32)

    def test_vae_applies_latent_mean_and_std(self):
        vae = self.components.require("vae")
        vae.config = SimpleNamespace(
            **vars(vae.config),
            latents_mean=(0.1, 0.1, 0.1, 0.1),
            latents_std=(0.5, 0.5, 0.5, 0.5),
        )
        modules = SDXLPipelineExportSpec().wrap_components(
            self.components, config(64, 80, (77, 77))
        )

        mean, std = modules["encode"](torch.full((1, 3, 8, 8), 0.7))
        image = modules["decode"](torch.full((1, 4, 8, 8), 0.3))

        torch.testing.assert_close(mean, torch.full_like(mean, 0.3))
        torch.testing.assert_close(std, torch.ones_like(std))
        torch.testing.assert_close(image, torch.full_like(image, 0.7))

    def test_warns_when_fp16_export_requires_vae_upcast(self):
        vae = self.components.require("vae")
        vae.config = SimpleNamespace(**vars(vae.config), force_upcast=True)
        with self.assertWarnsRegex(UserWarning, "force_upcast=True"):
            SDXLPipelineExportSpec().wrap_components(
                self.components, config(64, 80, (77, 77), vae_dtype=torch.float16)
            )

    def test_sdxl_passes_guidance_embedding_to_unet(self):
        class GuidedUNet(UNet):
            def forward(self, latents, timestep, **kwargs):
                self.timestep_cond = kwargs["timestep_cond"]
                return SimpleNamespace(sample=latents)

        denoiser = GuidedUNet()
        denoiser.config = SimpleNamespace(
            in_channels=4,
            time_cond_proj_dim=256,
            down_block_types=("DownBlock", "DownBlock", "DownBlock"),
        )
        components = DiffusionComponents(
            {**self.components.models, "denoiser": denoiser}
        )
        export_spec = SDXLPipelineExportSpec()
        inputs = export_spec.example_inputs(components, config(64, 80, (77, 77)))
        self.assertEqual(len(inputs["denoise"]), 6)
        export_spec.wrap_components(components, config(64, 80, (77, 77)))["denoise"](
            *inputs["denoise"]
        )
        torch.testing.assert_close(
            denoiser.timestep_cond,
            _guidance_scale_embedding(torch.tensor([5.0]), 256),
        )

    def test_sdxl_resolves_unet_guidance_capabilities(self):
        denoiser = UNet()
        denoiser.config = SimpleNamespace(
            in_channels=4,
            time_cond_proj_dim=256,
            down_block_types=("DownBlock", "DownBlock", "DownBlock"),
        )
        components = DiffusionComponents(
            {**self.components.models, "denoiser": denoiser}
        )
        model_info = SDXLPipelineExportSpec().resolve_model_info(
            components, resolved_pipeline().model_info
        )
        self.assertEqual(
            (
                model_info.supports_cfg,
                model_info.accepts_guidance_input,
                model_info.guidance_input_dim,
                model_info.dynamic_spatial_multiple,
            ),
            (False, True, 256, 32),
        )

    def test_sdxl_exports_dynamic_spatial_shapes(self):
        export_config = config(
            128,
            128,
            (77, 77),
            dynamic_shapes=True,
            min_size=64,
            max_size=256,
        )
        programs = export_programs(
            self.components, SDXLPipelineExportSpec(), export_config
        )
        latents = torch.zeros(1, 4, 24, 16)
        output = programs["denoise"].module()(
            latents,
            torch.zeros(1),
            torch.zeros(1, 77, 8),
            torch.zeros(1, 7),
            torch.tensor([[192, 128, 0, 0, 192, 128]], dtype=torch.float32),
        )
        self.assertEqual(output.shape, latents.shape)

    def test_sd15_exports_four_methods(self):
        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "denoiser": UNet(),
                "vae": VAE(),
            }
        )
        programs = export_programs(
            components, SD15PipelineExportSpec(), config(8, 8, (77,))
        )
        self.assertEqual(
            set(programs), {"text_encoder_0", "denoise", "decode", "encode"}
        )

    def test_portable_pte_runs_through_diffusion_runner(self):
        from executorch.extension.diffusers.runner import DiffusionRunner
        from executorch.runtime import Runtime, Verification

        class Tokenizer:
            model_max_length = 77

            def __call__(self, prompt, **kwargs):
                return SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))

        class Scheduler:
            init_noise_sigma = 1.0
            order = 1
            config = SimpleNamespace(prediction_type="epsilon")

            def set_timesteps(self, steps):
                self.timesteps = torch.arange(steps - 1, -1, -1)

            def scale_model_input(self, latents, timestep):
                return latents

            def step(self, prediction, timestep, latents, return_dict=False):
                return (latents - prediction,)

        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "denoiser": UNet(),
                "vae": VAE(),
            }
        )
        export_config = config(8, 8, (77,))
        pipeline = resolved_pipeline(
            SD15PipelineExportSpec(), max_sequence_lengths=(77,)
        )
        edge_program = lower_programs(
            export_programs(components, pipeline.export_spec, export_config),
            pipeline,
            export_config,
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "tiny_diffusion.pte"
            write_pte(edge_program, output)
            program = Runtime.get().load_program(
                str(output), verification=Verification.Minimal
            )
            image = DiffusionRunner(program, [Tokenizer()], Scheduler()).generate(
                "cat", steps=1, cfg_scale=1.0, seed=9
            )
            # Release the memory-mapped PTE before Windows removes the directory.
            del program

        self.assertEqual(image.shape, (1, 3, 1, 1))
        torch.testing.assert_close(image, torch.full_like(image, 0.5))

    def test_portable_flux_pte_runs_through_diffusion_runner(self):
        from executorch.extension.diffusers.runner import DiffusionRunner
        from executorch.runtime import Runtime, Verification

        class Tokenizer:
            def __init__(self, length):
                self.model_max_length = length

            def __call__(self, prompt, *, max_length, **kwargs):
                return SimpleNamespace(
                    input_ids=torch.zeros(1, max_length, dtype=torch.long)
                )

        class Scheduler:
            config = SimpleNamespace(
                prediction_type=None,
                base_image_seq_len=1,
                max_image_seq_len=4,
                base_shift=0.5,
                max_shift=1.0,
            )

            def set_timesteps(self, steps, *, sigmas, mu):
                self.timesteps = torch.arange(steps - 1, -1, -1)

            def step(self, prediction, timestep, latents, return_dict=False):
                return (latents - prediction,)

        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "text_encoder_2": TextEncoder(5),
                "denoiser": FluxTransformer(False),
                "vae": VAE(),
            }
        )
        export_config = config(16, 16, (77, 8))
        export_spec = FluxPipelineExportSpec()
        model_info = export_spec.resolve_model_info(
            components,
            DiffusionModelInfo(
                scheduler_name="FakeScheduler",
                max_sequence_lengths=(77, 8),
                zero_negative_prompt=False,
            ),
        )
        pipeline = ResolvedDiffusionPipeline(export_spec, model_info)
        edge_program = lower_programs(
            export_programs(components, export_spec, export_config),
            pipeline,
            export_config,
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "tiny_flux.pte"
            write_pte(edge_program, output)
            program = Runtime.get().load_program(
                str(output), verification=Verification.Minimal
            )
            image = DiffusionRunner(
                program, [Tokenizer(77), Tokenizer(8)], Scheduler()
            ).generate("cat", steps=1, seed=9)
            del program

        self.assertEqual(image.shape, (1, 3, 2, 2))
        torch.testing.assert_close(image, torch.full_like(image, 0.5))

    def test_dynamic_portable_flux_pte_runs_at_a_different_size(self):
        from executorch.extension.diffusers.runner import DiffusionRunner
        from executorch.runtime import Runtime, Verification

        class Tokenizer:
            def __init__(self, length):
                self.model_max_length = length

            def __call__(self, prompt, *, max_length, **kwargs):
                return SimpleNamespace(
                    input_ids=torch.zeros(1, max_length, dtype=torch.long)
                )

        class Scheduler:
            config = SimpleNamespace(
                prediction_type=None,
                base_image_seq_len=1,
                max_image_seq_len=64,
                base_shift=0.5,
                max_shift=1.0,
            )

            def set_timesteps(self, steps, *, sigmas, mu):
                self.timesteps = torch.arange(steps - 1, -1, -1)

            def step(self, prediction, timestep, latents, return_dict=False):
                return (latents - prediction,)

        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "text_encoder_2": TextEncoder(5),
                "denoiser": FluxTransformer(False),
                "vae": VAE(),
            }
        )
        export_config = config(
            64,
            64,
            (77, 8),
            dynamic_shapes=True,
            min_size=32,
            max_size=96,
        )
        export_spec = FluxPipelineExportSpec()
        model_info = export_spec.resolve_model_info(
            components,
            DiffusionModelInfo(
                scheduler_name="FakeScheduler",
                max_sequence_lengths=(77, 8),
                zero_negative_prompt=False,
            ),
        )
        pipeline = ResolvedDiffusionPipeline(export_spec, model_info)
        edge_program = lower_programs(
            export_programs(components, export_spec, export_config),
            pipeline,
            export_config,
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "dynamic_flux.pte"
            write_pte(edge_program, output)
            program = Runtime.get().load_program(
                str(output), verification=Verification.Minimal
            )
            image = DiffusionRunner(
                program, [Tokenizer(77), Tokenizer(8)], Scheduler()
            ).generate("cat", steps=1, height=32, width=48, seed=9)
            del program

        self.assertEqual(image.shape, (1, 3, 4, 6))
        torch.testing.assert_close(image, torch.full_like(image, 0.5))

    def test_sd15_exports_dynamic_spatial_shapes(self):
        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "denoiser": UNet(),
                "vae": VAE(),
            }
        )
        export_config = config(
            128,
            128,
            (77,),
            dynamic_shapes=True,
            min_size=64,
            max_size=256,
        )
        denoise = export_programs(components, SD15PipelineExportSpec(), export_config)[
            "denoise"
        ].module()
        latents = torch.zeros(1, 4, 24, 16)
        output = denoise(
            latents,
            torch.zeros(1),
            torch.zeros(1, 77, 3),
        )
        self.assertEqual(output.shape, latents.shape)

    def test_pipeline_specs_are_discovered(self):
        sd15 = pipeline_spec_from_model_index(
            {
                "_class_name": "StableDiffusionPipeline",
                "scheduler": ["diffusers", "DDIMScheduler"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        )
        sdxl = pipeline_spec_from_model_index(
            {
                "_class_name": "StableDiffusionXLPipeline",
                "scheduler": ["diffusers", "EulerDiscreteScheduler"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": [
                    "transformers",
                    "CLIPTextModelWithProjection",
                ],
            }
        )
        self.assertIsInstance(sd15, SD15PipelineExportSpec)
        self.assertIsInstance(sdxl, SDXLPipelineExportSpec)

        flux = pipeline_spec_from_model_index(
            {
                "_class_name": "FluxPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
        self.assertIsInstance(flux, FluxPipelineExportSpec)

    def test_flux_examples_use_packed_latents_and_position_ids(self):
        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "text_encoder_2": TextEncoder(5),
                "denoiser": FluxTransformer(True),
                "vae": VAE(),
            }
        )
        inputs = FluxPipelineExportSpec().example_inputs(
            components, config(64, 80, (77, 128))
        )["denoise"]
        self.assertEqual(inputs[0].shape, (1, 20, 16))
        self.assertEqual(inputs[2].shape, (1, 128, 5))
        self.assertEqual(inputs[4].shape, (128, 3))
        self.assertEqual(inputs[5].shape, (20, 3))
        self.assertEqual(len(inputs), 7)

    def test_flux_resolves_schnell_and_dev_defaults(self):
        def resolve(guidance_embeds):
            components = DiffusionComponents(
                {
                    "denoiser": FluxTransformer(guidance_embeds),
                    "vae": VAE(),
                }
            )
            return FluxPipelineExportSpec().resolve_model_info(
                components,
                DiffusionModelInfo(
                    scheduler_name="FlowMatchEulerDiscreteScheduler",
                    max_sequence_lengths=(77, 512),
                    zero_negative_prompt=False,
                ),
            )

        schnell, dev = resolve(False), resolve(True)
        self.assertEqual(
            (
                schnell.default_inference_steps,
                schnell.accepts_guidance_input,
                schnell.default_guidance_scale,
            ),
            (4, False, 0.0),
        )
        self.assertEqual(
            (
                dev.default_inference_steps,
                dev.accepts_guidance_input,
                dev.default_guidance_scale,
            ),
            (28, True, 3.5),
        )

    def test_flux_rejects_invalid_static_and_dynamic_sizes(self):
        components = DiffusionComponents(
            {
                "denoiser": FluxTransformer(False),
                "vae": VAE(),
            }
        )
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            FluxPipelineExportSpec().wrap_components(
                components, config(64, 72, (77, 512))
            )
        with self.assertRaisesRegex(ValueError, "dynamic size bounds"):
            FluxPipelineExportSpec().wrap_components(
                components,
                config(
                    64,
                    64,
                    (77, 512),
                    dynamic_shapes=True,
                    min_size=48,
                    max_size=100,
                ),
            )

    def test_flux_exports_dynamic_image_tokens_and_decode_shape(self):
        components = DiffusionComponents(
            {
                "text_encoder": TextEncoder(3),
                "text_encoder_2": TextEncoder(5),
                "denoiser": FluxTransformer(False),
                "vae": VAE(),
            }
        )
        export_config = config(
            64,
            64,
            (77, 8),
            dynamic_shapes=True,
            min_size=32,
            max_size=96,
        )
        programs = export_programs(components, FluxPipelineExportSpec(), export_config)
        denoise = programs["denoise"].module()
        output = denoise(
            torch.zeros(1, 15, 16),
            torch.zeros(1),
            torch.zeros(1, 8, 5),
            torch.zeros(1, 3),
            torch.zeros(8, 3),
            torch.zeros(15, 3),
        )
        self.assertEqual(output.shape, (1, 15, 16))
        decoded = programs["decode"].module()(torch.zeros(1, 4, 6, 10))
        self.assertEqual(decoded.shape, (1, 3, 6, 10))

    def test_flux_position_embeddings_are_frozen_without_losing_precision(self):
        class PositionEmbedding(torch.nn.Module):
            axes_dim = (2, 2, 2)

            def forward(self, ids):
                frequencies = torch.arange(2, dtype=torch.float64)
                angles = torch.cat(
                    [
                        ids[:, axis : axis + 1].double() * frequencies
                        for axis in range(3)
                    ],
                    dim=-1,
                )
                return angles.cos().float(), angles.sin().float()

        model = torch.nn.Module()
        model.pos_embed = PositionEmbedding()
        ids = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 3.0]])
        expected = model.pos_embed(ids)

        freeze_flux_pos_embed(model, max_positions=4)

        self.assertIsInstance(model.pos_embed, BoundedFluxPosEmbed)
        actual = model.pos_embed(ids)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertEqual(actual[0].dtype, torch.float32)

    def test_flux_rope_tables_cover_static_and_dynamic_spatial_bounds(self):
        class PositionEmbedding(torch.nn.Module):
            axes_dim = (2, 2, 2)

            def forward(self, ids):
                values = ids.repeat_interleave(2, dim=-1).float()
                return values, -values

        for dynamic, bound in ((False, 5), (True, 6)):
            with self.subTest(dynamic=dynamic):
                transformer = FluxTransformer(False)
                transformer.pos_embed = PositionEmbedding()
                components = DiffusionComponents(
                    {
                        "text_encoder": TextEncoder(3),
                        "text_encoder_2": TextEncoder(5),
                        "denoiser": transformer,
                        "vae": VAE(),
                    }
                )
                FluxPipelineExportSpec().wrap_components(
                    components,
                    config(
                        64,
                        80,
                        (77, 128),
                        dynamic_shapes=dynamic,
                        min_size=32,
                        max_size=96,
                    ),
                )
                self.assertIsInstance(transformer.pos_embed, BoundedFluxPosEmbed)
                self.assertEqual(transformer.pos_embed.cos_2.shape, (bound, 2))
                ids = torch.tensor([[0.0, 3.0, bound - 1]])
                actual = transformer.pos_embed(ids)
                expected = PositionEmbedding()(ids)
                torch.testing.assert_close(actual, expected)

    def test_bounded_flux_position_embeddings_gather_dynamic_ids(self):
        class PositionEmbedding(torch.nn.Module):
            axes_dim = (2, 2, 2)

            def forward(self, ids):
                parts = [ids[:, axis : axis + 1].repeat(1, 2) for axis in range(3)]
                values = torch.cat(parts, dim=-1).float()
                return values, -values

        embedding = BoundedFluxPosEmbed(PositionEmbedding(), 8)
        ids = torch.tensor([[0.0, 2.0, 3.0], [0.0, 4.0, 1.0]])
        cos, sin = embedding(ids)
        expected = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 4.0, 4.0, 1.0, 1.0]]
        )
        torch.testing.assert_close(cos, expected)
        torch.testing.assert_close(sin, -expected)

    def test_flux_rope_tables_preserve_coordinates_above_bfloat16_precision(self):
        class PositionEmbedding(torch.nn.Module):
            axes_dim = (2, 2, 2)

            def forward(self, ids):
                values = ids.repeat_interleave(2, dim=-1).float()
                return values, -values

        embedding = BoundedFluxPosEmbed(PositionEmbedding(), 260)
        ids = torch.tensor([[0, 257, 259]])
        torch.testing.assert_close(embedding(ids), PositionEmbedding()(ids))

    def test_flux_rope_tables_expand_when_reusing_components(self):
        class PositionEmbedding(torch.nn.Module):
            axes_dim = (2, 2, 2)

            def forward(self, ids):
                values = ids.repeat_interleave(2, dim=-1).float()
                return values, -values

        model = torch.nn.Module()
        model.pos_embed = PositionEmbedding()
        freeze_flux_pos_embed(model, max_positions=2)
        freeze_flux_pos_embed(model, max_positions=4)
        ids = torch.tensor([[0, 1, 3]])
        torch.testing.assert_close(model.pos_embed(ids), PositionEmbedding()(ids))

    def test_model_info_is_discovered_from_hugging_face(self):
        pipeline_config = {
            "_class_name": "StableDiffusionXLPipeline",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "CLIPTokenizer"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": [
                "transformers",
                "CLIPTextModelWithProjection",
            ],
            "force_zeros_for_empty_prompt": False,
        }
        tokenizer = mock.Mock()
        diffusers = SimpleNamespace(
            DiffusionPipeline=SimpleNamespace(
                load_config=mock.Mock(return_value=pipeline_config)
            ),
            EulerDiscreteScheduler=SimpleNamespace(
                load_config=mock.Mock(return_value={"prediction_type": "epsilon"})
            ),
        )
        transformers = SimpleNamespace(AutoTokenizer=tokenizer)
        with mock.patch.dict(
            "sys.modules", {"diffusers": diffusers, "transformers": transformers}
        ):
            pipeline = resolve_huggingface_pipeline("model")
        self.assertEqual(
            pipeline.behavior.tokenizer_slots, ("tokenizer", "tokenizer_2")
        )
        self.assertEqual(
            pipeline.behavior.text_encoder_slots,
            ("text_encoder", "text_encoder_2"),
        )
        self.assertEqual(pipeline.model_info.scheduler_name, "EulerDiscreteScheduler")
        self.assertEqual(pipeline.model_info.scheduler_prediction_type, "epsilon")
        self.assertEqual(pipeline.model_info.max_sequence_lengths, (77, 77))
        self.assertEqual(pipeline.export_spec.default_size, (1024, 1024))
        self.assertFalse(pipeline.model_info.zero_negative_prompt)
        tokenizer.from_pretrained.assert_not_called()

    def test_unknown_pipeline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported diffusion pipeline"):
            pipeline_spec_from_model_index(
                {"_class_name": "UnknownPipeline", "scheduler": ["x", "y"]}
            )

    def test_missing_pipeline_components_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires tokenizer components"):
            pipeline_spec_from_model_index(
                {
                    "_class_name": "StableDiffusionPipeline",
                    "scheduler": ["diffusers", "PNDMScheduler"],
                    "text_encoder": ["transformers", "CLIPTextModel"],
                }
            )

    def test_write_pte_normalizes_output_and_propagates_writes(self):
        edge_program = mock.Mock()
        program = edge_program.to_executorch.return_value
        with TemporaryDirectory() as directory:
            output = Path(directory) / "models" / "diffusion"
            write_pte(edge_program, output, backend_config="config")
            self.assertTrue(output.with_suffix(".pte").is_file())
        edge_program.to_executorch.assert_called_once_with(config="config")
        program.write_to_file.assert_called_once()
        program.write_tensor_data_to_file.assert_called_once_with(
            outdir=str(output.parent)
        )

    def test_write_pte_propagates_serialization_failure(self):
        edge_program = mock.Mock()
        edge_program.to_executorch.return_value.write_to_file.side_effect = OSError(
            "write failed"
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, "write failed"):
                write_pte(edge_program, Path(directory) / "diffusion")

    @mock.patch("executorch.export.huggingface.diffusers.backends.export_to_mlx")
    @mock.patch(
        "executorch.export.huggingface.diffusers.export.resolve_huggingface_pipeline"
    )
    def test_cli_dispatches_to_mlx(self, resolve_pipeline, export_to_mlx):
        resolve_pipeline.return_value = resolved_pipeline(
            SD15PipelineExportSpec(), max_sequence_lengths=(77,)
        )
        main(
            [
                "model",
                "model.pte",
                "--backend",
                "mlx",
                "--height",
                "512",
                "--width",
                "768",
                "--denoiser-dtype",
                "float16",
            ]
        )
        export_config = export_to_mlx.call_args.kwargs["config"]
        self.assertEqual(
            (
                export_config.model.height,
                export_config.model.width,
                export_config.model.max_sequence_lengths,
                export_config.model.denoiser_dtype,
            ),
            (512, 768, (77,), torch.float16),
        )

    @mock.patch("executorch.export.huggingface.diffusers.backends.export_to_mlx")
    @mock.patch(
        "executorch.export.huggingface.diffusers.export.resolve_huggingface_pipeline"
    )
    def test_cli_uses_pipeline_defaults(self, resolve_pipeline, export_to_mlx):
        resolve_pipeline.return_value = resolved_pipeline(
            SD15PipelineExportSpec(), max_sequence_lengths=(77,)
        )
        main(["model", "model.pte", "--backend", "mlx"])
        export_config = export_to_mlx.call_args.kwargs["config"]
        self.assertEqual(
            (
                export_config.model.height,
                export_config.model.width,
                export_config.model.max_sequence_lengths,
            ),
            (512, 512, (77,)),
        )


class LanguageModel(torch.nn.Module):
    config = SimpleNamespace(max_position_embeddings=1024)

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)

    @property
    def dtype(self):
        return self.embedding.weight.dtype

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return SimpleNamespace(
            hidden_states=tuple(hidden + index * 0.01 for index in range(31))
        )


class Tokenizer:
    model_max_length = 1024

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize", False):
            prompt = messages[0][-1]["content"][0]["text"]
            return self(prompt, max_length=kwargs["max_length"])
        return messages[0]["content"]

    def __call__(self, prompt, *, max_length, **kwargs):
        ids = torch.full((1, max_length), 1 if prompt else 0, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[:, -2:] = 0
        return {"input_ids": ids, "attention_mask": mask}


class Flux2ExportTest(unittest.TestCase):
    def _components(self, guidance_embeds=False):
        try:
            from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
        except ImportError:
            self.skipTest("requires Diffusers FLUX.2 support")
        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            block_out_channels=(4, 4, 4, 4),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            mid_block_add_attention=False,
        ).eval()
        vae.bn.running_mean.copy_(torch.arange(16) * 0.02)
        vae.bn.running_var.copy_(torch.linspace(0.3, 1.6, 16))
        transformer = Flux2Transformer2DModel(
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=24,
            timestep_guidance_channels=8,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=guidance_embeds,
        ).eval()
        return DiffusionComponents(
            {
                "text_encoder": LanguageModel().eval(),
                "denoiser": transformer,
                "vae": vae,
            }
        )

    def _config(self, *, dynamic=False):
        return DiffusionConfig(
            ModelConfig(height=64, width=64, max_sequence_lengths=(8,)),
            ExportConfig(dynamic_shapes=dynamic, min_size=32, max_size=96),
        )

    def _model_info(self, *, is_distilled=False):
        return DiffusionModelInfo(
            scheduler_name="FlowMatchEulerDiscreteScheduler",
            max_sequence_lengths=(8,),
            zero_negative_prompt=False,
            is_distilled=is_distilled,
        )

    def test_pipeline_discovery_and_sequence_length_override(self):
        for name, spec_type in (
            ("Flux2Pipeline", Flux2PipelineExportSpec),
            ("Flux2KleinPipeline", Flux2KleinPipelineExportSpec),
        ):
            with self.subTest(pipeline=name):
                spec = pipeline_spec_from_model_index(
                    {
                        "_class_name": name,
                        "tokenizer": ["transformers", "AutoProcessor"],
                        "text_encoder": ["transformers", "AutoModel"],
                    }
                )
                self.assertIsInstance(spec, spec_type)
                self.assertEqual(_resolve_max_sequence_lengths(spec, 128), (128,))

    def test_distilled_flag_is_discovered_from_pipeline_config(self):
        pipeline_config = {
            "_class_name": "Flux2KleinPipeline",
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "is_distilled": True,
        }
        diffusers = SimpleNamespace(
            DiffusionPipeline=SimpleNamespace(
                load_config=mock.Mock(return_value=pipeline_config)
            ),
            FlowMatchEulerDiscreteScheduler=SimpleNamespace(
                load_config=mock.Mock(return_value={})
            ),
        )
        with mock.patch.dict("sys.modules", {"diffusers": diffusers}):
            pipeline = resolve_huggingface_pipeline("klein")
        self.assertTrue(pipeline.model_info.is_distilled)
        self.assertIsInstance(pipeline.export_spec, Flux2KleinPipelineExportSpec)

    def test_klein_base_and_distilled_defaults(self):
        components = self._components()
        spec = Flux2KleinPipelineExportSpec()
        for is_distilled in (False, True):
            with self.subTest(is_distilled=is_distilled):
                info = spec.resolve_model_info(
                    components, self._model_info(is_distilled=is_distilled)
                )
                metadata = build_diffusion_metadata(
                    spec.pipeline_behavior, info, self._config()
                )
                self.assertEqual(
                    metadata.default_inference_steps, 4 if is_distilled else 50
                )
                self.assertEqual(
                    metadata.default_cfg_scale, 1.0 if is_distilled else 4.0
                )
                self.assertEqual(metadata.supports_cfg, not is_distilled)
                self.assertEqual(metadata.supports_negative_prompt, not is_distilled)
                self.assertFalse(metadata.accepts_guidance_input)

    def test_rejects_incompatible_vae_channels_and_spatial_bounds(self):
        components = self._components()
        spec = Flux2KleinPipelineExportSpec()
        export_config = self._config()
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            spec.example_inputs(
                components,
                replace(export_config, model=replace(export_config.model, width=72)),
            )
        with self.assertRaisesRegex(ValueError, "dynamic size bounds"):
            spec.dynamic_shapes(
                components,
                replace(
                    export_config,
                    export=ExportConfig(dynamic_shapes=True, min_size=32, max_size=100),
                ),
            )
        components.require("vae").register_to_config(latent_channels=8)
        with self.assertRaisesRegex(ValueError, "latent channels must match"):
            spec.example_inputs(components, export_config)

    def test_qwen_hidden_state_export_matches_reference(self):
        try:
            from transformers import Qwen3Config, Qwen3Model
        except ImportError:
            self.skipTest("requires Transformers Qwen3 support")
        model = Qwen3Model(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        ).eval()
        ids = torch.arange(8).reshape(1, 8)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        with torch.no_grad():
            reference = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
            expected = torch.cat(reference.hidden_states, dim=-1)
            wrapper = Flux2TextEncoder(model, (0, 1, 2))
            exported = torch.export.export(wrapper, (ids, mask))
            torch.testing.assert_close(exported.module()(ids, mask), expected)

    def test_vae_decoder_undoes_reference_batch_normalization_and_patching(self):
        components = self._components()
        from diffusers import Flux2Pipeline

        vae = components.require("vae")
        latents = torch.randn(1, 16, 2, 3)
        with torch.no_grad():
            normalized = latents * (
                vae.bn.running_var + vae.config.batch_norm_eps
            ).sqrt().reshape(1, -1, 1, 1)
            normalized = normalized + vae.bn.running_mean.reshape(1, -1, 1, 1)
            unpacked = Flux2Pipeline._unpatchify_latents(normalized)
            expected = (vae.decode(unpacked).sample / 2 + 0.5).clamp(0, 1)
            wrapper = Flux2VAEDecoder(vae, torch.float32)
            exported = torch.export.export(wrapper, (latents,))
            torch.testing.assert_close(exported.module()(latents), expected)

    def test_real_transformer_export_matches_reference_at_dynamic_resolution(self):
        for guidance in (False, True):
            with self.subTest(guidance=guidance):
                components = self._components(guidance)
                reference = copy.deepcopy(components.require("denoiser"))
                spec = (
                    Flux2PipelineExportSpec()
                    if guidance
                    else Flux2KleinPipelineExportSpec()
                )
                programs = export_programs(components, spec, self._config(dynamic=True))
                prompt = torch.randn(1, 8, 24)
                inputs = spec.pipeline_behavior.prepare_denoiser_inputs(
                    torch.randn(1, 6, 16),
                    torch.tensor([500.0]),
                    (prompt,),
                    height=32,
                    width=48,
                    vae_scale_factor=8,
                    dtype=torch.float32,
                    guidance_scale=4.0 if guidance else None,
                )
                with torch.no_grad():
                    expected = reference(
                        hidden_states=inputs[0],
                        timestep=inputs[1],
                        encoder_hidden_states=inputs[2],
                        txt_ids=inputs[3],
                        img_ids=inputs[4],
                        guidance=inputs[5] if guidance else None,
                        return_dict=False,
                    )[0]
                    torch.testing.assert_close(
                        programs["denoise"].module()(*inputs), expected
                    )
                    decoded = programs["decode"].module()(torch.randn(1, 16, 2, 3))
                    self.assertEqual(decoded.shape, (1, 3, 32, 48))

    def test_portable_klein_pte_matches_diffusers_at_dynamic_resolution(self):
        from executorch.extension.diffusers.runner import DiffusionRunner
        from executorch.runtime import Runtime, Verification

        for is_distilled in (False, True):
            with self.subTest(is_distilled=is_distilled):
                components = self._components()
                from diffusers import (
                    FlowMatchEulerDiscreteScheduler,
                    Flux2KleinPipeline,
                )

                reference = Flux2KleinPipeline(
                    scheduler=FlowMatchEulerDiscreteScheduler(
                        use_dynamic_shifting=True
                    ),
                    vae=copy.deepcopy(components.require("vae")),
                    transformer=copy.deepcopy(components.require("denoiser")),
                    text_encoder=copy.deepcopy(components.require("text_encoder")),
                    tokenizer=Tokenizer(),
                    is_distilled=is_distilled,
                )
                reference.set_progress_bar_config(disable=True)
                spec = Flux2KleinPipelineExportSpec()
                info = spec.resolve_model_info(
                    components, self._model_info(is_distilled=is_distilled)
                )
                export_config = self._config(dynamic=True)
                pipeline = ResolvedDiffusionPipeline(spec, info)
                edge = lower_programs(
                    export_programs(components, spec, export_config),
                    pipeline,
                    export_config,
                )
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "klein.pte"
                    write_pte(edge, path)
                    program = Runtime.get().load_program(
                        str(path), verification=Verification.Minimal
                    )
                    runner = DiffusionRunner(
                        program,
                        [Tokenizer()],
                        FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
                    )
                    actual = runner.generate(
                        "cat", steps=2, seed=9, height=32, width=48
                    )
                    with torch.no_grad():
                        expected = reference(
                            prompt="cat",
                            num_inference_steps=2,
                            guidance_scale=1.0 if is_distilled else 4.0,
                            generator=torch.Generator().manual_seed(9),
                            height=32,
                            width=48,
                            max_sequence_length=8,
                            output_type="pt",
                        ).images
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                    del runner, program


if __name__ == "__main__":
    unittest.main()
