# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the experimental Diffusers runtime extension."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from executorch.extension.diffusers.config import (
    DiffusionConfig,
    ExportConfig,
    ModelConfig,
)
from executorch.extension.diffusers.metadata import (
    build_diffusion_metadata,
    DiffusionMetadata,
    METADATA_METHODS,
)
from executorch.extension.diffusers.pipeline_specs import (
    DiffusionComponents,
    DiffusionModelInfo,
    ResolvedDiffusionPipeline,
    SD15PipelineSpec,
    SDXLPipelineSpec,
)
from executorch.extension.diffusers.runner import (
    _cast_floating_tensors,
    DiffusionRunner,
)
from executorch.extension.diffusers.wrappers import _guidance_scale_embedding


class _TextEncoder(torch.nn.Module):
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


class _UNet(torch.nn.Module):
    config = SimpleNamespace(
        in_channels=4,
        down_block_types=("DownBlock", "DownBlock", "DownBlock"),
    )

    def forward(self, latents, timestep, **kwargs):
        return SimpleNamespace(sample=latents)


class _VAE(torch.nn.Module):
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


def _config(
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


def _resolved_pipeline(
    spec=None,
    *,
    scheduler_name: str = "FakeScheduler",
    max_sequence_lengths: tuple[int, ...] = (77, 77),
    zero_negative_prompt: bool = True,
) -> ResolvedDiffusionPipeline:
    return ResolvedDiffusionPipeline(
        spec=spec or SDXLPipelineSpec(),
        model_info=DiffusionModelInfo(
            scheduler_name=scheduler_name,
            max_sequence_lengths=max_sequence_lengths,
            zero_negative_prompt=zero_negative_prompt,
        ),
    )


class ConfigValidationTest(unittest.TestCase):
    def test_model_dimensions_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "height and width"):
            ModelConfig(height=0, width=512, max_sequence_lengths=(77,))

    def test_max_sequence_lengths_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_sequence_lengths"):
            ModelConfig(
                height=512,
                width=512,
                max_sequence_lengths=(77, 0),
            )

    def test_component_dtypes_must_be_supported(self):
        with self.assertRaisesRegex(ValueError, "vae_dtype"):
            ModelConfig(
                height=512,
                width=512,
                max_sequence_lengths=(77,),
                vae_dtype=torch.float64,
            )

    def test_dynamic_example_size_must_be_within_bounds(self):
        with self.assertRaisesRegex(ValueError, "dynamic-shape bounds"):
            DiffusionConfig(
                model=ModelConfig(
                    height=128,
                    width=512,
                    max_sequence_lengths=(77,),
                ),
                export=ExportConfig(
                    dynamic_shapes=True,
                    min_size=256,
                    max_size=1024,
                ),
            )

    def test_dynamic_bounds_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "positive, ordered integers"):
            ExportConfig(min_size=1024, max_size=256)

    def test_dynamic_bounds_must_define_a_range(self):
        with self.assertRaisesRegex(ValueError, "less than max_size"):
            DiffusionConfig(
                model=ModelConfig(
                    height=512,
                    width=512,
                    max_sequence_lengths=(77,),
                ),
                export=ExportConfig(
                    dynamic_shapes=True,
                    min_size=512,
                    max_size=512,
                ),
            )


class GuidanceTest(unittest.TestCase):
    def test_guidance_embedding_computes_in_float32(self):
        embedding = _guidance_scale_embedding(
            torch.tensor([70.0], dtype=torch.float16), 256
        )
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(embedding).all()))

    def test_classifier_free_guidance_combines_two_predictions(self):
        prediction = SD15PipelineSpec().combine_predictions(
            (torch.tensor([2.0]), torch.tensor([5.0])),
            cfg_scale=3.0,
        )
        # unconditional + scale * (conditional - unconditional) = 2 + 3 * 3
        torch.testing.assert_close(prediction, torch.tensor([11.0]))

    def test_model_guidance_scale_is_a_denoiser_input(self):
        inputs = SD15PipelineSpec().prepare_denoiser_inputs(
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1),
            (torch.zeros(1, 77, 3),),
            height=64,
            width=64,
            dtype=torch.float16,
            guidance_scale=3.0,
        )
        torch.testing.assert_close(
            inputs[-1],
            torch.tensor([3.0], dtype=torch.float32),
        )

    def test_model_guidance_scale_preserves_float32_precision(self):
        inputs = SDXLPipelineSpec().prepare_denoiser_inputs(
            torch.zeros(1, 4, 8, 8, dtype=torch.float16),
            torch.zeros(1),
            (torch.zeros(1, 77, 3), torch.zeros(1, 3)),
            height=64,
            width=64,
            dtype=torch.float16,
            guidance_scale=7.3,
        )
        torch.testing.assert_close(inputs[-1], torch.tensor([7.3]))


class PipelineSpecTest(unittest.TestCase):
    def setUp(self):
        self.components = DiffusionComponents(
            {
                "text_encoder": _TextEncoder(3),
                "text_encoder_2": _TextEncoder(5, 7),
                "denoiser": _UNet(),
                "vae": _VAE(),
            }
        )
        self.config = _config(
            height=64,
            width=80,
            max_sequence_lengths=(77, 77),
        )

    def test_conditioning_and_examples(self):
        pipeline_spec = SDXLPipelineSpec()
        inputs = pipeline_spec.example_inputs(self.components, self.config)
        # The VAE downsamples 64x80 by 8, and the fake UNet has 4 input channels.
        self.assertEqual(inputs["denoise"][0].shape, (1, 4, 8, 10))
        # The fake encoders have widths 3 and 5; encoder 2 pools to width 7.
        self.assertEqual(inputs["denoise"][2].shape, (1, 77, 8))
        self.assertEqual(inputs["denoise"][3].shape, (1, 7))
        # SDXL time IDs contain original size, crop coordinates, and target size.
        torch.testing.assert_close(
            inputs["denoise"][4],
            torch.tensor([[64, 80, 0, 0, 64, 80]], dtype=torch.float32),
        )

    def test_metadata_describes_pipeline_contract(self):
        pipeline = _resolved_pipeline()
        metadata = build_diffusion_metadata(
            pipeline.spec, pipeline.model_info, self.config
        )
        methods = metadata.to_constant_methods()
        self.assertTrue(
            {
                METADATA_METHODS["pipeline"],
                METADATA_METHODS["max_sequence_lengths"],
                METADATA_METHODS["vae_scale_factor"],
            }.issubset(methods)
        )
        torch.testing.assert_close(
            methods[METADATA_METHODS["max_sequence_lengths"]],
            torch.tensor([77, 77], dtype=torch.int32),
        )
        values = {
            key: methods[method_name] for key, method_name in METADATA_METHODS.items()
        }
        self.assertEqual(
            DiffusionMetadata.from_constant_method_values(values), metadata
        )

    def test_metadata_supports_arbitrary_encoder_count(self):
        pipeline = _resolved_pipeline()
        metadata = build_diffusion_metadata(
            pipeline.spec, pipeline.model_info, self.config
        )
        metadata = replace(metadata, max_sequence_lengths=(77, 77, 256))
        methods = metadata.to_constant_methods()
        values = {
            key: methods[method_name] for key, method_name in METADATA_METHODS.items()
        }
        self.assertEqual(
            DiffusionMetadata.from_constant_method_values(values).max_sequence_lengths,
            (77, 77, 256),
        )

    def test_metadata_separates_cfg_and_model_guidance(self):
        pipeline = _resolved_pipeline()
        model_info = replace(
            pipeline.model_info,
            supports_cfg=False,
            accepts_guidance_input=True,
            guidance_input_dim=256,
        )
        metadata = build_diffusion_metadata(pipeline.spec, model_info, self.config)
        self.assertEqual(
            (
                metadata.default_cfg_scale,
                metadata.default_guidance_scale,
                metadata.guidance_input_dim,
            ),
            (1.0, 5.0, 256),
        )

    def test_vae_runs_in_float32_at_fp16_boundaries(self):
        config = _config(
            64,
            80,
            (77, 77),
            text_encoder_dtype=torch.float16,
            denoiser_dtype=torch.float16,
            vae_dtype=torch.float32,
        )
        modules = SDXLPipelineSpec().wrap_components(self.components, config)
        mean, std = modules["encode"](torch.zeros(1, 3, 8, 8, dtype=torch.float16))
        self.assertEqual(mean.dtype, torch.float16)
        self.assertEqual(std.dtype, torch.float16)
        image = modules["decode"](torch.zeros(1, 4, 8, 8, dtype=torch.float16))
        self.assertEqual(image.dtype, torch.float32)

    def test_warns_when_fp16_export_requires_vae_upcast(self):
        vae = self.components.require("vae")
        vae.config = SimpleNamespace(**vars(vae.config), force_upcast=True)
        config = _config(
            64,
            80,
            (77, 77),
            vae_dtype=torch.float16,
        )
        with self.assertWarnsRegex(UserWarning, "force_upcast=True"):
            SDXLPipelineSpec().wrap_components(self.components, config)

    def test_sdxl_passes_guidance_embedding_to_unet(self):
        class GuidedUNet(_UNet):
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
        inputs = SDXLPipelineSpec().example_inputs(components, self.config)
        self.assertEqual(len(inputs["denoise"]), 6)
        torch.testing.assert_close(inputs["denoise"][-1], torch.tensor([5.0]))
        SDXLPipelineSpec().wrap_components(components, self.config)["denoise"](
            *inputs["denoise"]
        )
        torch.testing.assert_close(
            denoiser.timestep_cond,
            _guidance_scale_embedding(torch.tensor([5.0]), 256),
        )

    def test_sdxl_resolves_unet_guidance_capabilities(self):
        denoiser = _UNet()
        denoiser.config = SimpleNamespace(
            in_channels=4,
            time_cond_proj_dim=256,
            down_block_types=("DownBlock", "DownBlock", "DownBlock"),
        )
        components = DiffusionComponents(
            {**self.components.models, "denoiser": denoiser}
        )
        model_info = SDXLPipelineSpec().resolve_model_info(
            components, _resolved_pipeline().model_info
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

    def test_model_input_scaling_is_optional(self):
        latents = torch.ones(1, 4, 8, 8)
        scheduler = SimpleNamespace()
        self.assertIs(
            SD15PipelineSpec().prepare_model_input(
                scheduler, latents, torch.tensor(1.0)
            ),
            latents,
        )

    def test_sd15_has_one_encoder_and_no_pooled_conditioning(self):
        components = DiffusionComponents(
            {
                "text_encoder": _TextEncoder(3),
                "denoiser": _UNet(),
                "vae": _VAE(),
            }
        )
        config = _config(
            64,
            64,
            (77,),
            denoiser_dtype=torch.float16,
        )
        pipeline_spec = SD15PipelineSpec()
        modules = pipeline_spec.wrap_components(components, config)
        inputs = pipeline_spec.example_inputs(components, config)
        self.assertNotIn("text_encoder_1", modules)
        self.assertEqual(len(inputs["denoise"]), 3)
        model_info = _resolved_pipeline(
            pipeline_spec, max_sequence_lengths=(77,)
        ).model_info
        self.assertEqual(
            build_diffusion_metadata(
                pipeline_spec, model_info, config
            ).max_sequence_lengths,
            (77,),
        )


class _Method:
    def __init__(self, function):
        self.function = function

    def execute(self, inputs):
        result = self.function(*inputs)
        return result if isinstance(result, (tuple, list)) else (result,)


class _TensorMeta:
    def __init__(self, shape=(1, 4, 2, 2)):
        self.shape = shape

    def sizes(self):
        return self.shape


class _MethodMeta:
    def __init__(self, shape):
        self.shape = shape

    def input_tensor_meta(self, index):
        return _TensorMeta(self.shape)


class _Program:
    def __init__(self, pipeline="sdxl", dynamic_shapes=False):
        max_sequence_lengths = (77, 77) if pipeline == "sdxl" else (77,)
        metadata = DiffusionMetadata(
            pipeline=pipeline,
            scheduler="FakeScheduler",
            scheduler_prediction_type="epsilon",
            supports_cfg=True,
            accepts_guidance_input=False,
            supports_negative_prompt=True,
            default_cfg_scale=7.5,
            default_guidance_scale=None,
            guidance_input_dim=0,
            default_inference_steps=50,
            text_encoder_dtype="float32",
            denoiser_dtype="float32",
            vae_dtype="float32",
            zero_negative_prompt=True,
            height=16,
            width=16,
            max_sequence_lengths=max_sequence_lengths,
            dynamic_shapes=dynamic_shapes,
            min_size=8 if dynamic_shapes else 0,
            max_size=16 if dynamic_shapes else 0,
            dynamic_spatial_multiple=8 if dynamic_shapes else 64,
        ).to_constant_methods()
        self.methods = {
            method: _Method(lambda value=value: value)
            for method, value in metadata.items()
        }
        self.methods.update(
            {
                "text_encoder_0": _Method(lambda ids: torch.ones(1, 77, 2)),
                "text_encoder_1": _Method(
                    lambda ids: (torch.ones(1, 77, 3), torch.ones(1, 2))
                ),
                "denoise": _Method(lambda latents, *args: torch.zeros_like(latents)),
                "decode": _Method(lambda latents: latents),
                "encode": _Method(
                    lambda image: (
                        torch.zeros(1, 4, 2, 2),
                        torch.zeros(1, 4, 2, 2),
                    )
                ),
            }
        )
        if pipeline == "sd15":
            del self.methods["text_encoder_1"]

    def load_method(self, name):
        return self.methods[name]

    def metadata(self, name):
        shape = (1, 3, 16, 16) if name == "encode" else (1, 4, 2, 2)
        return _MethodMeta(shape)


class _Tokenizer:
    model_max_length = 77

    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))


class _TokenizerWithMask(_Tokenizer):
    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(
            input_ids=torch.zeros(1, 77, dtype=torch.long),
            attention_mask=torch.ones(1, 77, dtype=torch.long),
        )


class FakeScheduler:
    init_noise_sigma = 1.0
    order = 1
    config = SimpleNamespace(prediction_type="epsilon")

    def set_timesteps(self, steps):
        self.timesteps = torch.arange(steps - 1, -1, -1)

    def scale_model_input(self, latents, timestep):
        return latents

    def step(
        self,
        prediction,
        timestep,
        latents,
        generator=None,
        return_dict=False,
    ):
        noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype)
        return (latents - prediction + noise,)

    def add_noise(self, latents, noise, timestep):
        return latents + noise


class RunnerTest(unittest.TestCase):
    def test_conditioning_cast_preserves_non_floating_dtypes(self):
        floating, integer, boolean = _cast_floating_tensors(
            (
                torch.ones(1, dtype=torch.float32),
                torch.ones(1, dtype=torch.int64),
                torch.ones(1, dtype=torch.bool),
            ),
            torch.float16,
        )
        self.assertEqual(
            (floating.dtype, integer.dtype, boolean.dtype),
            (torch.float16, torch.int64, torch.bool),
        )

    def test_tokenizer_outputs_form_component_specific_input_tuple(self):
        runner = DiffusionRunner(
            _Program("sd15"), [_TokenizerWithMask()], FakeScheduler()
        )
        runner.pipeline_spec = SimpleNamespace(
            text_encoder_input_names=(("input_ids", "attention_mask"),)
        )
        inputs = runner._tokenize("cat")[0]
        self.assertEqual(tuple(inputs[1].shape), (1, 77))

    def test_uses_exported_generation_defaults(self):
        scheduler = FakeScheduler()
        runner = DiffusionRunner(_Program(), [_Tokenizer(), _Tokenizer()], scheduler)
        runner.generate("cat")
        self.assertEqual(len(scheduler.timesteps), 50)

    def test_rejects_nonfinite_cfg_scale(self):
        runner = DiffusionRunner(
            _Program(), [_Tokenizer(), _Tokenizer()], FakeScheduler()
        )
        with self.assertRaisesRegex(ValueError, "cfg_scale"):
            runner.generate("cat", cfg_scale=float("nan"))

    def test_rejects_nonpositive_explicit_height(self):
        runner = DiffusionRunner(
            _Program(), [_Tokenizer(), _Tokenizer()], FakeScheduler()
        )
        with self.assertRaisesRegex(ValueError, "height must be a positive integer"):
            runner.generate("cat", height=0)

    def test_seed_is_deterministic(self):
        runner = DiffusionRunner(
            _Program(), [_Tokenizer(), _Tokenizer()], FakeScheduler()
        )
        first = runner.generate("cat", steps=2, seed=9)
        second = runner.generate("cat", steps=2, seed=9)
        torch.testing.assert_close(first, second)

    def test_image_validation(self):
        runner = DiffusionRunner(
            _Program(), [_Tokenizer(), _Tokenizer()], FakeScheduler()
        )
        with self.assertRaisesRegex(ValueError, "image shape"):
            runner.generate("cat", image=torch.zeros(1, 3, 8, 8))

    def test_image_to_image_selects_timesteps_and_adds_noise(self):
        class TrackingScheduler(FakeScheduler):
            def __init__(self):
                self.stepped_timesteps = []

            def set_begin_index(self, index):
                self.begin_index = index

            def add_noise(self, latents, noise, timestep):
                self.add_noise_inputs = (latents, noise, timestep)
                return super().add_noise(latents, noise, timestep)

            def step(self, prediction, timestep, latents, **kwargs):
                self.stepped_timesteps.append(int(timestep))
                return super().step(prediction, timestep, latents, **kwargs)

        scheduler = TrackingScheduler()
        runner = DiffusionRunner(_Program(), [_Tokenizer(), _Tokenizer()], scheduler)
        runner.generate(
            "cat",
            image=torch.zeros(1, 3, 16, 16),
            steps=4,
            strength=0.5,
        )
        image_latents, noise, noise_timestep = scheduler.add_noise_inputs
        self.assertEqual(scheduler.begin_index, 2)
        self.assertEqual(scheduler.stepped_timesteps, [1, 0])
        torch.testing.assert_close(noise_timestep, torch.tensor([1]))
        self.assertTrue(bool(torch.count_nonzero(noise)))
        self.assertEqual(noise.shape, image_latents.shape)

    def test_pipeline_decides_if_missing_negative_prompt_enables_cfg(self):
        class ExplicitNegativeSDXL(SDXLPipelineSpec):
            def should_use_cfg(self, cfg_scale, negative_prompt):
                return cfg_scale > 1 and negative_prompt is not None

        program = _Program()
        denoise = mock.Mock(
            side_effect=lambda latents, *args: torch.zeros_like(latents)
        )
        program.methods["denoise"] = _Method(denoise)
        runner = DiffusionRunner(program, [_Tokenizer(), _Tokenizer()], FakeScheduler())
        runner.pipeline_spec = ExplicitNegativeSDXL()
        runner.generate("cat", steps=1, cfg_scale=7.5)
        self.assertEqual(denoise.call_count, 1)

    def test_scheduler_class_can_change(self):
        program = _Program()
        program.methods[METADATA_METHODS["scheduler"]] = _Method(
            lambda: "OtherScheduler"
        )
        DiffusionRunner(program, [_Tokenizer(), _Tokenizer()], FakeScheduler())

    def test_scheduler_prediction_type_is_validated(self):
        program = _Program()
        program.methods[METADATA_METHODS["scheduler_prediction_type"]] = _Method(
            lambda: "v_prediction"
        )
        with self.assertRaisesRegex(ValueError, "prediction_type"):
            DiffusionRunner(program, [_Tokenizer(), _Tokenizer()], FakeScheduler())

    def test_sd15_uses_one_tokenizer_and_one_encoder(self):
        runner = DiffusionRunner(_Program("sd15"), [_Tokenizer()], FakeScheduler())
        image = runner.generate("cat", steps=2, seed=9)
        self.assertEqual(image.shape, (1, 4, 2, 2))

    def test_dynamic_program_accepts_another_resolution(self):
        runner = DiffusionRunner(
            _Program("sd15", dynamic_shapes=True), [_Tokenizer()], FakeScheduler()
        )
        image = runner.generate("cat", steps=2, height=8, width=16)
        self.assertEqual(image.shape, (1, 4, 1, 2))


if __name__ == "__main__":
    unittest.main()
