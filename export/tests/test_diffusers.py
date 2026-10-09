# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Hugging Face Diffusers export."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
from executorch.export.huggingface.diffusers.export import (
    _resolve_max_sequence_lengths,
    _validate_text_encoder_lengths,
    export_programs,
    main,
    write_pte,
)
from executorch.export.huggingface.diffusers.pipeline_discovery import (
    pipeline_spec_from_model_index,
    resolve_huggingface_pipeline,
)

from executorch.extension.diffusers.config import (
    DiffusionConfig,
    ExportConfig,
    ModelConfig,
)
from executorch.extension.diffusers.pipeline_specs import (
    DiffusionComponents,
    DiffusionModelInfo,
    ResolvedDiffusionPipeline,
    SD15PipelineSpec,
    SDXLPipelineSpec,
)


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
            default_max_sequence_lengths=(77, 77, 256),
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
            _validate_text_encoder_lengths(components, SD15PipelineSpec(), (77,))

    def test_sdxl_exports_dynamic_spatial_shapes(self):
        export_config = config(
            128,
            128,
            (77, 77),
            dynamic_shapes=True,
            min_size=64,
            max_size=256,
        )
        programs = export_programs(self.components, SDXLPipelineSpec(), export_config)
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
        programs = export_programs(components, SD15PipelineSpec(), config(8, 8, (77,)))
        self.assertEqual(
            set(programs), {"text_encoder_0", "denoise", "decode", "encode"}
        )

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
        denoise = export_programs(components, SD15PipelineSpec(), export_config)[
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
        self.assertIsInstance(sd15, SD15PipelineSpec)
        self.assertIsInstance(sdxl, SDXLPipelineSpec)

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
        self.assertEqual(pipeline.spec.tokenizer_slots, ("tokenizer", "tokenizer_2"))
        self.assertEqual(
            pipeline.spec.text_encoder_slots,
            ("text_encoder", "text_encoder_2"),
        )
        self.assertEqual(pipeline.model_info.scheduler_name, "EulerDiscreteScheduler")
        self.assertEqual(pipeline.model_info.scheduler_prediction_type, "epsilon")
        self.assertEqual(pipeline.model_info.max_sequence_lengths, (77, 77))
        self.assertEqual(pipeline.spec.default_size, (1024, 1024))
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
            SD15PipelineSpec(), max_sequence_lengths=(77,)
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
            SD15PipelineSpec(), max_sequence_lengths=(77,)
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


if __name__ == "__main__":
    unittest.main()
