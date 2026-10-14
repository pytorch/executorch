# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-neutral staged export for experimental diffusion pipelines."""

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from executorch.exir._warnings import experimental
from executorch.extension.diffusers.metadata import DiffusionMetadata
from executorch.extension.diffusers.pipeline_behaviors import DiffusionPipelineBehavior

from .config import DiffusionConfig, ExportConfig, ModelConfig

from .pipeline_discovery import resolve_huggingface_pipeline
from .pipeline_specs import (
    DiffusionComponents,
    DiffusionModelInfo,
    DiffusionPipelineExportSpec,
    ResolvedDiffusionPipeline,
)

ExportedPrograms = dict[str, torch.export.ExportedProgram]


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


def _resolve_max_sequence_lengths(
    export_spec: DiffusionPipelineExportSpec,
    override: int | None,
) -> tuple[int, ...]:
    lengths = list(export_spec.default_max_sequence_lengths)
    if override is None:
        return tuple(lengths)
    if override <= 0:
        raise ValueError("max_sequence_length must be positive")
    index = export_spec.max_sequence_length_override_index
    if index is None:
        raise ValueError(
            f"{export_spec.pipeline_behavior.name} does not support overriding "
            "sequence length"
        )
    lengths[index] = override
    return tuple(lengths)


def _validate_text_encoder_lengths(
    components: DiffusionComponents,
    pipeline_behavior: DiffusionPipelineBehavior,
    lengths: tuple[int, ...],
) -> None:
    if len(lengths) != len(pipeline_behavior.text_encoder_slots):
        raise ValueError(
            f"{pipeline_behavior.name} requires "
            f"{len(pipeline_behavior.text_encoder_slots)} "
            f"text encoder sequence lengths, found {len(lengths)}"
        )
    for slot, length in zip(pipeline_behavior.text_encoder_slots, lengths):
        config = components.require(slot).config
        positional_limit = getattr(config, "max_position_embeddings", None)
        if isinstance(positional_limit, int) and length > positional_limit:
            raise ValueError(
                f"text encoder {slot!r} supports {positional_limit} positions, "
                f"but export requires {length}"
            )


@experimental("This API is experimental and may change without notice.")
def export_programs(
    components: DiffusionComponents,
    export_spec: DiffusionPipelineExportSpec,
    config: DiffusionConfig,
) -> ExportedPrograms:
    """Export component methods without choosing a backend."""
    modules = export_spec.wrap_components(components, config)
    inputs = export_spec.example_inputs(components, config)
    dynamic_shapes = export_spec.dynamic_shapes(components, config)
    with torch.no_grad():
        return {
            name: torch.export.export(
                modules[name], inputs[name], dynamic_shapes=dynamic_shapes[name]
            )
            for name in modules
        }


@experimental("This API is experimental and may change without notice.")
def lower_programs(
    programs: Mapping[str, torch.export.ExportedProgram],
    pipeline: ResolvedDiffusionPipeline,
    config: DiffusionConfig,
    *,
    partitioners: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
    transform_passes: Sequence[Any] | None = None,
    edge_compile_config: Any | None = None,
):
    """Lower exported programs with caller-owned passes and partitioners."""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    if partitioners is not None and not isinstance(partitioners, Mapping):
        partitioners = {name: list(partitioners) for name in programs}
    return to_edge_transform_and_lower(
        dict(programs),
        transform_passes=list(transform_passes or ()),
        partitioner=partitioners,
        compile_config=edge_compile_config or EdgeCompileConfig(),
        constant_methods=build_diffusion_metadata(
            pipeline.behavior, pipeline.model_info, config
        ).to_constant_methods(),
    )


@experimental("This API is experimental and may change without notice.")
def write_pte(
    edge_program: Any, output_path: str | Path, *, backend_config=None
) -> None:
    """Serialize an edge program and propagate write failures."""
    output_path = Path(output_path).with_suffix(".pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    program = edge_program.to_executorch(config=backend_config)
    with output_path.open("wb") as file:
        program.write_to_file(file)
    program.write_tensor_data_to_file(outdir=str(output_path.parent))
    print(f"Saved PTE to {output_path.resolve()}")


@experimental("This API is experimental and may change without notice.")
def lower_and_write_pte(
    programs: Mapping[str, torch.export.ExportedProgram],
    pipeline: ResolvedDiffusionPipeline,
    config: DiffusionConfig,
    output_path: str | Path,
    *,
    partitioners: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
    transform_passes: Sequence[Any] | None = None,
    edge_compile_config: Any | None = None,
    backend_config: Any | None = None,
) -> None:
    """Apply delegate lowering and serialize a multi-method PTE."""
    edge_program = lower_programs(
        programs,
        pipeline,
        config,
        partitioners=partitioners,
        transform_passes=transform_passes,
        edge_compile_config=edge_compile_config,
    )
    write_pte(edge_program, output_path, backend_config=backend_config)


@experimental("This API is experimental and may change without notice.")
def export_huggingface(
    model_id: str,
    *,
    revision: str | None = None,
    config: DiffusionConfig,
    pipeline: ResolvedDiffusionPipeline | None = None,
    component_transform: (
        Callable[[DiffusionComponents, ResolvedDiffusionPipeline], DiffusionComponents]
        | None
    ) = None,
) -> tuple[ExportedPrograms, ResolvedDiffusionPipeline]:
    """Discover and export a supported Hugging Face diffusion pipeline."""
    pipeline = pipeline or resolve_huggingface_pipeline(model_id, revision)
    components = pipeline.export_spec.load_components(model_id, revision, config)
    _validate_text_encoder_lengths(
        components, pipeline.behavior, config.model.max_sequence_lengths
    )
    if component_transform is not None:
        components = component_transform(components, pipeline)
    pipeline = replace(
        pipeline,
        model_info=pipeline.export_spec.resolve_model_info(
            components, pipeline.model_info
        ),
    )
    return export_programs(components, pipeline.export_spec, config), pipeline


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a diffusion model to PTE")
    parser.add_argument("model_id")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--backend", choices=("mlx",), required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument(
        "--text-encoder-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--denoiser-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--revision")
    parser.add_argument("--qlinear")
    parser.add_argument("--qlinear-group-size", type=int)
    parser.add_argument("--dynamic-shapes", action="store_true")
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args(argv)

    pipeline = resolve_huggingface_pipeline(args.model_id, args.revision)
    height = (
        args.height if args.height is not None else pipeline.export_spec.default_size[0]
    )
    width = (
        args.width if args.width is not None else pipeline.export_spec.default_size[1]
    )
    max_sequence_lengths = _resolve_max_sequence_lengths(
        pipeline.export_spec, args.max_sequence_length
    )

    config = DiffusionConfig(
        model=ModelConfig(
            height=height,
            width=width,
            max_sequence_lengths=max_sequence_lengths,
            text_encoder_dtype=getattr(torch, args.text_encoder_dtype),
            denoiser_dtype=getattr(torch, args.denoiser_dtype),
            vae_dtype=getattr(torch, args.vae_dtype),
        ),
        export=ExportConfig(
            dynamic_shapes=args.dynamic_shapes,
            min_size=args.min_size,
            max_size=args.max_size,
        ),
    )

    if args.backend == "mlx":
        from .backends import export_to_mlx

        export_to_mlx(
            args.model_id,
            args.output_path,
            revision=args.revision,
            pipeline=pipeline,
            config=config,
            qlinear=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
        )


if __name__ == "__main__":
    main()
