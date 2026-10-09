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

from .config import DiffusionConfig, ExportConfig, ModelConfig
from .metadata import build_diffusion_metadata
from .pipeline_discovery import resolve_huggingface_pipeline
from .pipeline_specs import (
    DiffusionComponents,
    DiffusionPipelineSpec,
    ResolvedDiffusionPipeline,
)

ExportedPrograms = dict[str, torch.export.ExportedProgram]


def _resolve_max_sequence_lengths(
    pipeline_spec: DiffusionPipelineSpec,
    override: int | None,
) -> tuple[int, ...]:
    lengths = list(pipeline_spec.default_max_sequence_lengths)
    if override is None:
        return tuple(lengths)
    if override <= 0:
        raise ValueError("max_sequence_length must be positive")
    index = pipeline_spec.max_sequence_length_override_index
    if index is None:
        raise ValueError(
            f"{pipeline_spec.name} does not support overriding sequence length"
        )
    lengths[index] = override
    return tuple(lengths)


def _validate_text_encoder_lengths(
    components: DiffusionComponents,
    pipeline_spec: DiffusionPipelineSpec,
    lengths: tuple[int, ...],
) -> None:
    if len(lengths) != len(pipeline_spec.text_encoder_slots):
        raise ValueError(
            f"{pipeline_spec.name} requires {len(pipeline_spec.text_encoder_slots)} "
            f"text encoder sequence lengths, found {len(lengths)}"
        )
    for slot, length in zip(pipeline_spec.text_encoder_slots, lengths):
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
    pipeline_spec: DiffusionPipelineSpec,
    config: DiffusionConfig,
) -> ExportedPrograms:
    """Export component methods without choosing a backend."""
    modules = pipeline_spec.wrap_components(components, config)
    inputs = pipeline_spec.example_inputs(components, config)
    dynamic_shapes = pipeline_spec.dynamic_shapes(components, config)
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
            pipeline.spec, pipeline.model_info, config
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
    components = pipeline.spec.load_components(model_id, revision, config)
    _validate_text_encoder_lengths(
        components, pipeline.spec, config.model.max_sequence_lengths
    )
    if component_transform is not None:
        components = component_transform(components, pipeline)
    pipeline = replace(
        pipeline,
        model_info=pipeline.spec.resolve_model_info(components, pipeline.model_info),
    )
    return export_programs(components, pipeline.spec, config), pipeline


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
    height = args.height if args.height is not None else pipeline.spec.default_size[0]
    width = args.width if args.width is not None else pipeline.spec.default_size[1]
    max_sequence_lengths = _resolve_max_sequence_lengths(
        pipeline.spec, args.max_sequence_length
    )

    if args.backend == "mlx":
        from .backends import export_to_mlx

        export_to_mlx(
            args.model_id,
            args.output_path,
            revision=args.revision,
            pipeline=pipeline,
            config=DiffusionConfig(
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
            ),
            qlinear=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
        )


if __name__ == "__main__":
    main()
