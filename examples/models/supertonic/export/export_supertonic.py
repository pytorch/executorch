# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.export import ExportedProgram, export

from . import common
from ..model.config import TTSConfig
from ..source_transformations.mlx import (
    exportable_vector_estimator,
    replace_relative_attention,
    replace_same_padding,
    replace_vocoder_causal_padding,
)


def export_programs(
    models: Mapping[str, nn.Module],
    config: TTSConfig,
    bounds: common.ExportBounds,
    *,
    flow_steps: int = common.DEFAULT_FLOW_STEPS,
) -> dict[str, ExportedProgram]:
    common.validate_flow_steps(flow_steps)
    expected_methods = set(common.METHOD_NAMES)
    if set(models) != expected_methods:
        missing = sorted(expected_methods - set(models))
        extra = sorted(set(models) - expected_methods)
        raise ValueError(
            f"models must contain the exact method set; missing={missing}, extra={extra}"
        )
    samples = common.example_inputs(config, bounds, flow_steps=flow_steps)
    shapes = common.dynamic_shapes(bounds)
    programs = {}
    with torch.no_grad():
        for method_name in common.METHOD_NAMES:
            model = replace_same_padding(models[method_name].eval())
            if method_name in ("duration_predictor", "text_encoder"):
                model = replace_relative_attention(model)
            elif method_name == "vector_estimator":
                model = exportable_vector_estimator(model)
            elif method_name == "vocoder":
                model = replace_vocoder_causal_padding(model)
            programs[method_name] = export(
                model,
                samples[method_name],
                dynamic_shapes=shapes[method_name],
                strict=True,
            )
    return programs


def lower_to_mlx(
    programs: Mapping[str, ExportedProgram],
    metadata: Mapping[str, object],
):
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import (
        EdgeCompileConfig,
        to_edge_transform_and_lower,
    )

    return to_edge_transform_and_lower(
        dict(programs),
        transform_passes=get_default_passes(),
        partitioner={method_name: [MLXPartitioner()] for method_name in programs},
        # MLX lowers model buffers such as the vector-estimator frequencies as
        # delegate constants. Edge validation and dim-order rewriting currently
        # look them up as graph state and fail before partitioning.
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=dict(metadata),
    )


def to_executorch(edge_program):
    from executorch.exir import ExecutorchBackendConfig

    return edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )


def export_from_assets(
    asset_dir: str | Path,
    output_path: str | Path,
    *,
    bounds: common.ExportBounds = common.ExportBounds(),
    flow_steps: int = common.DEFAULT_FLOW_STEPS,
) -> Path:
    config, models = common.load_models(asset_dir)
    vocabulary_size = common.text_vocabulary_size(models)
    common.convert_models_to_fp16(models)
    programs = export_programs(models, config, bounds, flow_steps=flow_steps)
    edge_program = lower_to_mlx(
        programs,
        common.runtime_metadata(
            config,
            bounds,
            text_vocabulary_size=vocabulary_size,
            flow_steps=flow_steps,
        ),
    )
    return common.save_pte(to_executorch(edge_program), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one dynamic FP16 Supertonic MLX artifact."
    )
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-latent-length", type=int, default=512)
    parser.add_argument(
        "--flow-steps",
        type=int,
        choices=(common.DEFAULT_FLOW_STEPS,),
        default=common.DEFAULT_FLOW_STEPS,
    )
    args = parser.parse_args()
    bounds = common.ExportBounds(
        text_max=args.max_text_length,
        latent_max=args.max_latent_length,
    )
    export_from_assets(
        args.asset_dir,
        args.output,
        bounds=bounds,
        flow_steps=args.flow_steps,
    )


if __name__ == "__main__":
    main()
