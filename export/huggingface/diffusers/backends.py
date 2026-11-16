# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delegate-specific entry points for experimental diffusion export."""

from pathlib import Path

from executorch.exir._warnings import experimental

from .config import DiffusionConfig
from .export import export_huggingface, lower_and_write_pte
from .pipeline_specs import ResolvedDiffusionPipeline


@experimental("This API is experimental and may change without notice.")
def export_to_mlx(
    model_id: str,
    output_path: str | Path,
    *,
    config: DiffusionConfig,
    revision: str | None = None,
    pipeline: ResolvedDiffusionPipeline | None = None,
    qlinear: str | None = None,
    qlinear_group_size: int | None = None,
) -> None:
    """Discover and export a Hugging Face diffusion pipeline to MLX."""
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.llm.quantization import quantize_model_
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    def quantize(components, _pipeline):
        if qlinear is not None:
            quantize_model_(
                components.require("denoiser"),
                qlinear_config=qlinear,
                qlinear_group_size=qlinear_group_size,
                skip_incompatible_shapes=True,
            )
        return components

    programs, pipeline = export_huggingface(
        model_id,
        revision=revision,
        config=config,
        pipeline=pipeline,
        component_transform=quantize,
    )
    lower_and_write_pte(
        programs,
        pipeline,
        config,
        output_path,
        partitioners={name: [MLXPartitioner()] for name in programs},
        transform_passes=get_default_passes(),
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        backend_config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
