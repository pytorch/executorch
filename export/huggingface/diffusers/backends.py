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


@experimental("This API is experimental and may change without notice.")
def export_to_cuda(
    model_id: str,
    output_path: str | Path,
    *,
    config: DiffusionConfig,
    revision: str | None = None,
    pipeline: ResolvedDiffusionPipeline | None = None,
    qlinear: str | None = None,
    qlinear_group_size: int | None = None,
) -> None:
    """Discover and export a Hugging Face diffusion pipeline to CUDA."""
    from executorch.backends.cuda.aoti_packed_int4_tensor import (
        pack_int4_weights_for_aoti,
    )
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.llm.export.quantize import quantize_model_

    def quantize(components, _pipeline):
        if qlinear is not None:
            denoiser = components.require("denoiser")
            quantize_model_(
                denoiser,
                qlinear_config=qlinear,
                qlinear_group_size=qlinear_group_size,
                skip_incompatible_shapes=True,
            )
            if qlinear == "4w":
                pack_int4_weights_for_aoti(denoiser)
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
        partitioners={
            name: [
                CudaPartitioner([CudaBackend.generate_method_name_compile_spec(name)])
            ]
            for name in programs
        },
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        backend_config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            do_quant_fusion_and_const_prop=True,
        ),
    )


@experimental("This API is experimental and may change without notice.")
def export_to_coreai(
    model_id: str,
    output_path: str | Path,
    *,
    config: DiffusionConfig,
    revision: str | None = None,
    pipeline: ResolvedDiffusionPipeline | None = None,
) -> None:
    """Export static Hugging Face diffusion component methods to CoreAI."""
    import torch

    if config.export.dynamic_shapes:
        raise ValueError("CoreAI diffusion export currently requires static shapes")
    if any(
        dtype not in (torch.float16, torch.float32)
        for dtype in (
            config.model.text_encoder_dtype,
            config.model.denoiser_dtype,
            config.model.vae_dtype,
        )
    ):
        raise ValueError("CoreAI components must use float16 or float32")

    from executorch.backends.apple.coreai import (
        CoreAIPartitioner,
        get_default_compile_config,
        get_default_passes,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    programs, pipeline = export_huggingface(
        model_id, revision=revision, config=config, pipeline=pipeline
    )
    lower_and_write_pte(
        programs,
        pipeline,
        config,
        output_path,
        partitioners={name: [CoreAIPartitioner()] for name in programs},
        transform_passes=get_default_passes(),
        edge_compile_config=get_default_compile_config(),
        backend_config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
