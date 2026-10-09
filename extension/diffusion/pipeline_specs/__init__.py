# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supported diffusion pipeline specifications and registry."""

from .base import (
    DiffusionComponents,
    DiffusionModelInfo,
    DiffusionPipelineSpec,
    ResolvedDiffusionPipeline,
)
from .sd15 import SD15PipelineSpec
from .sdxl import SDXLPipelineSpec

PIPELINE_SPECS: dict[str, DiffusionPipelineSpec] = {
    SDXLPipelineSpec.name: SDXLPipelineSpec(),
    SD15PipelineSpec.name: SD15PipelineSpec(),
}

HF_PIPELINE_SPECS: dict[str, type[SD15PipelineSpec] | type[SDXLPipelineSpec]] = {
    "StableDiffusionPipeline": SD15PipelineSpec,
    "StableDiffusionXLPipeline": SDXLPipelineSpec,
}

__all__ = [
    "DiffusionComponents",
    "DiffusionModelInfo",
    "DiffusionPipelineSpec",
    "PIPELINE_SPECS",
    "HF_PIPELINE_SPECS",
    "ResolvedDiffusionPipeline",
    "SD15PipelineSpec",
    "SDXLPipelineSpec",
]
