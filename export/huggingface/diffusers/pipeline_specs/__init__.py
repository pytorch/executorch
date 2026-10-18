# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supported Hugging Face diffusion pipeline export specifications."""

from .base import (
    DiffusionComponents,
    DiffusionModelInfo,
    DiffusionPipelineExportSpec,
    ResolvedDiffusionPipeline,
)
from .sd15 import SD15PipelineExportSpec
from .sdxl import SDXLPipelineExportSpec

HF_PIPELINE_EXPORT_SPECS: dict[str, type[DiffusionPipelineExportSpec]] = {
    "StableDiffusionPipeline": SD15PipelineExportSpec,
    "StableDiffusionXLPipeline": SDXLPipelineExportSpec,
}

__all__ = [
    "DiffusionComponents",
    "DiffusionModelInfo",
    "DiffusionPipelineExportSpec",
    "HF_PIPELINE_EXPORT_SPECS",
    "ResolvedDiffusionPipeline",
    "SD15PipelineExportSpec",
    "SDXLPipelineExportSpec",
]
