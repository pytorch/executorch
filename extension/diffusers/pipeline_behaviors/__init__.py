# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supported diffusion pipeline specifications and registry."""

from .base import DiffusionPipelineBehavior
from .sd15 import SD15PipelineBehavior
from .sdxl import SDXLPipelineBehavior

PIPELINE_BEHAVIORS: dict[str, DiffusionPipelineBehavior] = {
    SDXLPipelineBehavior.name: SDXLPipelineBehavior(),
    SD15PipelineBehavior.name: SD15PipelineBehavior(),
}

__all__ = [
    "DiffusionPipelineBehavior",
    "PIPELINE_BEHAVIORS",
    "SD15PipelineBehavior",
    "SDXLPipelineBehavior",
]
