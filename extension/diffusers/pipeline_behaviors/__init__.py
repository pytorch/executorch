# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supported diffusion pipeline specifications and registry."""

from .base import DiffusionPipelineBehavior
from .flux import FluxPipelineBehavior
from .flux2 import Flux2PipelineBehavior
from .flux2_klein import Flux2KleinPipelineBehavior
from .sd15 import SD15PipelineBehavior
from .sdxl import SDXLPipelineBehavior

PIPELINE_BEHAVIORS: dict[str, DiffusionPipelineBehavior] = {
    FluxPipelineBehavior.name: FluxPipelineBehavior(),
    Flux2PipelineBehavior.name: Flux2PipelineBehavior(),
    Flux2KleinPipelineBehavior.name: Flux2KleinPipelineBehavior(),
    SDXLPipelineBehavior.name: SDXLPipelineBehavior(),
    SD15PipelineBehavior.name: SD15PipelineBehavior(),
}

__all__ = [
    "DiffusionPipelineBehavior",
    "FluxPipelineBehavior",
    "Flux2PipelineBehavior",
    "Flux2KleinPipelineBehavior",
    "PIPELINE_BEHAVIORS",
    "SD15PipelineBehavior",
    "SDXLPipelineBehavior",
]
