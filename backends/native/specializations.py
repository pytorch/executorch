# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Specialization registry and recipes for the native backend fat PTE.

Each recipe is a callable that takes an ExportedProgram and returns
serialized bytes.  Automatically imported by NativePartitioner to
populate the registry.
"""

from typing import Callable, Dict

from torch.export import ExportedProgram

SpecializationRecipe = Callable[[ExportedProgram], bytes]

_SPECIALIZATION_REGISTRY: Dict[str, SpecializationRecipe] = {}


def register_specialization(name: str, recipe: SpecializationRecipe) -> None:
    _SPECIALIZATION_REGISTRY[name] = recipe


# ---------------------------------------------------------------------------
# Built-in recipes
# ---------------------------------------------------------------------------


def _mlx_recipe(ep: ExportedProgram) -> bytes:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower

    lowered = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
    )
    et_program = lowered.to_executorch()
    return et_program.buffer


register_specialization("MLXBackend", _mlx_recipe)
