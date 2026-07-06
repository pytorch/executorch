# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Specialization registry and recipes for the native backend fat PTE.

Each recipe is a callable that takes an ExportedProgram and returns
a PreprocessResult.  The processed_bytes contain the delegate payload
and data_store_output (if any) carries externalized constants that
build_fat_result merges into the shared named-data store, enabling
cross-specialization dedup of identical weight data.

Automatically imported by NativePartitioner to populate the registry.
"""

from typing import Callable, Dict

from executorch.exir.backend.backend_details import PreprocessResult
from torch.export import ExportedProgram

SpecializationRecipe = Callable[[ExportedProgram], PreprocessResult]

_SPECIALIZATION_REGISTRY: Dict[str, SpecializationRecipe] = {}


def register_specialization(name: str, recipe: SpecializationRecipe) -> None:
    _SPECIALIZATION_REGISTRY[name] = recipe


# ---------------------------------------------------------------------------
# Built-in recipes
# ---------------------------------------------------------------------------


def _mlx_recipe(ep: ExportedProgram) -> PreprocessResult:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.lowered_backend_module import get_lowered_backend_modules

    lowered = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
    )
    lbms = get_lowered_backend_modules(
        lowered.exported_program().graph_module,
    )
    assert len(lbms) == 1, f"Expected 1 MLX delegate, got {len(lbms)}"
    lbm = lbms[0]
    return PreprocessResult(
        processed_bytes=lbm.processed_bytes,
        data_store_output=lbm.named_data_store_output,
    )


register_specialization("MLXBackend", _mlx_recipe)
