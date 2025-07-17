# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import recipe_registry

# Exposed Partitioners in XNNPACK Package
from .partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    XnnpackPartitioner,
)
from .recipes.xnnpack_recipe_provider import XNNPACKRecipeProvider
from .recipes.xnnpack_recipe_types import XNNPackRecipeType

# Auto-register XNNPACK recipe provider
recipe_registry.register_backend_recipe_provider(XNNPACKRecipeProvider())

# Exposed Configs in XNNPACK Package
from .utils.configs import (
    get_xnnpack_capture_config,
    get_xnnpack_edge_compile_config,
    get_xnnpack_executorch_backend_config,
)

# Easy util functions
from .utils.utils import capture_graph_for_xnnpack

# XNNPACK Backend
from .xnnpack_preprocess import XnnpackBackend

__all__ = [
    "XnnpackDynamicallyQuantizedPartitioner",
    "XnnpackPartitioner",
    "XnnpackBackend",
    "XNNPackRecipeType",
    "capture_graph_for_xnnpack",
    "get_xnnpack_capture_config",
    "get_xnnpack_edge_compile_config",
    "get_xnnpack_executorch_backend_config",
]
