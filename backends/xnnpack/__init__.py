# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exposed Partitioners in XNNPACK Package
from .partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    XnnpackPartitioner,
)

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
    "capture_graph_for_xnnpack",
    "get_xnnpack_capture_config",
    "get_xnnpack_edge_compile_config",
    "get_xnnpack_executorch_backend_config",
]
