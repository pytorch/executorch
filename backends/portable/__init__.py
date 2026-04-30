# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Portable Backend

A unified execution backend that can dispatch ops across multiple runtimes
(CPU, Metal, Vulkan) at runtime initialization based on device capabilities.

Key features:
1. Uses the standard ExecuTorch ExecutionPlan format (no custom serialization)
2. Runtime partitioning happens in C++ based on has_op() queries
3. Supports automatic memory aliasing via AllocationDetails.memory_id
4. Reuses existing portable ops from kernels/portable/cpu/

Usage:
    from executorch.exir import to_edge
    from executorch.backends.portable import PortablePartitioner
    
    edge_program = to_edge(exported_program)
    portable_program = edge_program.to_backend(PortablePartitioner())
"""

from .partitioner.portable_partitioner import PortablePartitioner
from .preprocess import PortableBackend

__all__ = [
    "PortablePartitioner",
    "PortableBackend",
]
