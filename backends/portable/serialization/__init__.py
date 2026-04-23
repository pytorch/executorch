# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Serialization utilities for Portable Backend.

The Portable Backend reuses the standard ExecuTorch serialization format
(ExecutionPlan in program.fbs). This module re-exports the relevant schema
classes for convenience.

For memory aliasing, tensors with the same AllocationDetails.memory_id
share the same storage pool slot. This is computed by the memory planning
pass and preserved in the serialized program.
"""

# Re-export standard ExecuTorch schema classes
from executorch.exir.schema import (
    AllocationDetails,
    EValue,
    ExecutionPlan,
    Instruction,
    KernelCall,
    Operator,
    Program,
    Tensor,
)

__all__ = [
    "AllocationDetails",
    "EValue",
    "ExecutionPlan",
    "Instruction",
    "KernelCall",
    "Operator",
    "Program",
    "Tensor",
]
