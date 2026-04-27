# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU operator support checks."""

from .axon_support import (
    AXON_CPU_ONLY_OPS,
    AXON_FUSED_ACTIVATIONS,
    AXON_OP_EXTENSIONS,
    AXON_SUPPORTED_OPS,
    check_conv2d,
    check_fully_connected,
    check_input_count,
    check_pooling,
    check_tensor_dimensions,
)

__all__ = [
    "AXON_SUPPORTED_OPS",
    "AXON_FUSED_ACTIVATIONS",
    "AXON_OP_EXTENSIONS",
    "AXON_CPU_ONLY_OPS",
    "check_fully_connected",
    "check_conv2d",
    "check_pooling",
    "check_tensor_dimensions",
    "check_input_count",
]
