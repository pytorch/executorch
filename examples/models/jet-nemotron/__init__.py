# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JetBlock example for ExecutorTorch.

JetBlock is a recurrent attention mechanism from NVIDIA's Jet-Nemotron model,
using the Gated Delta Rule for efficient linear attention with O(1) memory
during inference.

This example demonstrates how to use the JetBlock style in ExecutorTorch.
"""

from executorch.examples.models.jet_nemotron.model import JetBlockModel

__all__ = [
    "JetBlockModel",
]
