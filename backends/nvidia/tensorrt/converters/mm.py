# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for matrix multiplication operations."""

from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import set_layer_name


@converter("aten.mm.default")
def convert_mm(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.mm.default to TensorRT MatrixMultiply.

    Performs matrix multiplication of two 2D tensors (MxK) @ (KxN) -> (MxN).
    """
    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    lhs = input_map[lhs_arg]
    rhs = input_map[rhs_arg]

    layer = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    )
    set_layer_name(layer, node, "mm")

    return layer.get_output(0)
