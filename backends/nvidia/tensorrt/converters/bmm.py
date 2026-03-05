# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for batch matrix multiplication operations."""

from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import set_layer_name


@converter("aten.bmm.default")
def convert_bmm(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert aten.bmm.default to TensorRT MatrixMultiply.

    Performs batch matrix multiplication of two 3D tensors (B, M, K) @ (B, K, N) -> (B, M, N).
    TensorRT's IMatrixMultiplyLayer supports batch matrix multiplication natively.
    """
    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    if lhs_arg not in input_map:
        raise ValueError(f"Input node '{lhs_arg.name}' not found in input_map for bmm")
    if rhs_arg not in input_map:
        raise ValueError(f"Input node '{rhs_arg.name}' not found in input_map for bmm")

    lhs = input_map[lhs_arg]
    rhs = input_map[rhs_arg]

    layer = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    )
    set_layer_name(layer, node, "bmm")

    return layer.get_output(0)
