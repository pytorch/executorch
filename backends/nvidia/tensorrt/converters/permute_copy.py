# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for permute_copy (tensor transpose/permutation) operations."""

from typing import Any, Dict, List

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import set_layer_name


@converter("aten.permute_copy.default")
def convert_permute_copy(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.permute_copy.default to TensorRT Shuffle layer.

    Permutes the dimensions of the input tensor according to the given order.
    For nn.Linear, this is used to transpose the weight matrix.
    """
    input_arg = node.args[0]
    dims_arg = node.args[1]

    input_tensor = input_map[input_arg]

    # dims_arg is a list of integers specifying the new dimension order
    dims: List[int] = list(dims_arg)

    # Use TensorRT's shuffle layer with second_transpose for permutation
    layer = network.add_shuffle(input_tensor)
    layer.second_transpose = trt.Permutation(dims)
    set_layer_name(layer, node, "permute_copy")

    return layer.get_output(0)
