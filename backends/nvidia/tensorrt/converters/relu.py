# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for ReLU activation operations."""

from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import set_layer_name


@converter("aten.relu.default", "aten.relu_.default")
def convert_relu(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.relu.default and aten.relu_.default to TensorRT Activation RELU.

    ReLU(x) = max(0, x)
    Note: In-place variant (relu_) is handled identically since TensorRT doesn't
    have in-place operations.
    """
    input_arg = node.args[0]
    input_tensor = input_map[input_arg]

    layer = network.add_activation(input_tensor, trt.ActivationType.RELU)
    set_layer_name(layer, node, "relu")

    return layer.get_output(0)
