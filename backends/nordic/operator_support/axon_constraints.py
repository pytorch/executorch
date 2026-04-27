# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU hardware constraint checks for the ExecuTorch partitioner.

These ``OperatorSupportBase`` subclasses are passed as ``additional_checks``
to the ``TOSAPartitioner``. They reject nodes that TOSA accepts but AXON
cannot execute due to hardware limits (tensor size, input count, filter
dimensions, etc.).

Without these checks, over-sized operations would be delegated to AXON
and fail at compile time instead of falling back to CPU gracefully.
"""
from __future__ import annotations

import typing

import torch
from torch.fx import Node
from torch.fx.passes.operator_support import OperatorSupportBase

from executorch.backends.nordic.axon.compile_spec import (
    AXON_MAX_CONV2D_FILTER,
    AXON_MAX_CONV_STRIDE,
    AXON_MAX_FC_INPUT,
    AXON_MAX_FC_OUTPUT,
    AXON_MAX_INPUTS_PER_NODE,
    AXON_MAX_POOL_FILTER,
    AXON_MAX_TENSOR_DIM,
)


def _get_tensor_shape(node: Node) -> list[int] | None:
    """Extract the output tensor shape from a node's metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return list(val.shape)
    if hasattr(val, "shape"):
        return list(val.shape)
    return None


class AxonTensorDimensionCheck(OperatorSupportBase):
    """Reject nodes whose output tensors exceed AXON's max dimensions.

    AXON supports max 1024 for height, width, and channels.
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op != "call_function":
            return True
        shape = _get_tensor_shape(node)
        if shape is None:
            return True  # Can't determine shape; let TOSA checks handle it
        for dim in shape:
            if dim > AXON_MAX_TENSOR_DIM:
                return False
        return True


class AxonInputCountCheck(OperatorSupportBase):
    """Reject nodes with more than 2 activation tensor inputs.

    AXON allows a maximum of 2 inputs per node. This counts only
    activation (non-constant) tensor inputs — weight tensors and
    scalar parameters don't count toward this limit.
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op != "call_function":
            return True
        # Count Node args that are tensor-producing (activation inputs).
        # Skip scalar args, None args, and list/tuple args.
        tensor_inputs = 0
        for arg in node.args:
            if isinstance(arg, Node) and arg.op == "call_function":
                tensor_inputs += 1
        # Only reject if clearly over the limit
        if tensor_inputs > AXON_MAX_INPUTS_PER_NODE:
            return False
        return True


class AxonConvConstraintCheck(OperatorSupportBase):
    """Reject convolution nodes that exceed AXON's filter/stride limits.

    Conv2D: max 16x16 filter, max stride 31.
    Only rejects when we can definitively determine the constraint is
    violated. Returns True (allow) when metadata is unavailable.
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op != "call_function":
            return True

        target_name = str(node.target)

        # Check convolution constraints
        if any(op in target_name for op in ("conv2d", "convolution")):
            # Weight tensor is typically args[1]
            if len(node.args) >= 2 and isinstance(node.args[1], Node):
                weight_shape = _get_tensor_shape(node.args[1])
                if weight_shape and len(weight_shape) == 4:
                    kH, kW = weight_shape[2], weight_shape[3]
                    if kH > AXON_MAX_CONV2D_FILTER or kW > AXON_MAX_CONV2D_FILTER:
                        return False

            # Check stride if available
            if len(node.args) >= 4:
                stride = node.args[3]
                if isinstance(stride, (list, tuple)) and len(stride) >= 2:
                    if stride[0] > AXON_MAX_CONV_STRIDE or stride[1] > AXON_MAX_CONV_STRIDE:
                        return False

        return True


class AxonFCConstraintCheck(OperatorSupportBase):
    """Reject fully connected nodes that exceed AXON's size limits.

    FC max input: 2048 elements, max output: 2048 elements.
    Only rejects when we can definitively determine the constraint is
    violated. Returns True (allow) when metadata is unavailable.
    """

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        if node.op != "call_function":
            return True

        target_name = str(node.target)
        if "linear" not in target_name and "addmm" not in target_name:
            return True

        # Check input shape (last dim is feature size)
        if node.args and isinstance(node.args[0], Node):
            input_shape = _get_tensor_shape(node.args[0])
            if input_shape and input_shape[-1] > AXON_MAX_FC_INPUT:
                return False

        # Check output shape
        output_shape = _get_tensor_shape(node)
        if output_shape and output_shape[-1] > AXON_MAX_FC_OUTPUT:
            return False

        return True


def get_axon_constraint_checks() -> list[OperatorSupportBase]:
    """Return all AXON hardware constraint checks.

    Pass these as ``additional_checks`` to ``AxonPartitioner`` to ensure
    that only AXON-compatible operations are delegated.
    """
    return [
        AxonTensorDimensionCheck(),
        AxonInputCountCheck(),
        AxonConvConstraintCheck(),
        AxonFCConstraintCheck(),
    ]
