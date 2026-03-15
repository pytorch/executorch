# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Unary Operations.

Supported operations:
- aten.sqrt.default: Square root
- aten.rsqrt.default: Reciprocal square root (1/sqrt(x))
- aten.exp.default: Exponential
- aten.log.default: Natural logarithm
- aten.neg.default: Negation
- aten.abs.default: Absolute value
- aten.sin.default: Sine
- aten.cos.default: Cosine
- aten.floor.default: Floor
- aten.ceil.default: Ceiling
- aten.erf.default: Error function
- aten.reciprocal.default: Reciprocal (1/x)

These operations are commonly used in transformer models for layer normalization
and attention mechanisms. Design patterns follow TensorRT best practices.
"""

import logging
from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_unary(node: torch.fx.Node) -> bool:
    """Validate that a unary node can be converted to TensorRT."""
    if node.op != "call_function":
        return False
    if len(node.args) < 1:
        return False
    if not isinstance(node.args[0], torch.fx.Node):
        return False
    return True


def _cast_to_float_if_needed(
    network: trt.INetworkDefinition,
    input_trt: trt.ITensor,
    node_name: str,
) -> trt.ITensor:
    """Cast integer tensors to float32 for operations that require float input.

    TensorRT unary operations like sqrt, exp, log, sin, cos, etc. don't support
    int8 or int32 input types. This follows the TensorRT pattern of
    automatically casting to float32.

    Args:
        network: TensorRT network definition.
        input_trt: Input tensor.
        node_name: Node name for layer naming.

    Returns:
        Input tensor (casted if necessary).
    """
    if input_trt.dtype in (trt.int8, trt.int32):
        cast_layer = network.add_cast(input_trt, trt.float32)
        if cast_layer is None:
            raise RuntimeError(f"Failed to create cast layer for {node_name}")
        cast_layer.name = f"cast_to_float_{node_name}"
        return cast_layer.get_output(0)
    return input_trt


def _convert_unary_base(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    operation_type: trt.UnaryOperation,
    op_name: str,
    cast_to_float: bool = True,
) -> trt.ITensor:
    """Base function for unary operation conversion.

    This follows the TensorRT architecture pattern where a base function
    handles common logic (input validation, optional casting, layer creation)
    and specific converters can call this with appropriate parameters.

    Args:
        node: FX node to convert.
        network: TensorRT network definition.
        input_map: Map of FX nodes to TensorRT tensors.
        operation_type: TensorRT unary operation type.
        op_name: Name of the operation for logging/layer naming.
        cast_to_float: Whether to cast int inputs to float32.

    Returns:
        Output TensorRT tensor.
    """
    logger.debug(f"[TensorRT] Converting {op_name} node: {node.name}")

    input_node = node.args[0]
    if input_node not in input_map:
        raise ValueError(f"Input node '{input_node.name}' not found in input_map")

    input_trt = input_map[input_node]

    if cast_to_float:
        input_trt = _cast_to_float_if_needed(network, input_trt, node.name)

    layer = network.add_unary(input_trt, operation_type)
    if layer is None:
        raise RuntimeError(f"Failed to create {op_name} layer for {node.name}")
    layer.name = f"{op_name}_{node.name}"

    return layer.get_output(0)


@converter("aten.sqrt.default", validator_fn=validate_unary)
def convert_sqrt(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch sqrt to TensorRT.

    PyTorch signature: aten.sqrt(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.SQRT, "sqrt"
    )


@converter("aten.rsqrt.default", validator_fn=validate_unary)
def convert_rsqrt(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch rsqrt to TensorRT.

    PyTorch signature: aten.rsqrt(Tensor self) -> Tensor
    Computes 1/sqrt(x)

    Implemented as sqrt(x) followed by reciprocal, following TensorRT pattern.
    """
    logger.debug(f"[TensorRT] Converting rsqrt node: {node.name}")

    input_node = node.args[0]
    if input_node not in input_map:
        raise ValueError(f"Input node '{input_node.name}' not found in input_map")

    input_trt = input_map[input_node]
    input_trt = _cast_to_float_if_needed(network, input_trt, node.name)

    # First compute sqrt(x)
    sqrt_layer = network.add_unary(input_trt, trt.UnaryOperation.SQRT)
    if sqrt_layer is None:
        raise RuntimeError(f"Failed to create sqrt layer for rsqrt {node.name}")
    sqrt_layer.name = f"rsqrt_sqrt_{node.name}"

    # Then compute 1/sqrt(x) using reciprocal
    recip_layer = network.add_unary(sqrt_layer.get_output(0), trt.UnaryOperation.RECIP)
    if recip_layer is None:
        raise RuntimeError(f"Failed to create reciprocal layer for rsqrt {node.name}")
    recip_layer.name = f"rsqrt_{node.name}"

    return recip_layer.get_output(0)


@converter("aten.exp.default", validator_fn=validate_unary)
def convert_exp(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch exp to TensorRT.

    PyTorch signature: aten.exp(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.EXP, "exp"
    )


@converter("aten.log.default", validator_fn=validate_unary)
def convert_log(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch log (natural logarithm) to TensorRT.

    PyTorch signature: aten.log(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.LOG, "log"
    )


@converter("aten.neg.default", validator_fn=validate_unary)
def convert_neg(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch neg (negation) to TensorRT.

    PyTorch signature: aten.neg(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.NEG, "neg"
    )


@converter("aten.abs.default", validator_fn=validate_unary)
def convert_abs(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch abs to TensorRT.

    PyTorch signature: aten.abs(Tensor self) -> Tensor
    Note: ABS supports all data types, no float casting needed.
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.ABS, "abs", cast_to_float=False
    )


@converter("aten.sin.default", validator_fn=validate_unary)
def convert_sin(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch sin to TensorRT.

    PyTorch signature: aten.sin(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.SIN, "sin"
    )


@converter("aten.cos.default", validator_fn=validate_unary)
def convert_cos(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch cos to TensorRT.

    PyTorch signature: aten.cos(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.COS, "cos"
    )


@converter("aten.floor.default", validator_fn=validate_unary)
def convert_floor(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch floor to TensorRT.

    PyTorch signature: aten.floor(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.FLOOR, "floor"
    )


@converter("aten.ceil.default", validator_fn=validate_unary)
def convert_ceil(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch ceil to TensorRT.

    PyTorch signature: aten.ceil(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.CEIL, "ceil"
    )


@converter("aten.erf.default", validator_fn=validate_unary)
def convert_erf(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch erf (error function) to TensorRT.

    PyTorch signature: aten.erf(Tensor self) -> Tensor
    Used in GELU activation and other transformer operations.
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.ERF, "erf"
    )


@converter("aten.reciprocal.default", validator_fn=validate_unary)
def convert_reciprocal(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch reciprocal (1/x) to TensorRT.

    PyTorch signature: aten.reciprocal(Tensor self) -> Tensor
    """
    return _convert_unary_base(
        node, network, input_map, trt.UnaryOperation.RECIP, "reciprocal"
    )


__all__ = [
    "convert_sqrt",
    "convert_rsqrt",
    "convert_exp",
    "convert_log",
    "convert_neg",
    "convert_abs",
    "convert_sin",
    "convert_cos",
    "convert_floor",
    "convert_ceil",
    "convert_erf",
    "convert_reciprocal",
    "validate_unary",
]
