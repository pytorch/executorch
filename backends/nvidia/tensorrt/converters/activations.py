# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Activation Operations.

This module provides converters for PyTorch activation operations to TensorRT
activation layers.

Supported operations:
- aten.sigmoid.default: Sigmoid activation
- aten.tanh.default: Tanh activation
- aten.gelu.default: GELU activation
- aten.silu.default: SiLU/Swish activation (x * sigmoid(x))
- aten.softmax.int: Softmax (uses dedicated layer, not add_activation)
- aten.hardswish.default: Hard-swish (x * relu6(x + 3) / 6) - critical for MobileNetV3
- aten.hardsigmoid.default: Hard-sigmoid (min(max((x + 3) / 6, 0), 1)) - critical for SE blocks

Notes:
- Simple activations (sigmoid, tanh, gelu, silu) use network.add_activation()
- Softmax uses network.add_softmax() with axis configuration
- hardswish/hardsigmoid are decomposed into elementwise operations
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_unary_activation(node: torch.fx.Node) -> bool:
    """
    Validate that an activation node can be converted to TensorRT.

    Args:
        node: FX node representing the activation operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_activation: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Minimum args: input
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_activation: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_softmax(node: torch.fx.Node) -> bool:
    """
    Validate that a softmax node can be converted to TensorRT.

    Args:
        node: FX node representing the softmax operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_softmax: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: input, dim
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_softmax: node {node.name} has insufficient args"
        )
        return False

    # dim should be an int
    dim = args[1]
    if not isinstance(dim, int):
        logger.debug(
            f"[TensorRT] validate_softmax: dim must be int, got {type(dim)}"
        )
        return False

    return True


def _create_scalar_constant(
    network: Any,  # trt.INetworkDefinition
    scalar_value: float,
    name_suffix: str,
    target_ndims: int = 0,
) -> Any:  # trt.ITensor
    """
    Create a TensorRT constant tensor from a scalar value.

    Args:
        network: TensorRT network definition.
        scalar_value: The scalar value to create.
        name_suffix: Suffix for the layer name.
        target_ndims: Number of dimensions for the output shape.
                      The constant will be created with shape [1, 1, ..., 1] (target_ndims ones).
                      If 0, creates a scalar constant.

    Returns:
        TensorRT constant tensor for broadcasting.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy required") from e

    # Create shape with appropriate number of dimensions for broadcasting
    if target_ndims > 0:
        shape = [1] * target_ndims
        scalar_array = np.full(shape, scalar_value, dtype=np.float32)
    else:
        scalar_array = np.array([scalar_value], dtype=np.float32)
        shape = [1]

    weights = trt.Weights(scalar_array)
    layer = network.add_constant(trt.Dims(shape), weights)

    if layer is None:
        raise RuntimeError(f"Failed to create constant layer: {name_suffix}")

    layer.name = f"scalar_const_{name_suffix}"
    return layer.get_output(0)


@converter("aten.sigmoid.default", validator_fn=validate_unary_activation)
def convert_sigmoid(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch sigmoid to TensorRT activation layer.

    Args:
        node: FX node representing the sigmoid operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_sigmoid") from e

    logger.debug(f"[TensorRT] Converting sigmoid node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to sigmoid must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    layer = network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    if layer is None:
        raise RuntimeError(f"Failed to create sigmoid layer for node {node.name}")

    layer.name = f"sigmoid_{node.name}"
    logger.debug(f"[TensorRT] Created sigmoid layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.tanh.default", validator_fn=validate_unary_activation)
def convert_tanh(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch tanh to TensorRT activation layer.

    Args:
        node: FX node representing the tanh operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_tanh") from e

    logger.debug(f"[TensorRT] Converting tanh node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to tanh must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    layer = network.add_activation(input_trt, trt.ActivationType.TANH)
    if layer is None:
        raise RuntimeError(f"Failed to create tanh layer for node {node.name}")

    layer.name = f"tanh_{node.name}"
    logger.debug(f"[TensorRT] Created tanh layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.gelu.default", validator_fn=validate_unary_activation)
def convert_gelu(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch GELU to TensorRT activation layer.

    Args:
        node: FX node representing the gelu operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_gelu") from e

    logger.debug(f"[TensorRT] Converting gelu node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to gelu must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # TensorRT 8.6+ has native GELU support via GELU_ERF or GELU_TANH
    # For older versions, we need to fall back to a manual implementation
    try:
        layer = network.add_activation(input_trt, trt.ActivationType.GELU_ERF)
    except AttributeError:
        # Fallback: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # This is the tanh approximation of GELU
        import math

        # Constants
        sqrt_2_pi = math.sqrt(2.0 / math.pi)  # ~0.7979
        coeff = 0.044715

        # Create constant tensors matching input shape
        input_ndims = len(input_trt.shape)

        def create_scalar(val: float, name: str) -> Any:
            const_shape = tuple([1] * input_ndims)
            const_data = torch.tensor([val], dtype=torch.float32).numpy()
            const_weights = trt.Weights(const_data)
            layer = network.add_constant(const_shape, const_weights)
            if layer is None:
                raise RuntimeError(f"Failed to create constant {name}")
            layer.name = f"gelu_const_{name}_{node.name}"
            return layer.get_output(0)

        const_half = create_scalar(0.5, "half")
        const_one = create_scalar(1.0, "one")
        const_sqrt2pi = create_scalar(sqrt_2_pi, "sqrt2pi")
        const_coeff = create_scalar(coeff, "coeff")

        # x^3
        x_sq = network.add_elementwise(input_trt, input_trt, trt.ElementWiseOperation.PROD)
        x_sq.name = f"gelu_x_sq_{node.name}"
        x_cubed = network.add_elementwise(x_sq.get_output(0), input_trt, trt.ElementWiseOperation.PROD)
        x_cubed.name = f"gelu_x_cubed_{node.name}"

        # 0.044715 * x^3
        coeff_x_cubed = network.add_elementwise(const_coeff, x_cubed.get_output(0), trt.ElementWiseOperation.PROD)
        coeff_x_cubed.name = f"gelu_coeff_x_cubed_{node.name}"

        # x + 0.044715 * x^3
        x_plus_term = network.add_elementwise(input_trt, coeff_x_cubed.get_output(0), trt.ElementWiseOperation.SUM)
        x_plus_term.name = f"gelu_x_plus_term_{node.name}"

        # sqrt(2/π) * (x + 0.044715 * x^3)
        scaled = network.add_elementwise(const_sqrt2pi, x_plus_term.get_output(0), trt.ElementWiseOperation.PROD)
        scaled.name = f"gelu_scaled_{node.name}"

        # tanh(...)
        tanh_layer = network.add_activation(scaled.get_output(0), trt.ActivationType.TANH)
        tanh_layer.name = f"gelu_tanh_{node.name}"

        # 1 + tanh(...)
        one_plus_tanh = network.add_elementwise(const_one, tanh_layer.get_output(0), trt.ElementWiseOperation.SUM)
        one_plus_tanh.name = f"gelu_one_plus_tanh_{node.name}"

        # x * (1 + tanh(...))
        x_times_term = network.add_elementwise(input_trt, one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD)
        x_times_term.name = f"gelu_x_times_term_{node.name}"

        # 0.5 * x * (1 + tanh(...))
        layer = network.add_elementwise(const_half, x_times_term.get_output(0), trt.ElementWiseOperation.PROD)
        layer.name = f"gelu_final_{node.name}"

        logger.debug(f"[TensorRT] Using GELU tanh approximation (GELU_ERF not available)")
        return layer.get_output(0)

    if layer is None:
        raise RuntimeError(f"Failed to create gelu layer for node {node.name}")

    layer.name = f"gelu_{node.name}"
    logger.debug(f"[TensorRT] Created gelu layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.silu.default", validator_fn=validate_unary_activation)
def convert_silu(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch SiLU (Swish) to TensorRT activation layer.

    SiLU is defined as: x * sigmoid(x)

    Args:
        node: FX node representing the silu operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_silu") from e

    logger.debug(f"[TensorRT] Converting silu node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to silu must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # TensorRT has native Swish (SiLU) support
    layer = network.add_activation(input_trt, trt.ActivationType.SWISH)
    if layer is None:
        raise RuntimeError(f"Failed to create silu layer for node {node.name}")

    layer.name = f"silu_{node.name}"
    logger.debug(f"[TensorRT] Created silu layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.softmax.int", "aten._softmax.default", validator_fn=validate_softmax)
def convert_softmax(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch softmax to TensorRT softmax layer.

    Note: Softmax uses network.add_softmax() instead of add_activation().

    Args:
        node: FX node representing the softmax operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_softmax") from e

    logger.debug(f"[TensorRT] Converting softmax node: {node.name}")

    input_node = node.args[0]
    dim = node.args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to softmax must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Handle negative dim using TRT tensor shape.
    ndim = len(input_trt.shape)
    if dim < 0:
        dim = ndim + dim

    # Create softmax layer (NOT add_activation)
    layer = network.add_softmax(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create softmax layer for node {node.name}")

    # Set the axis for softmax
    # TensorRT uses axes as a bitmask
    layer.axes = 1 << dim

    layer.name = f"softmax_{node.name}"
    logger.debug(f"[TensorRT] Created softmax layer: {layer.name}, axis={dim}")

    return layer.get_output(0)


@converter("aten.log_softmax.int", "aten._log_softmax.default", validator_fn=validate_softmax)
def convert_log_softmax(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch _log_softmax to TensorRT softmax + log layers.

    We decompose log_softmax into softmax followed by log operation,
    reusing the existing converters.

    Args:
        node: FX node representing the _log_softmax operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_log_softmax") from e

    logger.debug(f"[TensorRT] Converting _log_softmax node: {node.name}")

    input_node = node.args[0]
    dim = node.args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to _log_softmax must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Handle negative dimension
    input_shape = input_trt.shape
    ndim = len(input_shape)
    if dim < 0:
        dim = ndim + dim

    # Step 1: Apply softmax
    softmax_layer = network.add_softmax(input_trt)
    if softmax_layer is None:
        raise RuntimeError(
            f"Failed to create softmax layer for _log_softmax node {node.name}"
        )
    softmax_layer.axes = 1 << dim
    softmax_layer.name = f"log_softmax_softmax_{node.name}"
    softmax_output = softmax_layer.get_output(0)

    logger.debug(
        f"[TensorRT] Created softmax layer for log_softmax: "
        f"{softmax_layer.name}, axis={dim}"
    )

    # Step 2: Apply log
    log_layer = network.add_unary(softmax_output, trt.UnaryOperation.LOG)
    if log_layer is None:
        raise RuntimeError(
            f"Failed to create log layer for _log_softmax node {node.name}"
        )
    log_layer.name = f"log_softmax_log_{node.name}"

    logger.debug(f"[TensorRT] Created log_softmax composite: {log_layer.name}")

    return log_layer.get_output(0)


@converter("aten.hardswish.default", "aten.hardswish_.default", validator_fn=validate_unary_activation)
def convert_hardswish(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch hardswish to TensorRT using native HardSigmoid activation.

    Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)

    TRT's HARD_SIGMOID computes: max(0, min(1, alpha * x + beta))
    With alpha=1/6, beta=0.5 this gives: clip(x/6 + 0.5, 0, 1) = clip((x+3)/6, 0, 1)

    This 2-layer approach (hardsigmoid + mul) replaces a 5-layer decomposition
    (add + relu + min + div + mul), enabling better TRT kernel fusion.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_hardswish") from e

    logger.debug(f"[TensorRT] Converting hardswish node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to hardswish must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Step 1: HardSigmoid(x) = clip(x/6 + 0.5, 0, 1)
    hs_layer = network.add_activation(input_trt, trt.ActivationType.HARD_SIGMOID)
    hs_layer.alpha = 1.0 / 6.0
    hs_layer.beta = 1.0 / 2.0
    hs_layer.name = f"hardswish_hardsigmoid_{node.name}"

    # Step 2: x * HardSigmoid(x)
    mul_layer = network.add_elementwise(
        input_trt, hs_layer.get_output(0), trt.ElementWiseOperation.PROD
    )
    mul_layer.name = f"hardswish_{node.name}"

    logger.debug(f"[TensorRT] Created hardswish via hardsigmoid: {mul_layer.name}")

    return mul_layer.get_output(0)


@converter("aten.hardsigmoid.default", validator_fn=validate_unary_activation)
def convert_hardsigmoid(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch hardsigmoid to TensorRT composite layers.

    Hardsigmoid is defined as: min(max((x + 3) / 6, 0), 1)
    Which is: clip((x + 3) / 6, 0, 1)

    This is critical for MobileNetV3 Squeeze-and-Excitation blocks.

    Args:
        node: FX node representing the hardsigmoid operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_hardsigmoid") from e

    logger.debug(f"[TensorRT] Converting hardsigmoid node: {node.name}")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to hardsigmoid must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get input dimensions for proper broadcasting
    input_ndims = len(input_trt.shape)

    # Step 1: Add 3
    const_3 = _create_scalar_constant(
        network, 3.0, f"{node.name}_const3", target_ndims=input_ndims
    )
    add_3_layer = network.add_elementwise(
        input_trt, const_3, trt.ElementWiseOperation.SUM
    )
    if add_3_layer is None:
        raise RuntimeError(f"Failed to create add_3 layer for hardsigmoid {node.name}")
    add_3_layer.name = f"hardsigmoid_add3_{node.name}"
    add_3_output = add_3_layer.get_output(0)

    # Step 2: Divide by 6
    const_6 = _create_scalar_constant(
        network, 6.0, f"{node.name}_const6", target_ndims=input_ndims
    )
    div_layer = network.add_elementwise(
        add_3_output, const_6, trt.ElementWiseOperation.DIV
    )
    if div_layer is None:
        raise RuntimeError(f"Failed to create div layer for hardsigmoid {node.name}")
    div_layer.name = f"hardsigmoid_div6_{node.name}"
    div_output = div_layer.get_output(0)

    # Step 3: ReLU (max((x+3)/6, 0))
    relu_layer = network.add_activation(div_output, trt.ActivationType.RELU)
    if relu_layer is None:
        raise RuntimeError(f"Failed to create relu layer for hardsigmoid {node.name}")
    relu_layer.name = f"hardsigmoid_relu_{node.name}"
    relu_output = relu_layer.get_output(0)

    # Step 4: Min with 1 (clip to [0, 1])
    const_1 = _create_scalar_constant(
        network, 1.0, f"{node.name}_const1", target_ndims=input_ndims
    )
    min_layer = network.add_elementwise(
        relu_output, const_1, trt.ElementWiseOperation.MIN
    )
    if min_layer is None:
        raise RuntimeError(f"Failed to create min layer for hardsigmoid {node.name}")
    min_layer.name = f"hardsigmoid_{node.name}"

    logger.debug(f"[TensorRT] Created hardsigmoid composite: {min_layer.name}")

    return min_layer.get_output(0)


def validate_clamp(node: torch.fx.Node) -> bool:
    """
    Validate that a clamp node can be converted to TensorRT.

    Args:
        node: FX node representing the clamp operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # clamp takes at least 1 arg (input), and optionally min and max
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_clamp: node {node.name} has insufficient args"
        )
        return False

    return True


@converter(
    "aten.clamp.default",
    "aten.clamp.Tensor",
    "aten.clamp_min.default",
    "aten.clamp_max.default",
    validator_fn=validate_clamp,
)
def convert_clamp(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch clamp to TensorRT elementwise operations.

    Clamp is defined as: output = min(max(input, min_val), max_val)

    Args:
        node: FX node representing the clamp operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting parameters.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_clamp") from e

    logger.debug(f"[TensorRT] Converting clamp node: {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to clamp must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get min and max values
    op_name = str(node.target)
    min_val = None
    max_val = None

    if "clamp_min" in op_name:
        # clamp_min only has min
        min_val = args[1] if len(args) > 1 else None
    elif "clamp_max" in op_name:
        # clamp_max only has max
        max_val = args[1] if len(args) > 1 else None
    else:
        # Regular clamp has both
        min_val = args[1] if len(args) > 1 else None
        max_val = args[2] if len(args) > 2 else None

    # Also check kwargs
    min_val = node.kwargs.get("min", min_val)
    max_val = node.kwargs.get("max", max_val)

    output = input_trt

    # Get input dimensions for proper broadcasting
    input_ndims = len(input_trt.shape)

    # Apply min (max with min_val)
    if min_val is not None:
        min_val_float = float(min_val) if isinstance(min_val, (int, float)) else 0.0
        min_const = _create_scalar_constant(
            network, min_val_float, f"{node.name}_min", target_ndims=input_ndims
        )
        max_layer = network.add_elementwise(
            output, min_const, trt.ElementWiseOperation.MAX
        )
        if max_layer is None:
            raise RuntimeError(f"Failed to create max layer for clamp {node.name}")
        max_layer.name = f"clamp_min_{node.name}"
        output = max_layer.get_output(0)

    # Apply max (min with max_val)
    if max_val is not None:
        max_val_float = float(max_val) if isinstance(max_val, (int, float)) else 1.0
        max_const = _create_scalar_constant(
            network, max_val_float, f"{node.name}_max", target_ndims=input_ndims
        )
        min_layer = network.add_elementwise(
            output, max_const, trt.ElementWiseOperation.MIN
        )
        if min_layer is None:
            raise RuntimeError(f"Failed to create min layer for clamp {node.name}")
        min_layer.name = f"clamp_max_{node.name}"
        output = min_layer.get_output(0)

    logger.debug(f"[TensorRT] Created clamp layers for node: {node.name}")

    return output


def validate_hardtanh(node: torch.fx.Node) -> bool:
    """
    Validate that a hardtanh node can be converted to TensorRT.

    Args:
        node: FX node representing the hardtanh operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_hardtanh: node {node.name} has insufficient args"
        )
        return False

    return True


@converter(
    "aten.hardtanh.default",
    "aten.hardtanh_.default",
    validator_fn=validate_hardtanh,
)
def convert_hardtanh(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch hardtanh to TensorRT elementwise operations.

    Hardtanh is defined as: output = clamp(input, min_val, max_val)
    This is used by ReLU6 (hardtanh with min=0, max=6) in MobileNetV2.

    Args:
        node: FX node representing the hardtanh operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting parameters.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_hardtanh") from e

    logger.debug(f"[TensorRT] Converting hardtanh node: {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to hardtanh must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get min and max values (default: min=-1, max=1)
    min_val = args[1] if len(args) > 1 else node.kwargs.get("min_val", -1.0)
    max_val = args[2] if len(args) > 2 else node.kwargs.get("max_val", 1.0)

    min_val_float = float(min_val)
    max_val_float = float(max_val)

    # Get input dimensions for proper broadcasting
    input_ndims = len(input_trt.shape)

    output = input_trt

    # Apply min (max with min_val) - clamp from below
    min_const = _create_scalar_constant(
        network, min_val_float, f"{node.name}_min", target_ndims=input_ndims
    )
    max_layer = network.add_elementwise(
        output, min_const, trt.ElementWiseOperation.MAX
    )
    if max_layer is None:
        raise RuntimeError(f"Failed to create max layer for hardtanh {node.name}")
    max_layer.name = f"hardtanh_min_{node.name}"
    output = max_layer.get_output(0)

    # Apply max (min with max_val) - clamp from above
    max_const = _create_scalar_constant(
        network, max_val_float, f"{node.name}_max", target_ndims=input_ndims
    )
    min_layer = network.add_elementwise(
        output, max_const, trt.ElementWiseOperation.MIN
    )
    if min_layer is None:
        raise RuntimeError(f"Failed to create min layer for hardtanh {node.name}")
    min_layer.name = f"hardtanh_max_{node.name}"
    output = min_layer.get_output(0)

    logger.debug(
        f"[TensorRT] Created hardtanh layers for node: {node.name} "
        f"(min={min_val_float}, max={max_val_float})"
    )

    return output


__all__ = [
    "convert_sigmoid",
    "convert_tanh",
    "convert_gelu",
    "convert_silu",
    "convert_softmax",
    "convert_log_softmax",
    "convert_hardswish",
    "convert_hardsigmoid",
    "convert_clamp",
    "convert_hardtanh",
    "convert_dropout",
    "validate_unary_activation",
    "validate_softmax",
    "validate_clamp",
    "validate_hardtanh",
]


def validate_dropout(node: torch.fx.Node) -> bool:
    """Validate that a dropout node can be converted to TensorRT."""
    if node.op != "call_function":
        return False

    # Dropout must have at least 1 arg (input)
    if len(node.args) < 1:
        return False

    return True


@converter(
    "aten.dropout.default",
    "aten.dropout_.default",
    validator_fn=validate_dropout,
)
def convert_dropout(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch dropout to TensorRT (no-op in inference mode).

    Dropout is a no-op during inference - just pass through the input tensor.

    Args:
        node: FX node representing the dropout operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor (same as input).
    """
    logger.debug(f"[TensorRT] Converting dropout node: {node.name} (no-op)")

    input_node = node.args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to dropout must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    # Dropout is a no-op in inference mode - return input directly
    return input_map[input_node]
