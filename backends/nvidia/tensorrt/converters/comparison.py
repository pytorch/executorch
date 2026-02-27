# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Comparison and Logical Operations.

This module provides converters for PyTorch comparison and logical operations
to TensorRT layers.

Supported operations:
- aten.eq.Scalar, aten.ne.Scalar, etc.: Comparison with scalar
- aten.logical_not.default: Logical NOT
- aten.where.self: Conditional selection
- aten.any.dim, aten.all.dim: Boolean reduction
- aten.full_like.default: Create tensor filled with value
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
)

logger: logging.Logger = logging.getLogger(__name__)


@converter("aten.eq.Scalar")
def convert_eq_scalar(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch eq (equal) with scalar to TensorRT.

    eq.Scalar(Tensor self, Scalar other) -> Tensor

    Args:
        node: FX node representing the eq operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor (boolean).
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to eq must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]
    
    # Get input shape from node metadata for proper broadcasting
    if isinstance(input_node, torch.fx.Node) and "val" in input_node.meta and hasattr(input_node.meta["val"], "shape"):
        input_shape = list(input_node.meta["val"].shape)
    else:
        # Fall back to TRT tensor shape (may be invalid during error conditions)
        try:
            input_shape = list(input_trt.shape)
        except (TypeError, ValueError):
            input_shape = [1]  # Fallback

    # Create constant for scalar with shape that can broadcast to input shape
    # Use shape [1, 1, ...] with same ndim as input for proper broadcasting
    ndim = len(input_shape)
    const_shape = [1] * ndim if ndim > 0 else [1]
    other_data = np.full(const_shape, other, dtype=np.float32)
    other_weights = trt.Weights(other_data)
    other_const = network.add_constant(const_shape, other_weights)
    other_const.name = f"eq_const_{node.name}"

    # TensorRT EQUAL comparison
    layer = network.add_elementwise(
        input_trt, other_const.get_output(0), trt.ElementWiseOperation.EQUAL
    )

    if layer is None:
        raise RuntimeError(f"Failed to create eq layer for {node.name}")

    layer.name = f"eq_scalar_{node.name}"
    return layer.get_output(0)


@converter("aten.ne.Scalar")
def convert_ne_scalar(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch ne (not equal) with scalar to TensorRT."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to ne must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    # Create constant for scalar
    other_weights = trt.Weights(np.array([other], dtype=np.float32))
    other_const = network.add_constant([1], other_weights)
    other_const.name = f"ne_const_{node.name}"

    # First compute EQUAL, then NOT
    eq_layer = network.add_elementwise(
        input_trt, other_const.get_output(0), trt.ElementWiseOperation.EQUAL
    )
    eq_layer.name = f"ne_eq_{node.name}"

    # Logical NOT for boolean tensors
    not_layer = network.add_unary(eq_layer.get_output(0), trt.UnaryOperation.NOT)

    if not_layer is None:
        raise RuntimeError(f"Failed to create ne layer for {node.name}")

    not_layer.name = f"ne_scalar_{node.name}"
    return not_layer.get_output(0)


@converter("aten.lt.Scalar")
def convert_lt_scalar(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch lt (less than) with scalar to TensorRT."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to lt must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    other_weights = trt.Weights(np.array([other], dtype=np.float32))
    other_const = network.add_constant([1], other_weights)
    other_const.name = f"lt_const_{node.name}"

    layer = network.add_elementwise(
        input_trt, other_const.get_output(0), trt.ElementWiseOperation.LESS
    )

    if layer is None:
        raise RuntimeError(f"Failed to create lt layer for {node.name}")

    layer.name = f"lt_scalar_{node.name}"
    return layer.get_output(0)


@converter("aten.gt.Scalar")
def convert_gt_scalar(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch gt (greater than) with scalar to TensorRT."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to gt must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    other_weights = trt.Weights(np.array([other], dtype=np.float32))
    other_const = network.add_constant([1], other_weights)
    other_const.name = f"gt_const_{node.name}"

    layer = network.add_elementwise(
        input_trt, other_const.get_output(0), trt.ElementWiseOperation.GREATER
    )

    if layer is None:
        raise RuntimeError(f"Failed to create gt layer for {node.name}")

    layer.name = f"gt_scalar_{node.name}"
    return layer.get_output(0)


@converter("aten.ge.Scalar")
def convert_ge_scalar(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch ge (greater than or equal) with scalar to TensorRT.

    ge(x, y) = gt(x, y) OR eq(x, y)

    TensorRT doesn't have a native GE operation, so we implement it as
    the logical OR of GREATER and EQUAL operations.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to ge must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    # Get input shape for proper broadcasting
    if (
        isinstance(input_node, torch.fx.Node)
        and "val" in input_node.meta
        and hasattr(input_node.meta["val"], "shape")
    ):
        input_shape = list(input_node.meta["val"].shape)
    else:
        try:
            input_shape = list(input_trt.shape)
        except (TypeError, ValueError):
            input_shape = [1]

    # Create constant with proper shape for broadcasting
    ndim = len(input_shape)
    const_shape = [1] * ndim if ndim > 0 else [1]
    other_data = np.full(const_shape, other, dtype=np.float32)
    other_weights = trt.Weights(other_data)
    other_const = network.add_constant(const_shape, other_weights)
    other_const.name = f"ge_const_{node.name}"
    other_tensor = other_const.get_output(0)

    # Compute GREATER
    gt_layer = network.add_elementwise(
        input_trt, other_tensor, trt.ElementWiseOperation.GREATER
    )
    if gt_layer is None:
        raise RuntimeError(f"Failed to create gt layer for ge_{node.name}")
    gt_layer.name = f"ge_gt_{node.name}"

    # Compute EQUAL
    eq_layer = network.add_elementwise(
        input_trt, other_tensor, trt.ElementWiseOperation.EQUAL
    )
    if eq_layer is None:
        raise RuntimeError(f"Failed to create eq layer for ge_{node.name}")
    eq_layer.name = f"ge_eq_{node.name}"

    # Compute OR (gt OR eq)
    or_layer = network.add_elementwise(
        gt_layer.get_output(0), eq_layer.get_output(0), trt.ElementWiseOperation.OR
    )
    if or_layer is None:
        raise RuntimeError(f"Failed to create or layer for ge_{node.name}")
    or_layer.name = f"ge_scalar_{node.name}"

    return or_layer.get_output(0)


@converter("aten.le.Scalar")
def convert_le_scalar(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch le (less than or equal) with scalar to TensorRT.

    le(x, y) = lt(x, y) OR eq(x, y)

    TensorRT doesn't have a native LE operation, so we implement it as
    the logical OR of LESS and EQUAL operations.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]
    other = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to le must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    # Get input shape for proper broadcasting
    if (
        isinstance(input_node, torch.fx.Node)
        and "val" in input_node.meta
        and hasattr(input_node.meta["val"], "shape")
    ):
        input_shape = list(input_node.meta["val"].shape)
    else:
        try:
            input_shape = list(input_trt.shape)
        except (TypeError, ValueError):
            input_shape = [1]

    # Create constant with proper shape for broadcasting
    ndim = len(input_shape)
    const_shape = [1] * ndim if ndim > 0 else [1]
    other_data = np.full(const_shape, other, dtype=np.float32)
    other_weights = trt.Weights(other_data)
    other_const = network.add_constant(const_shape, other_weights)
    other_const.name = f"le_const_{node.name}"
    other_tensor = other_const.get_output(0)

    # Compute LESS
    lt_layer = network.add_elementwise(
        input_trt, other_tensor, trt.ElementWiseOperation.LESS
    )
    if lt_layer is None:
        raise RuntimeError(f"Failed to create lt layer for le_{node.name}")
    lt_layer.name = f"le_lt_{node.name}"

    # Compute EQUAL
    eq_layer = network.add_elementwise(
        input_trt, other_tensor, trt.ElementWiseOperation.EQUAL
    )
    if eq_layer is None:
        raise RuntimeError(f"Failed to create eq layer for le_{node.name}")
    eq_layer.name = f"le_eq_{node.name}"

    # Compute OR (lt OR eq)
    or_layer = network.add_elementwise(
        lt_layer.get_output(0), eq_layer.get_output(0), trt.ElementWiseOperation.OR
    )
    if or_layer is None:
        raise RuntimeError(f"Failed to create or layer for le_{node.name}")
    or_layer.name = f"le_scalar_{node.name}"

    return or_layer.get_output(0)


@converter("aten.ge.Tensor")
def convert_ge_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch ge (greater than or equal) with tensor to TensorRT.

    ge(x, y) = gt(x, y) OR eq(x, y)
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to ge must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to ge must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    # Compute GREATER
    gt_layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.GREATER
    )
    if gt_layer is None:
        raise RuntimeError(f"Failed to create gt layer for ge_{node.name}")
    gt_layer.name = f"ge_gt_{node.name}"

    # Compute EQUAL
    eq_layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.EQUAL
    )
    if eq_layer is None:
        raise RuntimeError(f"Failed to create eq layer for ge_{node.name}")
    eq_layer.name = f"ge_eq_{node.name}"

    # Compute OR (gt OR eq)
    or_layer = network.add_elementwise(
        gt_layer.get_output(0), eq_layer.get_output(0), trt.ElementWiseOperation.OR
    )
    if or_layer is None:
        raise RuntimeError(f"Failed to create or layer for ge_{node.name}")
    or_layer.name = f"ge_tensor_{node.name}"

    return or_layer.get_output(0)


@converter("aten.le.Tensor")
def convert_le_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch le (less than or equal) with tensor to TensorRT.

    le(x, y) = lt(x, y) OR eq(x, y)
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to le must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to le must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    # Compute LESS
    lt_layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.LESS
    )
    if lt_layer is None:
        raise RuntimeError(f"Failed to create lt layer for le_{node.name}")
    lt_layer.name = f"le_lt_{node.name}"

    # Compute EQUAL
    eq_layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.EQUAL
    )
    if eq_layer is None:
        raise RuntimeError(f"Failed to create eq layer for le_{node.name}")
    eq_layer.name = f"le_eq_{node.name}"

    # Compute OR (lt OR eq)
    or_layer = network.add_elementwise(
        lt_layer.get_output(0), eq_layer.get_output(0), trt.ElementWiseOperation.OR
    )
    if or_layer is None:
        raise RuntimeError(f"Failed to create or layer for le_{node.name}")
    or_layer.name = f"le_tensor_{node.name}"

    return or_layer.get_output(0)


@converter("aten.eq.Tensor")
def convert_eq_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch eq (equal) with tensor to TensorRT."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to eq must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to eq must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.EQUAL
    )
    if layer is None:
        raise RuntimeError(f"Failed to create eq layer for {node.name}")
    layer.name = f"eq_tensor_{node.name}"

    return layer.get_output(0)


@converter("aten.ne.Tensor")
def convert_ne_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch ne (not equal) with tensor to TensorRT."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to ne must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to ne must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    # Compute EQUAL
    eq_layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.EQUAL
    )
    if eq_layer is None:
        raise RuntimeError(f"Failed to create eq layer for ne_{node.name}")
    eq_layer.name = f"ne_eq_{node.name}"

    # Logical NOT
    not_layer = network.add_unary(eq_layer.get_output(0), trt.UnaryOperation.NOT)
    if not_layer is None:
        raise RuntimeError(f"Failed to create not layer for ne_{node.name}")
    not_layer.name = f"ne_tensor_{node.name}"

    return not_layer.get_output(0)


@converter("aten.lt.Tensor")
def convert_lt_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch lt (less than) with tensor to TensorRT."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to lt must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to lt must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.LESS
    )
    if layer is None:
        raise RuntimeError(f"Failed to create lt layer for {node.name}")
    layer.name = f"lt_tensor_{node.name}"

    return layer.get_output(0)


@converter("aten.gt.Tensor")
def convert_gt_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert PyTorch gt (greater than) with tensor to TensorRT."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to gt must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to gt must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    layer = network.add_elementwise(
        input_trt, other_trt, trt.ElementWiseOperation.GREATER
    )
    if layer is None:
        raise RuntimeError(f"Failed to create gt layer for {node.name}")
    layer.name = f"gt_tensor_{node.name}"

    return layer.get_output(0)


@converter("aten.logical_not.default")
def convert_logical_not(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch logical_not to TensorRT.

    logical_not.default(Tensor self) -> Tensor
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to logical_not must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    layer = network.add_unary(input_trt, trt.UnaryOperation.NOT)

    if layer is None:
        raise RuntimeError(f"Failed to create logical_not layer for {node.name}")

    layer.name = f"logical_not_{node.name}"
    return layer.get_output(0)


@converter("aten.where.self", "aten.where.ScalarSelf")
def convert_where(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch where to TensorRT select layer.

    where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args

    condition_node = args[0]
    self_node = args[1]
    other_node = args[2]

    if not isinstance(condition_node, torch.fx.Node):
        raise ValueError(f"Condition must be a node, got {type(condition_node)}")

    condition_trt = input_map[condition_node]
    self_trt = input_map[self_node] if isinstance(self_node, torch.fx.Node) else None
    other_trt = input_map[other_node] if isinstance(other_node, torch.fx.Node) else None

    # Handle scalar inputs
    import numpy as np
    if self_trt is None:
        self_weights = trt.Weights(np.array([self_node], dtype=np.float32))
        self_const = network.add_constant([1], self_weights)
        self_const.name = f"where_self_const_{node.name}"
        self_trt = self_const.get_output(0)

    if other_trt is None:
        other_weights = trt.Weights(np.array([other_node], dtype=np.float32))
        other_const = network.add_constant([1], other_weights)
        other_const.name = f"where_other_const_{node.name}"
        other_trt = other_const.get_output(0)

    # TensorRT select: output = condition ? self : other
    layer = network.add_select(condition_trt, self_trt, other_trt)

    if layer is None:
        raise RuntimeError(f"Failed to create where/select layer for {node.name}")

    layer.name = f"where_{node.name}"
    return layer.get_output(0)


@converter("aten.any.dim")
def convert_any_dim(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch any.dim to TensorRT reduce layer.

    any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
    Returns True if any value along dim is True.

    TensorRT doesn't have native boolean reduce, so we:
    1. Cast bool to float (true=1, false=0)
    2. Sum along dimension
    3. Compare > 0
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to any must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]
    
    # Get ndim from node metadata for reliability (TRT shapes can be invalid during network building)
    ndim = len(get_node_shape(input_node) or input_trt.shape)

    # Handle negative dim
    if dim < 0:
        dim = ndim + dim

    # Cast to float if boolean
    identity = network.add_identity(input_trt)
    identity.set_output_type(0, trt.float32)
    identity.name = f"any_cast_{node.name}"
    float_input = identity.get_output(0)

    # Reduce sum along dimension
    axes = 1 << dim  # bitmask
    reduce_layer = network.add_reduce(
        float_input,
        trt.ReduceOperation.SUM,
        axes,
        keepdim,
    )
    reduce_layer.name = f"any_sum_{node.name}"

    # Get the output shape from node metadata for proper broadcasting
    if "val" in node.meta and hasattr(node.meta["val"], "shape"):
        output_shape = list(node.meta["val"].shape)
    else:
        # Compute output shape based on reduction
        input_shape = get_node_shape(input_node) or tuple(input_trt.shape)
        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
    
    # Create zero constant with shape matching the reduced output for proper broadcasting
    output_ndim = len(output_shape)
    const_shape = [1] * output_ndim if output_ndim > 0 else [1]
    zero_data = np.zeros(const_shape, dtype=np.float32)
    zero_weights = trt.Weights(zero_data)
    zero_const = network.add_constant(const_shape, zero_weights)
    zero_const.name = f"any_zero_const_{node.name}"

    gt_layer = network.add_elementwise(
        reduce_layer.get_output(0),
        zero_const.get_output(0),
        trt.ElementWiseOperation.GREATER,
    )

    if gt_layer is None:
        raise RuntimeError(f"Failed to create any layer for {node.name}")

    gt_layer.name = f"any_{node.name}"
    return gt_layer.get_output(0)


@converter("aten.full_like.default", "aten.zeros_like.default", "aten.ones_like.default")
def convert_full_like(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch full_like/zeros_like/ones_like to TensorRT.

    Creates a tensor filled with a constant value, matching the shape of input.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    args = node.args
    input_node = args[0]

    # Determine fill value based on op
    target_name = str(node.target)
    if "zeros_like" in target_name:
        fill_value = 0.0
    elif "ones_like" in target_name:
        fill_value = 1.0
    else:
        # full_like - fill value is second arg
        fill_value = args[1] if len(args) > 1 else 0.0

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input must be a node, got {type(input_node)}")

    # Get shape from node metadata instead of TensorRT tensor
    # This is more reliable as TensorRT shapes may have -1 for dynamic dims
    if "val" in node.meta and hasattr(node.meta["val"], "shape"):
        output_shape = list(node.meta["val"].shape)
    elif "val" in input_node.meta and hasattr(input_node.meta["val"], "shape"):
        output_shape = list(input_node.meta["val"].shape)
    else:
        # Fall back to TensorRT shape, but check for -1 values
        input_trt = input_map[input_node]
        output_shape = list(input_trt.shape)
        if any(d == -1 for d in output_shape):
            raise ValueError(
                f"Cannot create full_like with dynamic shape {output_shape}. "
                "Shape must be static."
            )

    logger.debug(f"[TensorRT] full_like: shape={output_shape}, fill_value={fill_value}")

    # Create constant filled with value
    fill_array = np.full(output_shape, fill_value, dtype=np.float32)
    fill_weights = trt.Weights(fill_array)
    layer = network.add_constant(trt.Dims(output_shape), fill_weights)

    if layer is None:
        raise RuntimeError(f"Failed to create full_like layer for {node.name}")

    layer.name = f"full_like_{node.name}"
    return layer.get_output(0)


@converter("aten.full.default")
def convert_full(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch full to TensorRT.

    full.default(int[] size, Scalar fill_value, ...) -> Tensor
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        _torch_dtype_to_numpy,
    )

    args = node.args

    size = args[0]
    fill_value = args[1] if len(args) > 1 else 0.0

    # Convert size to list
    if isinstance(size, (list, tuple)):
        shape = list(size)
    else:
        shape = [size]

    # Determine dtype from node metadata or kwargs, defaulting to float32.
    np_dtype = np.float32
    dtype_kwarg = node.kwargs.get("dtype", None)
    if dtype_kwarg is not None and isinstance(dtype_kwarg, torch.dtype):
        np_dtype = _torch_dtype_to_numpy(dtype_kwarg)
    elif "val" in node.meta:
        val = node.meta["val"]
        if hasattr(val, "dtype"):
            np_dtype = _torch_dtype_to_numpy(val.dtype)

    fill_array = np.full(shape, fill_value, dtype=np_dtype)
    fill_weights = trt.Weights(fill_array)
    layer = network.add_constant(trt.Dims(shape), fill_weights)

    if layer is None:
        raise RuntimeError(f"Failed to create full layer for {node.name}")

    layer.name = f"full_{node.name}"
    return layer.get_output(0)


@converter("aten.scalar_tensor.default")
def convert_scalar_tensor(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.scalar_tensor to a TRT constant."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    value = node.args[0]
    np_value = np.array([float(value)], dtype=np.float32)
    layer = network.add_constant([1], trt.Weights(np_value))
    layer.name = f"scalar_tensor_{node.name}"
    return layer.get_output(0)


@converter("aten.arange.start_step")
def convert_arange(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.arange.start_step to a TRT constant.

    Computes the range on the host and embeds it as a constant tensor.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    start = node.args[0]
    end = node.args[1] if len(node.args) > 1 else node.kwargs.get("end")
    step = node.args[2] if len(node.args) > 2 else node.kwargs.get("step", 1)

    values = np.arange(start, end, step, dtype=np.float32)
    layer = network.add_constant(list(values.shape), trt.Weights(values))
    layer.name = f"arange_{node.name}"
    return layer.get_output(0)


@converter("aten.constant_pad_nd.default")
def convert_constant_pad_nd(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.constant_pad_nd to TRT using ISliceLayer with FILL mode.

    pad format: [left_last_dim, right_last_dim, left_2nd_last, right_2nd_last, ...]
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    input_node = node.args[0]
    pad = list(node.args[1])
    value = float(node.args[2]) if len(node.args) > 2 else 0.0

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]
    input_shape = list(input_trt.shape)
    ndim = len(input_shape)
    num_padded_dims = len(pad) // 2

    start = [0] * ndim
    output_shape = list(input_shape)

    for i in range(num_padded_dims):
        dim = ndim - 1 - i
        left_pad = pad[2 * i]
        right_pad = pad[2 * i + 1]
        start[dim] = -left_pad
        output_shape[dim] = input_shape[dim] + left_pad + right_pad

    layer = network.add_slice(
        input_trt, start=start, shape=output_shape, stride=[1] * ndim
    )
    layer.mode = trt.SampleMode.FILL

    fill_const = network.add_constant(
        [1] * ndim, trt.Weights(np.array([value], dtype=np.float32))
    )
    fill_const.name = f"pad_fill_{node.name}"
    layer.set_input(4, fill_const.get_output(0))

    layer.name = f"constant_pad_nd_{node.name}"
    return layer.get_output(0)


@converter("aten.alias_copy.default")
def convert_alias_copy(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Pass-through for alias_copy (no-op in TRT)."""
    return input_map[node.args[0]]


@converter("aten.copy.default")
def convert_copy(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.copy to TRT pass-through.

    PyTorch signature: aten.copy(Tensor self, Tensor src) -> Tensor
    Copies src into self and returns self. For TRT, we return the src tensor
    (arg[1]) since TRT manages memory internally — effectively a pass-through.
    """
    src_node = node.args[1]
    if isinstance(src_node, torch.fx.Node) and src_node in input_map:
        return input_map[src_node]
    raise ValueError(f"Source node for copy not found in input_map: {node.name}")


__all__ = [
    "convert_eq_scalar",
    "convert_eq_tensor",
    "convert_ne_scalar",
    "convert_ne_tensor",
    "convert_lt_scalar",
    "convert_lt_tensor",
    "convert_gt_scalar",
    "convert_gt_tensor",
    "convert_ge_scalar",
    "convert_ge_tensor",
    "convert_le_scalar",
    "convert_le_tensor",
    "convert_logical_not",
    "convert_where",
    "convert_any_dim",
    "convert_full_like",
    "convert_full",
]
