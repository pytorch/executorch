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
    
    # Match the dtype of the input tensor for TensorRT compatibility
    # TensorRT elementwise operations require matching types
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32
    
    other_data = np.full(const_shape, other, dtype=np_dtype)
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

    # Match the dtype of the input tensor for TensorRT compatibility
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32
    
    # Create constant for scalar
    other_weights = trt.Weights(np.array([other], dtype=np_dtype))
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

    # Match the dtype of the input tensor for TensorRT compatibility
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32

    other_weights = trt.Weights(np.array([other], dtype=np_dtype))
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

    # Match the dtype of the input tensor for TensorRT compatibility
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32

    other_weights = trt.Weights(np.array([other], dtype=np_dtype))
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

    # Match the dtype of the input tensor for TensorRT compatibility
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32

    # Create constant with proper shape for broadcasting
    ndim = len(input_shape)
    const_shape = [1] * ndim if ndim > 0 else [1]
    other_data = np.full(const_shape, other, dtype=np_dtype)
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

    # Match the dtype of the input tensor for TensorRT compatibility
    input_dtype = input_trt.dtype
    if input_dtype == trt.int64 or input_dtype == trt.int32:
        np_dtype = np.int64 if input_dtype == trt.int64 else np.int32
    else:
        np_dtype = np.float32

    # Create constant with proper shape for broadcasting
    ndim = len(input_shape)
    const_shape = [1] * ndim if ndim > 0 else [1]
    other_data = np.full(const_shape, other, dtype=np_dtype)
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

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        promote_and_cast_tensors,
    )

    args = node.args
    input_node = args[0]
    other_node = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to lt must be a node, got {type(input_node)}")
    if not isinstance(other_node, torch.fx.Node):
        raise ValueError(f"Other input to lt must be a node, got {type(other_node)}")

    input_trt = input_map[input_node]
    other_trt = input_map[other_node]

    # TRT requires matching types for comparison ops
    input_trt, other_trt = promote_and_cast_tensors(
        network, input_trt, other_trt, f"lt_{node.name}"
    )

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
    
    Note: TensorRT select requires the condition to be boolean type.
    If the condition is not boolean, we convert it by comparing != 0.
    Also handles broadcasting for scalar inputs.
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

    # Get condition shape for broadcasting reference
    cond_shape = get_node_shape(condition_node)
    if cond_shape is None:
        try:
            cond_shape = tuple(condition_trt.shape)
        except (ValueError, TypeError):
            cond_shape = None
    
    max_ndim = len(cond_shape) if cond_shape else 0

    # Handle scalar and tensor inputs - track shapes for broadcasting
    import numpy as np
    
    def get_input_trt_and_shape(node_or_val, input_trt, name_suffix):
        """Get TRT tensor and its shape, handling scalars."""
        if input_trt is not None:
            # It's already a tensor
            if isinstance(node_or_val, torch.fx.Node):
                shape = get_node_shape(node_or_val)
            else:
                shape = None
            if shape is None:
                try:
                    shape = tuple(input_trt.shape)
                except (ValueError, TypeError):
                    shape = (1,)
            return input_trt, shape
        else:
            # It's a scalar - create constant with shape [1]
            val = float(node_or_val) if not isinstance(node_or_val, (int, float)) else node_or_val
            weights = trt.Weights(np.array([val], dtype=np.float32))
            const = network.add_constant([1], weights)
            const.name = f"where_{name_suffix}_const_{node.name}"
            return const.get_output(0), (1,)
    
    self_trt, self_shape = get_input_trt_and_shape(self_node, self_trt, "self")
    other_trt, other_shape = get_input_trt_and_shape(other_node, other_trt, "other")
    
    # Update max_ndim based on all inputs
    max_ndim = max(max_ndim, len(self_shape), len(other_shape))
    
    def prepend_ones_to_shape(tensor, tensor_shape, target_ndim, name_suffix):
        """Prepend 1s to tensor shape for broadcasting."""
        current_ndim = len(tensor_shape)
        if current_ndim < target_ndim:
            diff = target_ndim - current_ndim
            new_shape = (1,) * diff + tuple(tensor_shape)
            shuffle = network.add_shuffle(tensor)
            shuffle.reshape_dims = trt.Dims(new_shape)
            shuffle.name = f"where_broadcast_{name_suffix}_{node.name}"
            return shuffle.get_output(0)
        return tensor
    
    # Broadcast all inputs to max_ndim
    if cond_shape and len(cond_shape) < max_ndim:
        condition_trt = prepend_ones_to_shape(condition_trt, cond_shape, max_ndim, "cond")
    self_trt = prepend_ones_to_shape(self_trt, self_shape, max_ndim, "self")
    other_trt = prepend_ones_to_shape(other_trt, other_shape, max_ndim, "other")

    # TensorRT select requires boolean condition
    # If condition is not boolean, convert it by comparing != 0
    # Pattern from TensorRT: cast to float, then compare with 0
    if condition_trt.dtype != trt.bool:
        # Cast condition to float32 first
        cast_layer = network.add_identity(condition_trt)
        cast_layer.set_output_type(0, trt.float32)
        cast_layer.name = f"where_cast_cond_{node.name}"
        float_condition = cast_layer.get_output(0)
        
        # Create zero constant for comparison with broadcast-compatible shape
        zero_shape = [1] * max_ndim if max_ndim > 0 else [1]
        zero_weights = trt.Weights(np.zeros(zero_shape, dtype=np.float32))
        zero_const = network.add_constant(zero_shape, zero_weights)
        zero_const.name = f"where_zero_const_{node.name}"
        
        # Compare condition != 0 to get boolean (using EQUAL then NOT)
        eq_layer = network.add_elementwise(
            float_condition, 
            zero_const.get_output(0), 
            trt.ElementWiseOperation.EQUAL
        )
        eq_layer.name = f"where_eq_zero_{node.name}"
        
        # NOT the result to get != 0
        not_layer = network.add_unary(eq_layer.get_output(0), trt.UnaryOperation.NOT)
        not_layer.name = f"where_not_{node.name}"
        condition_trt = not_layer.get_output(0)

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


@converter("aten.full.default", supports_dynamic_shapes=True)
def convert_full(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert PyTorch full to TensorRT.

    full.default(int[] size, Scalar fill_value, ...) -> Tensor

    Supports dynamic sizes: when size elements are FX Nodes (shape
    tensor outputs from sym_size/add/sub/floordiv), they are gathered
    from input_map and concatenated into a shape tensor for IFillLayer.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        _torch_dtype_to_numpy,
        resolve_shape,
    )

    args = node.args
    size = args[0]
    fill_value = args[1] if len(args) > 1 else 0.0

    # Determine dtype from node metadata or kwargs, defaulting to float32.
    np_dtype = np.float32
    dtype_kwarg = node.kwargs.get("dtype", None)
    if dtype_kwarg is not None and isinstance(dtype_kwarg, torch.dtype):
        np_dtype = _torch_dtype_to_numpy(dtype_kwarg)
    elif "val" in node.meta:
        val = node.meta["val"]
        if hasattr(val, "dtype"):
            np_dtype = _torch_dtype_to_numpy(val.dtype)

    # Check if any size element is an FX Node (shape tensor from scalar ops).
    has_node_sizes = any(isinstance(s, torch.fx.Node) for s in size)

    if has_node_sizes:
        # Build shape tensor from a mix of FX Nodes and concrete ints.
        components = []
        for i, s in enumerate(size):
            if isinstance(s, torch.fx.Node) and s in input_map:
                t = input_map[s]
                shuf = network.add_shuffle(t)
                shuf.reshape_dims = trt.Dims([1])
                shuf.name = f"full_dim{i}_{node.name}"
                cast = network.add_cast(shuf.get_output(0), trt.int32)
                cast.name = f"full_dim_i32_{i}_{node.name}"
                components.append(cast.get_output(0))
            else:
                c = network.add_constant(
                    [1], trt.Weights(np.array([int(s)], dtype=np.int32))
                )
                c.name = f"full_c{i}_{node.name}"
                components.append(c.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"full_shape_{node.name}"
        # Clamp each dim to min 1 so TRT can prove output is non-empty.
        ones_c = network.add_constant(
            [len(size)], trt.Weights(np.ones(len(size), dtype=np.int32)))
        ones_c.name = f"full_ones_{node.name}"
        shape_clamp = network.add_elementwise(
            shape_cat.get_output(0), ones_c.get_output(0), trt.ElementWiseOperation.MAX)
        shape_clamp.name = f"full_clamp_{node.name}"

        ndims = len(size)
        layer = network.add_fill([0] * ndims, trt.FillOperation.LINSPACE)
        layer.set_input(0, shape_clamp.get_output(0))
        start_c = network.add_constant(
            [], trt.Weights(np.array(float(fill_value), dtype=np.float32)))
        start_c.name = f"full_val_{node.name}"
        layer.set_input(1, start_c.get_output(0))
        delta_c = network.add_constant(
            [ndims], trt.Weights(np.zeros(ndims, dtype=np.float32)))
        delta_c.name = f"full_delta_{node.name}"
        layer.set_input(2, delta_c.get_output(0))
    else:
        # Convert size to list, resolving symbolic dims.
        if isinstance(size, (list, tuple)):
            shape = resolve_shape(size)
        else:
            shape = [resolve_shape([size])[0]]

        has_dynamic = any(d == -1 for d in shape)
        if has_dynamic:
            dummy_shape = [0] * len(shape)
            layer = network.add_fill(dummy_shape, trt.FillOperation.LINSPACE)
            safe_shape = [max(d, 1) for d in shape]
            shape_const = network.add_constant(
                [len(safe_shape)],
                trt.Weights(np.array(safe_shape, dtype=np.int32)))
            shape_const.name = f"full_shape_{node.name}"
            layer.set_input(0, shape_const.get_output(0))
            start_c = network.add_constant(
                [], trt.Weights(np.array(float(fill_value), dtype=np.float32)))
            start_c.name = f"full_val_{node.name}"
            layer.set_input(1, start_c.get_output(0))
            ndims = len(shape)
            delta_c = network.add_constant(
                [ndims], trt.Weights(np.zeros(ndims, dtype=np.float32)))
            delta_c.name = f"full_delta_{node.name}"
            layer.set_input(2, delta_c.get_output(0))
        else:
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


@converter("aten.arange.start_step", supports_dynamic_shapes=True)
def convert_arange(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.arange.start_step to a TRT constant or fill layer.

    For concrete args, computes the range on the host. For symbolic args
    (dynamic shapes), uses the output shape from node metadata with -1
    and lets TRT infer the length at runtime via IFillLayer LINSPACE.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required") from e

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        resolve_shape,
        resolve_sym_dim,
    )

    start = node.args[0]
    end = node.args[1] if len(node.args) > 1 else node.kwargs.get("end")
    step = node.args[2] if len(node.args) > 2 else node.kwargs.get("step", 1)

    start_val = resolve_sym_dim(start)
    end_val = resolve_sym_dim(end)

    # If all args are concrete, compute on host (original fast path).
    if isinstance(start_val, int) and start_val >= 0 and isinstance(end_val, int) and end_val >= 0:
        values = np.arange(start_val, end_val, step, dtype=np.float32)
        layer = network.add_constant(list(values.shape), trt.Weights(values))
        layer.name = f"arange_{node.name}"
        return layer.get_output(0)

    # Dynamic path: use TRT IFillLayer with LINSPACE.
    # Create with dummy shape [0], then set the actual shape via shape tensor.
    fill_layer = network.add_fill([0], trt.FillOperation.LINSPACE)
    if fill_layer is None:
        raise RuntimeError(f"Failed to create fill layer for arange {node.name}")

    # Set output shape as a shape tensor (input 0). If `end` is an FX Node,
    # get the TRT tensor for it from input_map (it was produced by another op).
    if isinstance(end, torch.fx.Node) and end in input_map:
        # end is a TRT tensor — reshape to [1] for shape input.
        # Clamp to min 1 so TRT can prove the output is non-empty.
        end_trt = input_map[end]
        shape_shuffle = network.add_shuffle(end_trt)
        shape_shuffle.reshape_dims = trt.Dims([1])
        shape_shuffle.name = f"arange_shape_{node.name}"
        cast_i32 = network.add_cast(shape_shuffle.get_output(0), trt.int32)
        cast_i32.name = f"arange_shape_i32_{node.name}"
        one_c = network.add_constant([1], trt.Weights(np.array([1], dtype=np.int32)))
        one_c.name = f"arange_one_{node.name}"
        clamp = network.add_elementwise(
            cast_i32.get_output(0), one_c.get_output(0), trt.ElementWiseOperation.MAX
        )
        clamp.name = f"arange_clamp_{node.name}"
        fill_layer.set_input(0, clamp.get_output(0))
    else:
        # Fallback: use output metadata shape
        output_meta = node.meta.get("val", None)
        if output_meta is not None and hasattr(output_meta, "shape"):
            out_shape = resolve_shape(output_meta.shape)
            # Replace -1 with a placeholder; TRT will infer from profile
            safe_shape = [max(d, 1) for d in out_shape]
            shape_const = network.add_constant(
                [len(safe_shape)],
                trt.Weights(np.array(safe_shape, dtype=np.int32)),
            )
            shape_const.name = f"arange_shape_const_{node.name}"
            fill_layer.set_input(0, shape_const.get_output(0))

    # Set start (scalar, rank 0) and delta (rank 1 for LINSPACE)
    start_val = float(start) if isinstance(start, (int, float)) else 0.0
    start_const = network.add_constant(
        [], trt.Weights(np.array(start_val, dtype=np.float32))
    )
    start_const.name = f"arange_start_{node.name}"
    fill_layer.set_input(1, start_const.get_output(0))

    step_val = float(step) if isinstance(step, (int, float)) else 1.0
    delta_const = network.add_constant(
        [1], trt.Weights(np.array([step_val], dtype=np.float32))
    )
    delta_const.name = f"arange_delta_{node.name}"
    fill_layer.set_input(2, delta_const.get_output(0))

    fill_layer.name = f"arange_{node.name}"
    return fill_layer.get_output(0)


@converter("aten.constant_pad_nd.default", supports_dynamic_shapes=True)
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
        if input_shape[dim] < 0:
            # Dynamic dim stays dynamic — don't add padding to the -1 marker.
            output_shape[dim] = -1
        else:
            output_shape[dim] = input_shape[dim] + left_pad + right_pad

    has_dynamic = any(d < 0 for d in output_shape)

    if not has_dynamic:
        # Static path — all dims concrete.
        layer = network.add_slice(
            input_trt, start=start, shape=output_shape, stride=[1] * ndim
        )
    else:
        # Dynamic path — build per-dim output shape tensor.
        # Use constants for concrete dims, gather+add for dynamic dims.
        # This preserves concrete dim info for downstream ops.
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"pad_shape_{node.name}"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"pad_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        # Build per-dim pad offset lookup.
        total_pad = [0] * ndim
        for i in range(num_padded_dims):
            dim = ndim - 1 - i
            total_pad[dim] = pad[2 * i] + pad[2 * i + 1]

        components = []
        for d in range(ndim):
            if output_shape[d] >= 0:
                # Concrete dim — use constant directly.
                c = network.add_constant(
                    [1], trt.Weights(np.array([output_shape[d]], dtype=np.int32))
                )
                c.name = f"pad_c{d}_{node.name}"
                components.append(c.get_output(0))
            else:
                # Dynamic dim — gather runtime value and add padding.
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([d], dtype=np.int32))
                )
                idx_c.name = f"pad_idx{d}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"pad_g{d}_{node.name}"
                if total_pad[d] != 0:
                    pad_c = network.add_constant(
                        [1], trt.Weights(np.array([total_pad[d]], dtype=np.int32))
                    )
                    pad_c.name = f"pad_off{d}_{node.name}"
                    add_op = network.add_elementwise(
                        g.get_output(0), pad_c.get_output(0),
                        trt.ElementWiseOperation.SUM,
                    )
                    add_op.name = f"pad_add{d}_{node.name}"
                    components.append(add_op.get_output(0))
                else:
                    components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"pad_outshape_{node.name}"

        layer = network.add_slice(
            input_trt, start=start, shape=[1] * ndim, stride=[1] * ndim
        )
        layer.set_input(2, shape_cat.get_output(0))

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


@converter("aten._to_copy.default")
def convert_to_copy(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten._to_copy to TRT identity or cast layer.

    _to_copy copies a tensor, optionally changing dtype. In TRT this is either
    a pass-through (same dtype) or an identity layer with output type set.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        torch_dtype_to_trt,
    )

    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to _to_copy must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]

    target_dtype = node.kwargs.get("dtype", None)
    if target_dtype is not None and target_dtype != input_trt.dtype:
        layer = network.add_identity(input_trt)
        if layer is None:
            raise RuntimeError(f"Failed to create identity layer for {node.name}")
        layer.set_output_type(0, torch_dtype_to_trt(target_dtype))
        layer.name = f"to_copy_{node.name}"
        return layer.get_output(0)

    return input_trt


@converter("aten.bitwise_not.default")
def convert_bitwise_not(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.bitwise_not to TRT unary NOT."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to bitwise_not must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]
    layer = network.add_unary(input_trt, trt.UnaryOperation.NOT)
    if layer is None:
        raise RuntimeError(f"Failed to create NOT layer for {node.name}")
    layer.name = f"bitwise_not_{node.name}"
    return layer.get_output(0)


@converter("aten.logical_and.default")
def convert_logical_and(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.logical_and to TRT elementwise AND."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    lhs_node = node.args[0]
    rhs_node = node.args[1]

    if not isinstance(lhs_node, torch.fx.Node):
        raise ValueError(f"LHS of logical_and must be a node, got {type(lhs_node)}")
    if not isinstance(rhs_node, torch.fx.Node):
        raise ValueError(f"RHS of logical_and must be a node, got {type(rhs_node)}")

    lhs_trt = input_map[lhs_node]
    rhs_trt = input_map[rhs_node]

    # TRT AND requires Bool inputs; cast non-Bool operands.
    if lhs_trt.dtype != trt.bool:
        cast_lhs = network.add_cast(lhs_trt, trt.bool)
        cast_lhs.name = f"logical_and_lhs_bool_{node.name}"
        lhs_trt = cast_lhs.get_output(0)
    if rhs_trt.dtype != trt.bool:
        cast_rhs = network.add_cast(rhs_trt, trt.bool)
        cast_rhs.name = f"logical_and_rhs_bool_{node.name}"
        rhs_trt = cast_rhs.get_output(0)

    layer = network.add_elementwise(lhs_trt, rhs_trt, trt.ElementWiseOperation.AND)
    if layer is None:
        raise RuntimeError(f"Failed to create AND layer for {node.name}")
    layer.name = f"logical_and_{node.name}"
    return layer.get_output(0)


@converter("aten.argmax.default")
def convert_argmax(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.argmax to TRT TopK with k=1.

    argmax(input, dim=None, keepdim=False) -> indices
    TRT's TopK layer returns both values and indices; we return only indices.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to argmax must be a node, got {type(input_node)}")

    input_trt = input_map[input_node]
    input_shape = input_trt.shape
    ndim = len(input_shape)

    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)

    if dim is None:
        # Flatten to [1, N] — TRT TopK requires at least 2D input
        total = 1
        for d in input_shape:
            total *= d
        shuffle = network.add_shuffle(input_trt)
        shuffle.reshape_dims = trt.Dims([1, total])
        shuffle.name = f"argmax_flatten_{node.name}"
        input_trt = shuffle.get_output(0)
        reduce_axes = 1 << 1  # reduce over dim 1 (the data dim)
        squeezed_dim = None
    else:
        if dim < 0:
            dim += ndim
        # TRT TopK requires at least 2D. If input is 1D, unsqueeze to [1, N].
        if ndim == 1:
            shuffle = network.add_shuffle(input_trt)
            shuffle.reshape_dims = trt.Dims([1, input_shape[0]])
            shuffle.name = f"argmax_unsqueeze_{node.name}"
            input_trt = shuffle.get_output(0)
            reduce_axes = 1 << 1
            squeezed_dim = dim
        else:
            reduce_axes = 1 << dim
            squeezed_dim = None

    topk_layer = network.add_topk(input_trt, trt.TopKOperation.MAX, 1, reduce_axes)
    if topk_layer is None:
        raise RuntimeError(f"Failed to create TopK layer for {node.name}")
    topk_layer.name = f"argmax_{node.name}"

    # TopK output 1 is the indices tensor
    indices = topk_layer.get_output(1)

    # Determine output shape
    if dim is None:
        # Full reduction — output is scalar, reshape to [1]
        squeeze = network.add_shuffle(indices)
        squeeze.reshape_dims = trt.Dims([1])
        squeeze.name = f"argmax_squeeze_{node.name}"
        indices = squeeze.get_output(0)
    elif not keepdim:
        out_shape = list(input_shape)
        out_shape.pop(dim)
        if not out_shape:
            out_shape = [1]
        squeeze = network.add_shuffle(indices)
        squeeze.reshape_dims = trt.Dims(out_shape)
        squeeze.name = f"argmax_squeeze_{node.name}"
        indices = squeeze.get_output(0)

    return indices


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
    "convert_logical_and",
    "convert_bitwise_not",
    "convert_to_copy",
    "convert_argmax",
    "convert_where",
    "convert_any_dim",
    "convert_full_like",
    "convert_full",
]
