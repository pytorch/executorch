# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for TensorRT converters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import tensorrt as trt
import torch

# Type alias for tensor shapes
Shape = Tuple[int, ...]


def has_dynamic_shape(shape: Shape) -> bool:
    """Determine if the given shape has dynamic dimensions.

    In TensorRT, dynamic dimensions are represented as -1. This function
    checks if any dimension in the shape is dynamic.

    Args:
        shape: Shape of a tensor. A sequence of integers where -1 indicates
               a dynamic (unknown at build time) dimension.

    Returns:
        True if any dimension is -1 (dynamic), False otherwise.

    Example:
        >>> has_dynamic_shape((3, 224, 224))
        False
        >>> has_dynamic_shape((3, -1, -1))
        True
        >>> has_dynamic_shape((-1, 3, 224, 224))
        True
    """
    return any(dim == -1 for dim in shape)


@overload
def get_positive_dim(dim: int, dim_size: int) -> int: ...


@overload
def get_positive_dim(dim: Sequence[int], dim_size: int) -> Tuple[int, ...]: ...


def get_positive_dim(
    dim: Union[int, Sequence[int]], dim_size: int
) -> Union[int, Tuple[int, ...]]:
    """Convert negative dimension index to positive, or clamp to valid range.

    Given an integer or tuple representing dimension(s), transform negative
    indices to positive using Python's modulo semantics. Positive indices
    are clamped to the valid range [0, dim_size].

    Args:
        dim: A single integer or sequence of integers representing dimension
             indices. Negative values are converted using modulo (e.g., -1
             becomes dim_size - 1).
        dim_size: The total number of dimensions in the tensor.

    Returns:
        A positive integer or tuple of positive integers representing the
        same dimension(s) as the input.

    Example:
        >>> get_positive_dim(-1, 4)  # Last dimension of 4D tensor
        3
        >>> get_positive_dim(-2, 4)  # Second to last
        2
        >>> get_positive_dim(1, 4)   # Already positive
        1
        >>> get_positive_dim(10, 4)  # Clamped to dim_size
        4
        >>> get_positive_dim((-1, -2), 4)  # Tuple of dims
        (3, 2)
    """

    def positive_dim(d: int) -> int:
        if dim_size == 0:
            return 0
        if d < 0:
            return d % dim_size
        else:
            return min(d, dim_size)

    return (
        positive_dim(dim)
        if isinstance(dim, int)
        else tuple(positive_dim(d) for d in dim)
    )


def flatten_dims(
    input: Union[trt.ITensor, torch.Tensor, np.ndarray],
    start_dim: int,
    end_dim: int,
) -> Tuple[int, ...]:
    """Calculate the flattened shape for a range of dimensions.

    Given an input tensor and start/end dimension indices, compute the new
    shape that results from flattening those dimensions together into a
    single dimension.

    Args:
        input: An input tensor (TensorRT, PyTorch, or NumPy) whose shape
               will be used for the calculation.
        start_dim: The first dimension to flatten (inclusive). Negative
                   indices are supported.
        end_dim: The last dimension to flatten (inclusive). Negative
                 indices are supported.

    Returns:
        A tuple representing the new shape after flattening.

    Example:
        >>> # For a tensor with shape (2, 3, 4, 5)
        >>> flatten_dims(tensor, 1, 2)  # Flatten dims 1 and 2
        (2, 12, 5)  # 3 * 4 = 12
        >>> flatten_dims(tensor, 0, -1)  # Flatten all dims
        (120,)  # 2 * 3 * 4 * 5 = 120
        >>> flatten_dims(tensor, -2, -1)  # Flatten last two dims
        (2, 3, 20)  # 4 * 5 = 20
    """
    shape = input.shape
    dim_size = len(shape)
    start_dim = get_positive_dim(start_dim, dim_size)
    end_dim = get_positive_dim(end_dim, dim_size)

    # Calculate the product of dimensions being flattened
    num_elements = 1
    for i in range(start_dim, end_dim + 1):
        num_elements *= shape[i]

    # Construct new shape: dims before + flattened dim + dims after
    new_shape = tuple(shape[:start_dim]) + (num_elements,) + tuple(shape[end_dim + 1 :])

    return new_shape


def get_axes_for_reduce_op(
    dim: Union[int, Sequence[int]],
) -> int:
    """Generate the axes bitmask for TensorRT reduce operations.

    TensorRT reduce layers use a binary representation for axes selection.
    Each bit position corresponds to a dimension, and setting that bit
    indicates the dimension should be reduced.

    Args:
        dim: An integer or sequence of integers representing the dimension(s)
             to reduce. Must be non-negative (use get_positive_dim first if
             needed).

    Returns:
        An integer whose binary representation indicates which dimensions
        to reduce. For example, reducing dims 1 and 2 returns 6 (binary 110).

    Example:
        >>> get_axes_for_reduce_op(0)
        1  # Binary: 001
        >>> get_axes_for_reduce_op(1)
        2  # Binary: 010
        >>> get_axes_for_reduce_op(2)
        4  # Binary: 100
        >>> get_axes_for_reduce_op((1, 2))
        6  # Binary: 110
        >>> get_axes_for_reduce_op((0, 2))
        5  # Binary: 101
    """
    if isinstance(dim, int):
        dim = (dim,)

    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


@dataclass
class ConversionContext:
    """Context for TensorRT network conversion.

    This class holds state needed during network conversion, avoiding global state.
    A new context is created for each network build.

    Usage:
        ctx = ConversionContext(net=network)
        # Pass ctx to converters and utility functions
        set_layer_name(layer, node, "add", ctx=ctx)
    """

    net: trt.INetworkDefinition
    layer_counter: int = field(default=0)
    # Track layers per node to detect multi-layer converters
    node_layer_counts: Dict[str, int] = field(default_factory=dict)

    def next_counter(self) -> int:
        """Get next unique counter value for layer naming."""
        self.layer_counter += 1
        return self.layer_counter

    def get_unique_suffix(self, node_name: str) -> str:
        """Get unique suffix for a node's layer.

        Returns empty string for first layer, "_2", "_3", etc. for subsequent layers.
        This ensures unique names when a single node creates multiple TensorRT layers.
        """
        count = self.node_layer_counts.get(node_name, 0) + 1
        self.node_layer_counts[node_name] = count
        return "" if count == 1 else f"_{count}"


def torch_dtype_to_trt(dtype: torch.dtype) -> trt.DataType:
    """Convert PyTorch dtype to TensorRT DataType.
    """
    _TORCH_TO_TRT_DTYPE: Dict[torch.dtype, trt.DataType] = {
        torch.bool: trt.bool,
        torch.int8: trt.int8,
        torch.int32: trt.int32,
        torch.int64: trt.int64,
        torch.uint8: trt.uint8,
        torch.float16: trt.float16,
        torch.float32: trt.float32,
        torch.bfloat16: trt.bfloat16,
        torch.float8_e4m3fn: trt.fp8,
    }

    if dtype not in _TORCH_TO_TRT_DTYPE:
        raise TypeError(f"{dtype} is not supported by TensorRT")
    return _TORCH_TO_TRT_DTYPE[dtype]


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    """Convert TensorRT DataType to PyTorch dtype.
    """
    _TRT_TO_TORCH_DTYPE: Dict[trt.DataType, torch.dtype] = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.uint8: torch.uint8,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.bfloat16: torch.bfloat16,
        trt.fp8: torch.float8_e4m3fn,
    }

    if dtype not in _TRT_TO_TORCH_DTYPE:
        raise TypeError(f"{dtype} is not supported by PyTorch")
    return _TRT_TO_TORCH_DTYPE[dtype]


def get_trt_tensor(
    network: trt.INetworkDefinition,
    value: Any,
    name: str,
    dtype: Optional[torch.dtype] = None,
) -> trt.ITensor:
    """Convert a value to a TensorRT tensor.

    Handles:
    - TensorRT ITensor (returned as-is)
    - Python scalars (int, float) → constant tensor
    - PyTorch tensors → constant tensor (including FakeTensors/subclasses)
    - numpy arrays → constant tensor

    Note: Uses unset_fake_temporarily to handle tensor subclasses like FakeTensor
    that don't support .numpy() directly. This follows the TensorRT pattern.
    """
    if isinstance(value, trt.ITensor):
        return value

    if dtype is None:
        dtype = torch.float32

    if isinstance(value, (int, float)):
        value = np.array([value], dtype=_torch_dtype_to_numpy(dtype))
        return create_constant(network, value, name)

    if isinstance(value, torch.Tensor):
        # Handle tensor subclasses (FakeTensor, etc.) that don't support .numpy()
        # by temporarily exiting fake tensor mode. This follows TensorRT pattern.
        try:
            from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
            with unset_fake_temporarily():
                # Create a real tensor from the fake tensor's data if needed
                if hasattr(value, '_local_scalar_dense') or not value.is_contiguous():
                    value = value.contiguous()
                np_value = value.detach().cpu().numpy()
        except (ImportError, RuntimeError):
            # Fallback: try to convert via creating a new tensor
            try:
                np_value = _tensor_to_numpy(value)
            except RuntimeError:
                # Last resort: create tensor from metadata if available
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    # For FakeTensors, we may need to create a zero tensor as placeholder
                    # This should only happen during tracing, not actual execution
                    np_dtype = _torch_dtype_to_numpy(value.dtype)
                    np_value = np.zeros(tuple(value.shape), dtype=np_dtype)
                else:
                    raise RuntimeError(
                        f"Cannot convert tensor subclass {type(value)} to numpy. "
                        f"Tensor may be a FakeTensor from tracing."
                    )
        return create_constant(network, np_value, name)

    if isinstance(value, np.ndarray):
        return create_constant(network, value, name)

    raise TypeError(f"Cannot convert {type(value)} to TensorRT tensor")


def create_constant(
    network: trt.INetworkDefinition,
    value: np.ndarray,
    name: str,
) -> trt.ITensor:
    """Create a TensorRT constant tensor from numpy array.

    Note: TensorRT doesn't support int64 (i64), so we convert to int32.
    Also, TensorRT doesn't handle 0-d tensors well in elementwise ops,
    so we reshape scalars to 1-d tensors with shape (1,).
    """
    # TensorRT doesn't support int64 - convert to int32
    if value.dtype == np.int64:
        value = value.astype(np.int32)
    # Ensure float64 is converted to float32
    if value.dtype == np.float64:
        value = value.astype(np.float32)

    # TensorRT requires at least 1-d tensors for elementwise ops.
    # Reshape 0-d scalars to 1-d tensors with shape (1,).
    if value.ndim == 0:
        value = value.reshape((1,))

    layer = network.add_constant(value.shape, trt.Weights(value))
    layer.name = f"const_{name}"
    return layer.get_output(0)


def get_safe_shape(tensor: trt.ITensor) -> List[int]:
    """Get tensor shape safely, handling dynamic shapes.

    TensorRT tensors can have invalid shapes during network building
    (e.g., negative length for dynamic dimensions). This function
    safely extracts the shape as a list.

    Args:
        tensor: TensorRT tensor to get shape from.

    Returns:
        List of dimension sizes, or empty list if shape is invalid.
    """
    try:
        shape = tensor.shape
        if shape is None:
            return []
        shape_list = list(shape)
        return shape_list
    except (ValueError, TypeError):
        return []


def broadcast_tensors(
    network: trt.INetworkDefinition,
    tensors: Sequence[trt.ITensor],
    target_ndim: int,
    name_prefix: str = "broadcast",
    ctx: Optional[ConversionContext] = None,
) -> List[trt.ITensor]:
    """Broadcast tensors to target number of dimensions by prepending 1s.

    Args:
        network: TensorRT network definition.
        tensors: Sequence of TensorRT tensors to broadcast.
        target_ndim: Target number of dimensions.
        name_prefix: Prefix for naming the broadcast layers (should be unique per call).
        ctx: Optional conversion context for unique naming. If not provided,
             uses a simple index-based naming scheme.

    Returns:
        List of broadcasted tensors.
    """
    result = []
    for i, tensor in enumerate(tensors):
        shape = get_safe_shape(tensor)
        current_ndim = len(shape) if shape else target_ndim

        if current_ndim < target_ndim:
            diff = target_ndim - current_ndim
            existing_shape = tuple(shape) if shape else tuple([-1] * current_ndim)
            new_shape = (1,) * diff + existing_shape
            layer = network.add_shuffle(tensor)
            layer.reshape_dims = new_shape
            # Use context counter if available, otherwise use simple naming
            if ctx is not None:
                counter = ctx.next_counter()
                layer.name = f"{name_prefix}_bc_{i}_{counter}"
                output = layer.get_output(0)
                if output is not None:
                    output.name = f"{name_prefix}_bc_{i}_{counter}_out"
            else:
                layer.name = f"{name_prefix}_bc_{i}"
                output = layer.get_output(0)
            result.append(output if output is not None else layer.get_output(0))
        else:
            result.append(tensor)
    return result


def get_op_name(node: torch.fx.Node) -> str:
    """Extract operation name from an FX graph node.

    Returns the op name in format "namespace.op_name.overload" (e.g., "aten.add.Tensor").
    Handles torch.ops operations, Edge dialect ops, and built-in callables.

    For call_function nodes with a schema (torch.ops and EdgeOpOverload), the format is:
    - "aten.add.Tensor" for named overloads
    - "aten.add.default" for the default overload (empty overload_name)

    For other callables (e.g., operator.getitem), returns the function name.

    Args:
        node: FX graph node to extract operation name from.

    Returns:
        Operation name string, or empty string for non-call_function nodes.

    Example:
        >>> # For torch.ops.aten.add.Tensor
        >>> get_op_name(node)
        'aten.add.Tensor'
        >>> # For operator.getitem
        >>> get_op_name(node)
        'getitem'
    """
    if node.op != "call_function":
        return ""

    target = node.target

    # Handle torch.ops operations (e.g., torch.ops.aten.add.Tensor)
    # and Edge dialect ops (EdgeOpOverload)
    if hasattr(target, "_schema"):
        schema = target._schema
        # Extract schema name which is in format "aten::add"
        base_name = schema.name.replace("::", ".")
        # Append overload name if present (e.g., "Tensor" from "add.Tensor")
        # Note: For the "default" overload, overload_name is an empty string "",
        # so we use "default" as the overload name in that case.
        if hasattr(schema, "overload_name"):
            overload_name = schema.overload_name
            if overload_name:
                return f"{base_name}.{overload_name}"
            else:
                # Empty overload_name means "default" overload
                return f"{base_name}.default"
        return base_name

    # Handle callable with __module__ and __name__ (e.g., operator.getitem)
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        module = target.__module__
        name = target.__name__
        if "aten" in module:
            return f"aten.{name}"
        return name

    # Fallback cases
    if hasattr(target, "__name__"):
        return target.__name__
    if hasattr(target, "name"):
        return target.name()
    return str(target)


def get_node_dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
    """Extract dtype from FX node metadata if available."""
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return val.dtype
        if isinstance(val, (list, tuple)) and len(val) > 0:
            if isinstance(val[0], torch.Tensor):
                return val[0].dtype
    return None


def get_node_shape(node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    """Extract shape from FX node metadata if available.

    During TensorRT network building, tensor shapes from TRT tensors can be
    unreliable (containing -1 for dynamic dimensions). The FX graph node
    metadata contains the correct shape information from the traced graph.

    Args:
        node: FX node that may contain shape metadata in node.meta["val"].

    Returns:
        Tuple of dimension sizes, or None if shape cannot be determined.
    """
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            if isinstance(val[0], torch.Tensor):
                return tuple(val[0].shape)
    return None


def set_layer_name(
    layer: trt.ILayer,
    node: torch.fx.Node,
    prefix: str = "",
    ctx: Optional[ConversionContext] = None,
) -> None:
    """Set descriptive name on TensorRT layer for debugging.

    Names layers using the pattern: [prefix_]<node_name>[_counter]

    When a ConversionContext is provided, uses its counter for unique naming.
    This is especially important when the same node creates multiple layers
    (e.g., addmm creates matmul + add layers).

    Args:
        layer: TensorRT layer to name.
        node: FX node that generated this layer.
        prefix: Optional prefix (e.g., "add", "conv2d").
        ctx: Optional conversion context for unique counter-based naming.
    """
    if ctx is not None:
        counter = ctx.next_counter()
        name = f"{prefix}_{node.name}_{counter}" if prefix else f"{node.name}_{counter}"
    else:
        name = f"{prefix}_{node.name}" if prefix else node.name
    layer.name = name
    # Set output tensor name to avoid TensorRT naming collisions
    if layer.num_outputs > 0:
        output = layer.get_output(0)
        if output is not None:
            output.name = f"{name}_out"


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert PyTorch dtype to numpy dtype."""
    _TORCH_TO_NUMPY = {
        torch.bool: np.bool_,
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # fp32 preserves all bf16 values; numpy bf16 support is unreliable
        torch.float32: np.float32,
        torch.float64: np.float64,
    }

    if dtype not in _TORCH_TO_NUMPY:
        raise TypeError(f"{dtype} is not supported for numpy conversion")
    return _TORCH_TO_NUMPY[dtype]


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array, handling unsupported dtypes.

    NumPy doesn't support bfloat16 or float8 natively. This helper
    converts such tensors to float32 before calling .numpy(), which
    preserves all representable values. TRT's builder flags control
    the actual engine precision — the weight values just need to be
    numerically correct.
    """
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn):
        tensor = tensor.float()
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.numpy()


def get_trt_tensor_from_node(
    network: trt.INetworkDefinition,
    node: torch.fx.Node,
    input_map: Dict[torch.fx.Node, trt.ITensor],
    name: str,
) -> trt.ITensor:
    """Get or create TensorRT tensor from an FX node.

    This is the centralized utility for converters that need to handle
    node inputs that may be runtime tensors or lifted constants/buffers.
    1. Already in input_map (normal case for runtime tensors)
    2. A placeholder or get_attr node with tensor metadata (lifted constants/buffers)

    This is the centralized utility for converters that need to handle
    both runtime tensors and constant tensors from the graph.

    Args:
        network: TensorRT network definition for creating constant layers.
        node: FX node representing the input (may be placeholder, get_attr, or call_function).
        input_map: Mapping from FX nodes to their TensorRT tensors.
        name: Name for the created constant layer if one is needed.

    Returns:
        TensorRT tensor for the input node.

    Raises:
        ValueError: If the node cannot be converted to a TRT tensor.

    Example:
        >>> # In a converter:
        >>> input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)
    """
    # Fast path: node already converted
    if node in input_map:
        return input_map[node]

    # Handle lifted buffers/parameters/constants that aren't in input_map
    # These are placeholder nodes (for lifted constants) or get_attr nodes
    if node.op in ("placeholder", "get_attr"):
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            trt_tensor = get_trt_tensor(
                network, node.meta["val"], f"const_{name}"
            )
            input_map[node] = trt_tensor  # Cache for future use
            return trt_tensor

    raise ValueError(
        f"Node '{node.name}' not found in input_map and cannot be converted to constant. "
        f"Node op: {node.op}, target: {node.target}. "
        f"This may be a node that depends on an unconverted upstream node."
    )


def promote_types(
    lhs_dtype: trt.DataType,
    rhs_dtype: trt.DataType,
) -> trt.DataType:
    """Promote two TensorRT data types to a common type.

    This follows PyTorch's type promotion rules for binary operations.
    The promotion hierarchy is: bool < int8 < int32 < int64 < float16 < float32

    Args:
        lhs_dtype: TensorRT data type of left operand.
        rhs_dtype: TensorRT data type of right operand.

    Returns:
        The promoted TensorRT data type.
    """
    if lhs_dtype == rhs_dtype:
        return lhs_dtype

    # Convert TRT types to torch types for promotion
    lhs_torch = trt_dtype_to_torch(lhs_dtype)
    rhs_torch = trt_dtype_to_torch(rhs_dtype)

    # Use PyTorch's built-in type promotion
    promoted_torch = torch.promote_types(lhs_torch, rhs_torch)

    # Convert back to TRT type
    return torch_dtype_to_trt(promoted_torch)


def cast_trt_tensor(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
    name: str,
) -> trt.ITensor:
    """Cast a TensorRT tensor to a target data type.

    Uses TensorRT's identity layer with output type override for casting.

    Args:
        network: TensorRT network definition.
        tensor: Input TensorRT tensor to cast.
        target_dtype: Target TensorRT data type.
        name: Name for the cast layer.

    Returns:
        Cast TensorRT tensor, or original tensor if already correct type.
    """
    if tensor.dtype == target_dtype:
        return tensor

    identity_layer = network.add_cast(tensor, target_dtype)
    identity_layer.name = f"cast_{name}"
    return identity_layer.get_output(0)


def promote_and_cast_tensors(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
    name_prefix: str,
) -> Tuple[trt.ITensor, trt.ITensor]:
    """Promote and cast two tensors to a common type for binary operations.

    This ensures type consistency for elementwise operations by:
    1. Determining the promoted type using PyTorch's promotion rules
    2. Casting both tensors to the promoted type if needed

    Args:
        network: TensorRT network definition.
        lhs: Left operand TensorRT tensor.
        rhs: Right operand TensorRT tensor.
        name_prefix: Prefix for naming cast layers.

    Returns:
        Tuple of (lhs_cast, rhs_cast) with matching promoted types.
    """
    lhs_dtype = lhs.dtype
    rhs_dtype = rhs.dtype

    if lhs_dtype == rhs_dtype:
        return lhs, rhs

    promoted_dtype = promote_types(lhs_dtype, rhs_dtype)

    lhs_cast = cast_trt_tensor(network, lhs, promoted_dtype, f"{name_prefix}_lhs")
    rhs_cast = cast_trt_tensor(network, rhs, promoted_dtype, f"{name_prefix}_rhs")

    return lhs_cast, rhs_cast


def convert_binary_elementwise(
    network: trt.INetworkDefinition,
    node: torch.fx.Node,
    input_map: Dict[torch.fx.Node, Any],
    op_type: trt.ElementWiseOperation,
    op_name: str,
    ctx: Optional[ConversionContext] = None,
) -> trt.ITensor:
    """Shared helper for binary elementwise operations.

    Handles tensor + tensor, tensor + scalar, and scalar + tensor cases.
    Automatically handles type promotion and broadcasting.

    Args:
        network: TensorRT network definition.
        node: FX node representing the operation.
        input_map: Mapping from FX nodes to TensorRT tensors.
        op_type: TensorRT ElementWiseOperation type (SUM, PROD, SUB, DIV, etc.).
        op_name: Name for the operation (used in layer naming).
        ctx: Optional conversion context for unique naming.

    Returns:
        TensorRT tensor representing the result.

    Raises:
        ValueError: If required inputs are missing.
    """
    if len(node.args) < 2:
        raise ValueError(
            f"{op_name} requires at least 2 arguments, got {len(node.args)}"
        )

    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    # Get output dtype from node metadata
    dtype = get_node_dtype(node)

    # Get LHS tensor
    if isinstance(lhs_arg, torch.fx.Node):
        if lhs_arg not in input_map:
            raise ValueError(
                f"LHS node '{lhs_arg.name}' not found in input_map for {op_name}"
            )
        lhs = input_map[lhs_arg]
    else:
        lhs = get_trt_tensor(network, lhs_arg, f"{op_name}_lhs_{node.name}", dtype)

    # Get RHS tensor
    if isinstance(rhs_arg, torch.fx.Node):
        if rhs_arg not in input_map:
            raise ValueError(
                f"RHS node '{rhs_arg.name}' not found in input_map for {op_name}"
            )
        rhs = input_map[rhs_arg]
    else:
        rhs = get_trt_tensor(network, rhs_arg, f"{op_name}_rhs_{node.name}", dtype)

    # Type promotion
    lhs, rhs = promote_and_cast_tensors(network, lhs, rhs, f"{op_name}_{node.name}")

    # Get target ndim for broadcasting
    lhs_ndim = len(lhs.shape) if lhs.shape else 0
    rhs_ndim = len(rhs.shape) if rhs.shape else 0
    target_ndim = max(lhs_ndim, rhs_ndim)

    # Fall back to output shape from node metadata if needed
    if target_ndim == 0 and "val" in node.meta and hasattr(node.meta["val"], "shape"):
        target_ndim = len(node.meta["val"].shape)

    if target_ndim == 0:
        target_ndim = 1

    # Broadcast tensors
    lhs, rhs = broadcast_tensors(
        network, [lhs, rhs], target_ndim, f"{op_name}_{node.name}", ctx
    )

    # Create elementwise layer
    layer = network.add_elementwise(lhs, rhs, op_type)
    if layer is None:
        raise RuntimeError(
            f"Failed to create elementwise {op_name} layer for {node.name}"
        )
    set_layer_name(layer, node, op_name, ctx)

    return layer.get_output(0)
