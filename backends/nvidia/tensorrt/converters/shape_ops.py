# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TensorRT Converters for scalar shape arithmetic operations.

These ops compute derived sizes (e.g., (s18 - 1) // 8 + 1) that flow
into tensor-creating ops like arange/full.  In TRT they become
elementwise operations on int32 shape tensors so that derived
dimensions stay symbolic and don't create partition boundaries.

Supported operations:
- aten.sym_size.int: query a tensor dimension → add_shape + add_gather
- scalar add / sub / mul / floordiv → add_elementwise on shape tensors
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger = logging.getLogger(__name__)


def _scalar_to_shape_tensor(network: Any, val: Any, input_map: Dict, name: str) -> Any:
    """Convert a scalar value (int, SymInt, or FX Node) to a 1-element int32 shape tensor."""
    import tensorrt as trt
    import numpy as np

    if isinstance(val, torch.fx.Node):
        if val in input_map:
            t = input_map[val]
            # If already a tensor, ensure it's int32 and 1D
            if hasattr(t, "shape"):
                ndim = len(t.shape)
                if ndim == 0:
                    # Scalar tensor → reshape to [1]
                    shuf = network.add_shuffle(t)
                    shuf.reshape_dims = trt.Dims([1])
                    shuf.name = f"{name}_reshape"
                    t = shuf.get_output(0)
                cast = network.add_cast(t, trt.int32)
                cast.name = f"{name}_i32"
                return cast.get_output(0)
        raise ValueError(f"FX Node {val.name} not in input_map for {name}")

    # Concrete int constant
    c = network.add_constant([1], trt.Weights(np.array([int(val)], dtype=np.int32)))
    c.name = name
    return c.get_output(0)


@converter("aten.sym_size.int", supports_dynamic_shapes=True)
def convert_sym_size(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert aten.sym_size.int to TRT add_shape + add_gather."""
    import tensorrt as trt
    import numpy as np

    input_node = node.args[0]
    dim = node.args[1]

    if not isinstance(input_node, torch.fx.Node) or input_node not in input_map:
        raise ValueError(f"Input node not found in input_map for {node.name}")

    input_trt = input_map[input_node]

    shape_layer = network.add_shape(input_trt)
    shape_layer.name = f"sym_size_shape_{node.name}"

    shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
    shape_i32.name = f"sym_size_i32_{node.name}"

    idx = network.add_constant([1], trt.Weights(np.array([dim], dtype=np.int32)))
    idx.name = f"sym_size_idx_{node.name}"

    gather = network.add_gather(shape_i32.get_output(0), idx.get_output(0), axis=0)
    gather.name = f"sym_size_{node.name}"

    return gather.get_output(0)


def _binary_scalar_op(node, network, input_map, trt_op, op_name):
    """Generic binary scalar operation on shape tensors."""
    lhs = _scalar_to_shape_tensor(network, node.args[0], input_map, f"{op_name}_lhs_{node.name}")
    rhs = _scalar_to_shape_tensor(network, node.args[1], input_map, f"{op_name}_rhs_{node.name}")

    layer = network.add_elementwise(lhs, rhs, trt_op)
    layer.name = f"{op_name}_{node.name}"
    return layer.get_output(0)


@converter("aten.add.default", "add", supports_dynamic_shapes=True)
def convert_scalar_add(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert scalar add to TRT elementwise SUM on shape tensors."""
    import tensorrt as trt
    return _binary_scalar_op(node, network, input_map, trt.ElementWiseOperation.SUM, "scalar_add")


@converter("aten.sub.default", "sub", supports_dynamic_shapes=True)
def convert_scalar_sub(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert scalar sub to TRT elementwise SUB on shape tensors."""
    import tensorrt as trt
    return _binary_scalar_op(node, network, input_map, trt.ElementWiseOperation.SUB, "scalar_sub")


@converter("aten.mul.default", "mul", supports_dynamic_shapes=True)
def convert_scalar_mul(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert scalar mul to TRT elementwise PROD on shape tensors."""
    import tensorrt as trt
    return _binary_scalar_op(node, network, input_map, trt.ElementWiseOperation.PROD, "scalar_mul")


@converter("aten.floordiv.default", "floordiv", supports_dynamic_shapes=True)
def convert_scalar_floordiv(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """Convert scalar floordiv to TRT elementwise FLOOR_DIV on shape tensors."""
    import tensorrt as trt
    return _binary_scalar_op(node, network, input_map, trt.ElementWiseOperation.FLOOR_DIV, "scalar_floordiv")
