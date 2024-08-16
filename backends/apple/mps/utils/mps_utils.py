#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import cast, Optional, Union

import torch
from executorch.backends.apple.mps.serialization.mps_graph_schema import MPSDataType
from executorch.exir import ExportedProgram
from torch._export.utils import get_buffer, get_param, is_buffer, is_param


def get_input_node(node: torch.fx.Node, input_index: int) -> Union[torch.fx.Node, None]:
    return None if node is None else cast(torch.fx.Node, node.args[input_index])


def get_scalar_val(node: torch.fx.Node, input_index: int) -> Union[float, int]:
    return node.args[input_index]


def edge_dtype_to_mps_dtype(dtype: torch.dtype):
    if not hasattr(edge_dtype_to_mps_dtype, "map"):
        edge_dtype_to_mps_dtype.map = {
            torch.float16: MPSDataType.mps_data_type_float16,
            torch.float32: MPSDataType.mps_data_type_float32,
            torch.float64: MPSDataType.mps_data_type_float32,
            torch.bfloat16: MPSDataType.mps_data_type_bfloat16,
            torch.int8: MPSDataType.mps_data_type_int8,
            torch.int16: MPSDataType.mps_data_type_int16,
            torch.int32: MPSDataType.mps_data_type_int32,
            torch.int64: MPSDataType.mps_data_type_int64,
            torch.uint8: MPSDataType.mps_data_type_uint8,
            torch.bool: MPSDataType.mps_data_type_bool,
            torch.cfloat: MPSDataType.mps_data_type_complex_float32,
            torch.chalf: MPSDataType.mps_data_type_complex_float16,
        }
    try:
        return edge_dtype_to_mps_dtype.map[dtype]
    except KeyError:
        raise RuntimeError(f"Invalid data type: {dtype}")


def get_param_tensor(
    exp_prog: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    if node is None:
        return None
    elif is_param(exp_prog, node):
        return get_param(exp_prog, node)
    elif is_buffer(exp_prog, node):
        return get_buffer(exp_prog, node)
    elif is_get_attr(node):
        # Support both lifted and unlifted graph
        try:
            # Unlifted graph (coming from old exir.capture API)
            return getattr(node.graph.owning_module, node.target)
        except AttributeError:
            return getattr(exp_prog.graph_module, node.target)
    raise RuntimeError(f"unsupported param type, {node.op}.")


def is_get_attr(node: torch.fx.Node):
    """
    Returns true if the given node is a get attr node for a tensor of the model
    """
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def is_parameter(exp_prog: torch.export.ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Check if a node is a lifted parameter (static data like weights and bias are
    are supplied as inputs to the graph.

    Args:
        edge_program (torch.export.ExportedProgram): _description_
        node (torch.fx.Node): _description_

    Returns:
        bool: _description_
    """
    return is_get_attr(node) or is_param(exp_prog, node) or is_buffer(exp_prog, node)
