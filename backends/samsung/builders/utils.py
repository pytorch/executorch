# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch
from executorch.backends.samsung.utils.utils import is_graph_input, is_graph_output
from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from torch.export import ExportedProgram

DATA_TYPE_STR_MAPPING = {
    torch.int8: "INT8",
    torch.uint8: "UINT8",
    torch.int16: "INT16",
    torch.uint16: "UINT16",
    torch.int32: "INT32",
    torch.int64: "INT64",
    torch.float16: "FLOAT16",
    torch.float32: "FLOAT32",
}

TORCH_TYPE_QTYPE_MAPPING = {
    torch.int8: torch.qint8,
    torch.uint8: torch.quint8,
    torch.int32: torch.qint32,
}


class TensorType(Enum):
    INPUT = 0
    OUTPUT = 1
    CONSTANT = 2
    FEATUREMAP = 3


def get_tensor_type(exported_program: ExportedProgram, tensor: torch.fx.Node) -> str:
    if is_graph_input(exported_program, tensor):
        return TensorType.INPUT
    elif is_graph_output(tensor):
        return TensorType.OUTPUT
    elif is_param_node(exported_program, tensor):
        return TensorType.CONSTANT
    else:
        return TensorType.FEATUREMAP


def get_map_dtype(dtype):
    if dtype not in DATA_TYPE_STR_MAPPING:
        raise RuntimeError("Data type cannot be decided: ", dtype)
    return DATA_TYPE_STR_MAPPING[dtype]


def get_tensor(exported_program: ExportedProgram, node: torch.fx.Node):
    if not is_param_node(exported_program, node):
        return node.meta["val"]
    tensor = get_param_tensor(exported_program, node)
    return tensor.contiguous()


def affine_type_to_str(ttype: TensorType):
    return str(ttype).removeprefix("TensorType.")
