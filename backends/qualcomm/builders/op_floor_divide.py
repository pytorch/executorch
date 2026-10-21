# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import (
    OpElementWiseDivide,
    OpElementWiseMod,
    OpElementWiseSubtract,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)


def define_int_floor_divide(
    visitor: NodeVisitor,
    node: torch.fx.Node,
    nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
):
    """Integer floor division, exact in int32: (x - x mod y) / y.

    x - x mod y is a multiple of y, so the integer Divide's rounding mode does not
    matter, and Mod takes the divisor's sign like torch.
    """
    out_tensor = visitor.get_tensor(node, node)
    x, y = (
        visitor.define_tensor(
            input_node,
            node,
            visitor.get_tensor(input_node, node),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        for input_node in (visitor.get_node(arg) for arg in node.args[:2])
    )
    quant_encoding, quant_configs = visitor.get_quant_encoding_conf(node, node)

    def intermediate(suffix):
        return visitor.define_custom_tensor_wrapper(
            node_name=f"{node.name}_{suffix}",
            tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=visitor.get_data_type(out_tensor, quant_configs),
            quant_encoding=quant_encoding,
            quant_configs=quant_configs,
            dims=out_tensor.size(),
            tensor=out_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )

    def op(name, op_name, inputs, outputs):
        wrapper = PyQnnManager.PyQnnOpWrapper(
            name, QNN_OP_PACKAGE_NAME_QTI_AISW, op_name
        )
        wrapper.AddInputTensors(inputs)
        wrapper.AddOutputTensors(outputs)
        return wrapper

    remainder, multiple = intermediate("mod"), intermediate("sub")
    output = visitor.define_tensor(
        node,
        node,
        out_tensor,
        PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
        nodes_to_wrappers,
    )
    return [
        op(f"{node.name}_mod", OpElementWiseMod.op_name, [x, y], [remainder]),
        op(
            f"{node.name}_sub",
            OpElementWiseSubtract.op_name,
            [x, remainder],
            [multiple],
        ),
        op(node.name, OpElementWiseDivide.op_name, [multiple, y], [output]),
    ]


@register_node_visitor
class IntFloorDivide(NodeVisitor):
    """div with rounding_mode="floor" on integers. Integer floor_divide is handled in
    op_binary; float and "trunc" division are decomposed by DecomposeDivMode."""

    target = ["aten.div.Tensor_mode"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        rounding_mode = node.kwargs.get(
            "rounding_mode", node.args[2] if len(node.args) > 2 else None
        )
        if node.meta["val"].is_floating_point() or rounding_mode != "floor":
            return None
        return define_int_floor_divide(self, node, nodes_to_wrappers)
