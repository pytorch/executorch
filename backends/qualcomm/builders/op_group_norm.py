# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpGroupNorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class GroupNormVisitor(NodeVisitor):
    target = ["aten.native_group_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        weight_node = node.args[1]
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        bias_node = node.args[2]
        bias_tensor = get_parameter(bias_node, self.edge_program)
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            bias_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        group = node.args[6]
        epsilon = node.args[7]

        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        group_norm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGroupNorm.op_name,
        )
        group_norm_op.AddInputTensors(
            [input_tensor_wrapper, weight_tensor_wrapper, bias_tensor_wrapper]
        )
        group_norm_op.AddOutputTensors([output_tensor_wrapper])
        group_norm_op.AddScalarParam(
            OpGroupNorm.param_epsilon,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {"data": np.float32(epsilon)},
        )
        group_norm_op.AddScalarParam(
            OpGroupNorm.param_group,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(group)},
        )

        return group_norm_op
