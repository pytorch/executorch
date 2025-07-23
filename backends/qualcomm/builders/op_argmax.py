# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpArgmax, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class argmax(NodeVisitor):
    target = ["aten.argmax.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        output_tensor = self.get_tensor(node, node)
        argmax_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        argmax_input_tensors = [argmax_inp_tensor_wrapper]
        argmax_out_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor.to(torch.int32),
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        argmax_output_tensors = [argmax_out_tensor_wrapper]

        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        argmax_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpArgmax.op_name,
        )
        argmax_op.AddInputTensors(argmax_input_tensors)
        argmax_op.AddOutputTensors(argmax_output_tensors)

        argmax_op.AddScalarParam(
            OpArgmax.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            argmax_op.AddScalarParam(
                OpArgmax.param_keep_dims,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: keep_dims},
            )

        return argmax_op
