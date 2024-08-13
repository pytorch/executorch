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

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpLogSoftmax, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class LogSoftmax(NodeVisitor):
    target = ["aten._log_softmax.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)

        log_softmax_inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        log_softmax_input_tensors = [log_softmax_inp_tensor_wrapper]
        output_tensor = self.get_tensor(node, node)

        log_softmax_output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        log_softmax_output_tensors = [log_softmax_output_tensor_wrapper]

        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        # logsoftmax only supports last dimension for now, which is channel in QNN
        if dim != input_tensor.dim() - 1:
            return None

        log_softmax_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpLogSoftmax.op_name,
        )
        log_softmax_op.AddInputTensors(log_softmax_input_tensors)
        log_softmax_op.AddOutputTensors(log_softmax_output_tensors)

        log_softmax_op.AddScalarParam(
            OpLogSoftmax.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )
        return log_softmax_op
