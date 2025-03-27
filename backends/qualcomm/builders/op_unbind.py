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
from .qnn_constants import OpUnpack, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Unbind(NodeVisitor):
    target = ["aten.unbind.int"]

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
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        unbind_input_tensors = [input_tensor_wrapper]

        unbind_output_tensors = []
        for i in range(len(node.meta["val"])):
            output_tensor = self.get_tensor(node, node, i)
            output_tensor_wrapper = self.define_tensor(
                node,
                node,
                output_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
                wrapper_idx=i,
            )
            unbind_output_tensors.append(output_tensor_wrapper)

        dim = 0 if len(node.args) == 1 else cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)
        unbind_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpUnpack.op_name,
        )
        unbind_op.AddInputTensors(unbind_input_tensors)
        unbind_op.AddOutputTensors(unbind_output_tensors)

        unbind_op.AddScalarParam(
            OpUnpack.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        return unbind_op
