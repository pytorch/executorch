# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpElementWiseNeuron, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class HardSigmoidVisitor(NodeVisitor):
    target = ["aten.hardsigmoid.default"]

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

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        hardsigmoid_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseNeuron.op_name,
        )
        hardsigmoid_op.AddInputTensors([input_tensor_wrapper])
        hardsigmoid_op.AddOutputTensors([output_tensor_wrapper])

        # The operation enum of hardsigmoid in QNN
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_operation,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(2)},
        )

        # The parameter used in Pytorch definition for hardsigmoid
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_alpha,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(1 / 6)},
        )
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_beta,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(1 / 2)},
        )

        return hardsigmoid_op
