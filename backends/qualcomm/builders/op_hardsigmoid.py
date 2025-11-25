# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpElementWiseNeuron, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class HardSigmoidVisitor(NodeVisitor):
    target = ["aten.hardsigmoid.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        hardsigmoid_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseNeuron.op_name,
        )
        hardsigmoid_op.AddInputTensors([input_tensor_wrapper])
        hardsigmoid_op.AddOutputTensors([output_tensor_wrapper])

        # The operation enum of hardsigmoid in QNN
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_operation,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(2)},
        )

        # The parameter used in Pytorch definition for hardsigmoid
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_alpha,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(1 / 6)},
        )
        hardsigmoid_op.AddScalarParam(
            OpElementWiseNeuron.param_beta,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(1 / 2)},
        )

        return hardsigmoid_op
