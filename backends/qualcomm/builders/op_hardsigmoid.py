# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import numpy as np

from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_ELEMENTWISE_NEURON,
    QNN_OP_ELEMENT_WISE_RULES_ALPHA,
    QNN_OP_ELEMENT_WISE_RULES_BETA,
    QNN_OP_ELEMENT_WISE_RULES_OPERATION,
    ElementwiseNeuronOperation
)

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class HardSigmoidVisitor(NodeVisitor):
    target = "aten.hardsigmoid.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = get_input_node(node, 0)
        input_tensor, use_memo = self.get_tensor_shape(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers if use_memo else {},
        )

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        hardsigmoid_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENTWISE_NEURON
        )
        hardsigmoid_op.AddInputTensors([input_tensor_wrapper])
        hardsigmoid_op.AddOutputTensors([output_tensor_wrapper])
        hardsigmoid_op.AddScalarParam(
            QNN_OP_ELEMENT_WISE_RULES_OPERATION,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(ElementwiseNeuronOperation.HARD_SIGMOID.value)},
        )
        hardsigmoid_op.AddScalarParam(
            QNN_OP_ELEMENT_WISE_RULES_ALPHA,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {"data": np.float32(1.0 / 6)},
        )
        hardsigmoid_op.AddScalarParam(
            QNN_OP_ELEMENT_WISE_RULES_BETA,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {"data": np.float32(0.5)},
        )

        return hardsigmoid_op
