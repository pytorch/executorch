# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpReshape, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Copy(NodeVisitor):
    target = ["aten.copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[1])
        input_tensor = self.get_tensor(input_node, node)
        copy_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        copy_input_tensors = [copy_inp_tensor_wrapper]

        if quant_attrs := input_node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = quant_attrs.copy()
            # Because there is no output after convert_pt2e, the QCOM_QUANT_ATTRS of node is none
            node.meta[QCOM_QUANT_ATTRS] = quant_attrs
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        copy_output_tensors = [output_tensor_wrapper]

        copy_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        copy_op.AddInputTensors(copy_input_tensors)
        copy_op.AddOutputTensors(copy_output_tensors)

        return copy_op
