# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpIsNan, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class IsNan(NodeVisitor):
    target = ["aten.isnan.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)

        if input_tensor.dtype not in [torch.float32, torch.float16]:
            warnings.warn(
                "[QNN Delegate Op Builder]: QNN IsNan only supports FP32 or FP16 inputs.",
                stacklevel=1,
            )
            return None

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            self.get_tensor(input_node, node),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        input_tensors = [input_tensor_wrapper]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        output_tensors = [output_tensor_wrapper]
        isnan_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpIsNan.op_name,
        )
        isnan_op.AddInputTensors(input_tensors)
        isnan_op.AddOutputTensors(output_tensors)

        return isnan_op
