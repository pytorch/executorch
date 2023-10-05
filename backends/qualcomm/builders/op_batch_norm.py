# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    OpBatchnorm,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class BatchNorm(NodeVisitor):
    target = "aten._native_batch_norm_legit_no_training.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = get_input_node(node, 0)
        input_tensor, use_memo = self.get_tensor_shape(input_node, node)

        mean_node, var_node, eps = node.args[3], node.args[4], 1e-5
        mean_tensor = getattr(mean_node.graph.owning_module, mean_node.target)
        var_tensor = getattr(var_node.graph.owning_module, var_node.target)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers if use_memo else {},
        )

        bias_node = get_input_node(node, 2)
        bias_tensor = getattr(bias_node.graph.owning_module, bias_node.target)
        filter_node = get_input_node(node, 1)
        filter_tensor = getattr(filter_node.graph.owning_module, filter_node.target)

        amount = (filter_tensor * mean_tensor) / torch.sqrt(var_tensor + eps)
        bias_tensor = bias_tensor - amount
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            bias_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        filter_tensor = filter_tensor / torch.sqrt(var_tensor + eps)
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        batch_norm_input_tensors = [
            input_tensor_wrapper,
            filter_tensor_wrapper,
            bias_tensor_wrapper,
        ]

        output_tensor, _ = self.get_tensor_shape(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        batch_norm_output_tensors = [output_tensor_wrapper]

        batch_norm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpBatchnorm.op_name,
        )
        batch_norm_op.AddInputTensors(batch_norm_input_tensors)
        batch_norm_op.AddOutputTensors(batch_norm_output_tensors)

        return batch_norm_op
