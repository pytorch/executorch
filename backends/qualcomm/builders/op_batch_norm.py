# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_SCALE,
)

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpBatchnorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class BatchNorm(NodeVisitor):
    target = ["aten._native_batch_norm_legit_no_training.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def update_encoding(self, node: torch.fx.Node, tensor: torch.Tensor, eps):
        if isinstance(tensor, torch._subclasses.FakeTensor):
            return

        if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            # scale value equals to zero will cause failure in HTP
            diff = max(abs(tensor.max()), abs(tensor.min())) + eps
            quant_attrs[QCOM_SCALE] = diff / quant_attrs[QCOM_QUANT_MAX]

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)

        mean_node, var_node, eps = node.args[3], node.args[4], 1e-9
        mean_tensor = get_parameter(mean_node, self.edge_program)
        var_tensor = get_parameter(var_node, self.edge_program)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        bias_node = node.args[2]
        bias_tensor = get_parameter(bias_node, self.edge_program)
        filter_node = node.args[1]
        filter_tensor = get_parameter(filter_node, self.edge_program)

        amount = (filter_tensor * mean_tensor) / torch.sqrt(var_tensor + eps)
        bias_tensor = bias_tensor - amount
        self.update_encoding(bias_node, bias_tensor, eps)
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            bias_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        filter_tensor = filter_tensor / torch.sqrt(var_tensor + eps)
        self.update_encoding(filter_node, filter_tensor, eps)
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        batch_norm_input_tensors = [
            input_tensor_wrapper,
            filter_tensor_wrapper,
            bias_tensor_wrapper,
        ]

        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
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
