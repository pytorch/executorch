# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpInstanceNorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class InstanceNorm(NodeVisitor):
    target = ["aten.instance_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        weight_node = self.get_node(node.args[1])
        bias_node = self.get_node(node.args[2])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        input_tensor_wrappers = [input_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        output_tensor_wrappers = [output_tensor_wrapper]
        n_feature = output_tensor.shape[-1 if QCOM_AXIS_ORDER in node.meta else 1]

        if weight_node is not None:
            weight_tensor = get_parameter(weight_node, self.edge_program)
        else:
            # 'graph', 'name', 'op', 'target', 'args', and 'kwargs'
            weight_node = torch.fx.Node(
                node.graph,
                node.name + "_weight",
                "call_function",
                exir_ops.edge.aten.scalar_tensor.default,
                (),  # args
                {},  # kwargs
            )
            weight_tensor = torch.ones(n_feature)

            if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
                quant_attrs = quant_attrs.copy()
                quant_range = quant_attrs[QCOM_QUANT_MAX] - quant_attrs[QCOM_QUANT_MIN]
                quant_attrs[QCOM_ZERO_POINT] = 0
                quant_attrs[QCOM_SCALE] = 1.0 / quant_range
                weight_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        input_tensor_wrappers.append(weight_tensor_wrapper)

        if bias_node is not None:
            bias_tensor = get_parameter(bias_node, self.edge_program)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                node,
                bias_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
            input_tensor_wrappers.append(bias_tensor_wrapper)

        instance_norm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpInstanceNorm.op_name,
        )
        instance_norm_op.AddInputTensors(input_tensor_wrappers)
        instance_norm_op.AddOutputTensors(output_tensor_wrappers)

        return instance_norm_op
