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
    QCOM_QUANT_MIN,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpElementWiseEqual, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Equal(NodeVisitor):
    target = ["aten.eq.Tensor", "aten.eq.Scalar"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        output_tensors = [output_tensor_wrapper]

        input_tensors = []
        for index in range(2):
            input_node = node.args[index]
            if isinstance(input_node, torch.fx.Node):
                input_tensor = self.get_tensor(input_node, node)
                tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE
            else:
                scalar = input_node
                input_tensor = torch.tensor(scalar, dtype=torch.float32)
                tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC

                # 'graph', 'name', 'op', 'target', 'args', and 'kwargs'
                input_node = torch.fx.Node(
                    node.graph,
                    node.name + "_runtime_scalar",
                    "call_function",
                    exir_ops.edge.aten.scalar_tensor.default,
                    (),  # args
                    {},  # kwargs
                )
                # Because the output data type of the ge node is boolean.
                # We need to take the quant attr from the non-scalar node.
                if quant_attrs := node.args[index ^ 1].meta.get(QCOM_QUANT_ATTRS):
                    quant_attrs = quant_attrs.copy()
                    quant_range = (
                        quant_attrs[QCOM_QUANT_MAX] - quant_attrs[QCOM_QUANT_MIN]
                    )
                    quant_attrs[QCOM_ZERO_POINT] = (
                        0 if scalar >= 0 else quant_attrs[QCOM_QUANT_MAX]
                    )
                    quant_attrs[QCOM_SCALE] = (
                        scalar / quant_range if scalar >= 0 else -scalar / quant_range
                    )
                    input_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

            input_tensor_wrapper = self.define_tensor(
                input_node,
                node,
                input_tensor,
                tensor_type,
                nodes_to_wrappers,
            )
            input_tensors.append(input_tensor_wrapper)

        eq_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseEqual.op_name,
        )
        eq_op.AddInputTensors(input_tensors)
        eq_op.AddOutputTensors(output_tensors)

        return eq_op
