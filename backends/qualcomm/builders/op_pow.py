# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpElementWisePower, QNN_OP_PACKAGE_NAME_QTI_AISW


# TODO Add more class Like PowTensorTensor if needed
@register_node_visitor
class PowTensorScalar(NodeVisitor):
    target = ["aten.pow.Tensor_Scalar"]

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
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        pow_output_tensors = [output_tensor_wrapper]

        # tensor input
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)

        tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            tensor_type,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        # scalar input
        scalar = node.args[1]
        scalar_tensor = torch.tensor(scalar).to(torch.float32)

        # 'graph', 'name', 'op', 'target', 'args', and 'kwargs'
        scalar_node = torch.fx.Node(
            node.graph,
            node.name + "_runtime_scalar",
            "call_function",
            exir_ops.edge.aten.scalar_tensor.default,
            (),  # args
            {},  # kwargs
        )

        if pow_quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = pow_quant_attrs.copy()
            quant_range = quant_attrs["quant_max"] - quant_attrs["quant_min"]
            quant_attrs["zero_point"] = 0 if scalar >= 0 else quant_attrs["quant_max"]
            quant_attrs["scale"] = (
                scalar / quant_range if scalar >= 0 else -scalar / quant_range
            )
            scalar_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

        scalar_tensor_wrapper = self.define_tensor(
            scalar_node,
            scalar_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        pow_input_tensors = [input_tensor_wrapper, scalar_tensor_wrapper]

        pow_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWisePower.op_name,
        )
        pow_op.AddInputTensors(pow_input_tensors)
        pow_op.AddOutputTensors(pow_output_tensors)

        return pow_op
