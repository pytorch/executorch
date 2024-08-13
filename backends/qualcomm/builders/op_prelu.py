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

from .node_visitor import get_parameter, NodeVisitor, register_node_visitor
from .qnn_constants import OpPRelu, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class PReLU(NodeVisitor):
    target = ["aten.leaky_relu.default", "aten.prelu.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        prelu_inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        if node.target.__name__ == "aten.leaky_relu.default":
            coeff = 1e-2 if len(node.args) < 2 else node.args[1]
            coeff_tensor = torch.full(input_tensor.shape, coeff).to(torch.float32)
        else:
            coeff_node = node.args[1]
            coeff_tensor = torch.zeros(input_node.meta["val"].shape)
            coeff = get_parameter(coeff_node, self.edge_program)
            # param nodes will be FakeTensor when doing partition
            # fill in random numeric for validation
            if isinstance(coeff, torch._subclasses.fake_tensor.FakeTensor):
                coeff = torch.ones(coeff.shape)
            # per-channel activation
            if coeff_node.meta["val"].shape[0] > 1:
                for i in range(input_node.meta["val"].shape[1]):
                    coeff_tensor = coeff_tensor.index_fill(
                        1, torch.tensor([i]), coeff[i]
                    )
                if QCOM_AXIS_ORDER in input_node.meta:
                    axis_order = input_node.meta[QCOM_AXIS_ORDER]
                    coeff_tensor = coeff_tensor.permute(dims=axis_order).contiguous()
                # simple min-max quantization
                coeff = torch.max(coeff).item()
            else:
                coeff = coeff.item()
                coeff_tensor = torch.full(input_tensor.shape, coeff).to(torch.float32)

        # 'graph', 'name', 'op', 'target', 'args', and 'kwargs'
        scalar_node = torch.fx.Node(
            node.graph,
            node.name + "_runtime_scalar",
            "call_function",
            exir_ops.edge.aten.full.default,
            (),  # args
            {},  # kwargs
        )
        if pow_quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = pow_quant_attrs.copy()
            quant_range = quant_attrs[QCOM_QUANT_MAX] - quant_attrs[QCOM_QUANT_MIN]
            # coeff is guaranteed to be positive
            quant_attrs[QCOM_ZERO_POINT] = 0
            quant_attrs[QCOM_SCALE] = coeff / quant_range
            scalar_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

        scalar_tensor_wrapper = self.define_tensor(
            scalar_node,
            coeff_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        prelu_input_tensors = [prelu_inp_tensor_wrapper, scalar_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        prelu_output_tensors = [output_tensor_wrapper]

        prelu_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPRelu.op_name,
        )
        prelu_op.AddInputTensors(prelu_input_tensors)
        prelu_op.AddOutputTensors(prelu_output_tensors)

        return prelu_op
