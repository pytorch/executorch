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
from .qnn_constants import OpBatchnorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class BatchNorm(NodeVisitor):
    target = [
        "aten._native_batch_norm_legit_no_training.default",
        "aten._native_batch_norm_legit.no_stats",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def update_encoding(self, node: torch.fx.Node, tensor: torch.Tensor, eps):
        if isinstance(tensor, torch._subclasses.FakeTensor):
            return

        if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            # scale value equals to zero will cause failure in HTP
            diff = max(abs(tensor.max()), abs(tensor.min())) + eps
            quant_attrs[QCOM_SCALE] = (diff / quant_attrs[QCOM_QUANT_MAX]).item()

    def try_dequantize(self, node: torch.fx.Node, tensor: torch.Tensor):
        if tensor.dtype == torch.float:
            return tensor

        scale = node.meta[QCOM_QUANT_ATTRS][QCOM_SCALE]
        offset = node.meta[QCOM_QUANT_ATTRS][QCOM_ZERO_POINT]
        return tensor.sub(offset).mul(scale).to(torch.float32).contiguous()

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)

        eps = 1e-9
        if "no_stats" in str(node.target):
            mean_tensor = torch.Tensor([node.args[4]])
            var_tensor = torch.Tensor([node.args[5]])
        else:
            mean_tensor = get_parameter(node.args[3], self.edge_program)
            var_tensor = get_parameter(node.args[4], self.edge_program)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        batch_norm_input_tensors = [input_tensor_wrapper]

        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        batch_norm_output_tensors = [output_tensor_wrapper]

        n_feature = output_tensor.shape[-1 if QCOM_AXIS_ORDER in node.meta else 1]
        filter_node = self.get_node(node.args[1])
        if filter_node is not None:
            # dequantize here for post-process
            filter_tensor = self.try_dequantize(
                filter_node, get_parameter(filter_node, self.edge_program)
            )
        else:
            # 'graph', 'name', 'op', 'target', 'args', and 'kwargs'
            filter_node = torch.fx.Node(
                node.graph,
                node.name + "_filter",
                "call_function",
                exir_ops.edge.aten.scalar_tensor.default,
                (),  # args
                {},  # kwargs
            )
            filter_tensor = torch.ones(n_feature)
            if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
                quant_attrs = quant_attrs.copy()
                quant_range = quant_attrs[QCOM_QUANT_MAX] - quant_attrs[QCOM_QUANT_MIN]
                quant_attrs[QCOM_ZERO_POINT] = 0
                quant_attrs[QCOM_SCALE] = 1.0 / quant_range
                filter_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

        filter_tensor = filter_tensor / torch.sqrt(var_tensor + eps)
        self.update_encoding(filter_node, filter_tensor, eps)
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        batch_norm_input_tensors.append(filter_tensor_wrapper)

        bias_node = self.get_node(node.args[2])
        if bias_node is not None:
            # dequantize here for post-process
            bias_tensor = self.try_dequantize(
                bias_node, get_parameter(bias_node, self.edge_program)
            )
            amount = filter_tensor * mean_tensor
            bias_tensor = bias_tensor - amount
            self.update_encoding(bias_node, bias_tensor, eps)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                node,
                bias_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
            batch_norm_input_tensors.append(bias_tensor_wrapper)

        batch_norm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpBatchnorm.op_name,
        )
        batch_norm_op.AddInputTensors(batch_norm_input_tensors)
        batch_norm_op.AddOutputTensors(batch_norm_output_tensors)

        return batch_norm_op
