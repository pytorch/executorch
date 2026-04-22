# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_QUANT_ATTRS,
    QCOM_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpLayerNorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class LayerNormVisitor(NodeVisitor):
    target = ["aten.native_layer_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        # args of node : ['input', 'normalized_shape', 'weight', 'bias', 'eps']
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        normalized_shapes = node.args[1]
        if (
            len(normalized_shapes) != 1
            and normalized_shapes[0] != input_tensor.shape[-1]
        ):
            warnings.warn(
                "[QNN Delegate Op Builder]: Only supports normalization with last input dimension.",
                stacklevel=1,
            )
            return
        axis = [len(input_tensor.shape) - 1]
        axis_shape = [len(axis)]

        has_weight = len(node.args) > 2 and node.args[2] is not None
        if has_weight:
            weight_node = self.get_node(node.args[2])
            weight_tensor = get_parameter(weight_node, self.edge_program)
        else:
            # elementwise_affine=False: use all-ones weight as identity
            weight_tensor = torch.ones(normalized_shapes, dtype=torch.float32)
            weight_node = torch.fx.Node(
                node.graph,
                node.name + "_runtime_weight",
                "call_function",
                exir_ops.edge.aten.tensor.default,
                (),
                {},
            )
            if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
                quant_attrs = quant_attrs.copy()
                quant_attrs[QCOM_ZERO_POINT] = 0
                weight_node.meta[QCOM_QUANT_ATTRS] = quant_attrs
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        # Fake node: even when original bias is absent, QNN still needs it
        has_bias = len(node.args) > 3 and node.args[3] is not None
        if has_bias:
            bias_node = self.get_node(node.args[3])
            bias_tensor = get_parameter(bias_node, self.edge_program)
        else:
            bias_tensor = torch.zeros(normalized_shapes, dtype=torch.float32)
            bias_node = torch.fx.Node(
                node.graph,
                node.name + "_runtime_bias",
                "call_function",
                exir_ops.edge.aten.tensor.default,
                (),
                {},
            )
            if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
                quant_attrs = quant_attrs.copy()
                quant_attrs[QCOM_ZERO_POINT] = 0
                bias_node.meta[QCOM_QUANT_ATTRS] = quant_attrs
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            node,
            bias_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        epsilon = node.args[4] if len(node.args) > 4 else 1e-05

        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        layer_norm_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpLayerNorm.op_name,
        )
        layer_norm_op.AddInputTensors(
            [input_tensor_wrapper, weight_tensor_wrapper, bias_tensor_wrapper]
        )
        layer_norm_op.AddOutputTensors([output_tensor_wrapper])
        layer_norm_op.AddScalarParam(
            OpLayerNorm.param_epsilon,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(epsilon)},
        )
        layer_norm_op.AddTensorParam(
            OpLayerNorm.param_axes,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(axis_shape),
            axis_shape,
            np.array(axis, dtype=np.uint32),
            True,
        )

        return layer_norm_op
