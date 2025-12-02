# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_QUANT_ATTRS,
    QCOM_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpRmsNorm, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class RmsNormVisitor(NodeVisitor):
    target = ["aten.rms_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        # args of node : ['input', 'normalized_shape', 'weight', 'eps']
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # should be a immutable list
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
        axes = [node.args[0].meta["val"].dim() - 1]
        axes_shape = [len(axes)]

        weight_node = self.get_node(node.args[2])
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        # Fake node, nn module seems to be inconsistent with document
        bias_tensor = torch.zeros(weight_tensor.shape, dtype=weight_tensor.dtype)
        bias_node = torch.fx.Node(
            node.graph,
            node.name + "_runtime_bias",
            "call_function",
            exir_ops.edge.aten.tensor.default,
            (),  # args
            {},  # kwargs
        )
        if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = quant_attrs.copy()
            quant_attrs[QCOM_ZERO_POINT] = 0
            bias_node.meta[QCOM_QUANT_ATTRS] = quant_attrs
        bias_tensor_wrapper = self.define_tensor(
            bias_node,
            node,
            bias_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        epsilon = torch.finfo(torch.float32).eps
        if len(node.args) > 3:
            epsilon = node.args[3]
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        rms_nrom_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpRmsNorm.op_name,
        )

        rms_nrom_op.AddInputTensors(
            [input_tensor_wrapper, weight_tensor_wrapper, bias_tensor_wrapper]
        )
        rms_nrom_op.AddOutputTensors([output_tensor_wrapper])
        rms_nrom_op.AddScalarParam(
            OpRmsNorm.param_epsilon,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(epsilon)},
        )
        rms_nrom_op.AddTensorParam(
            OpRmsNorm.param_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(axes_shape),
            axes_shape,
            np.array(axes, dtype=np.uint32),
            True,
        )

        return rms_nrom_op
