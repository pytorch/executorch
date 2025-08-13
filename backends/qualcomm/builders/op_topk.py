# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_DATA,
    QCOM_QUANT_ATTRS,
)

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpTopK, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class TopK(NodeVisitor):
    target = ["aten.topk.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:

        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        k = cast(int, node.args[1])

        if len(node.args) > 2:
            dim = cast(int, node.args[2])
            if dim < 0:
                dim = dim % len(input_tensor.shape)
            if QCOM_AXIS_ORDER in node.meta:
                dim = node.meta[QCOM_AXIS_ORDER].index(dim)
            if dim != len(input_tensor.shape) - 1:
                warnings.warn(
                    "[QNN Delegate Op Builder]: QNN currently only supports channel as dimension for topK.",
                    stacklevel=1,
                )
                return

        topk_input_tensors = [input_tensor_wrapper]

        output_val_tensor = self.get_tensor(node, node, 0)
        output_idx_tensor = self.get_tensor(node, node, 1).to(torch.int32)

        # QNN constraint, topk output_0 requires having the same quant config as input
        node.meta[QCOM_QUANT_ATTRS] = input_node.meta.get(QCOM_QUANT_ATTRS)
        output_val_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_val_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # topk output_1 is index, do not quantize it.
        node.meta.pop(QCOM_QUANT_ATTRS, None)
        output_index_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_idx_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            wrapper_idx=1,
        )
        topk_output_tensors = [output_val_tensor_wrapper, output_index_tensor_wrapper]

        topk_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTopK.op_name,
        )
        topk_op.AddInputTensors(topk_input_tensors)
        topk_op.AddOutputTensors(topk_output_tensors)

        topk_op.AddScalarParam(
            OpTopK.param_k,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(k)},
        )

        # As of QNN 2.26, QNN HTP backend only allows users to set this value to 1, or it will fail at op validation
        if len(node.args) > 3:
            largest = cast(bool, node.args[3])
            topk_op.AddScalarParam(
                OpTopK.param_largest,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: largest},
            )

        return topk_op
