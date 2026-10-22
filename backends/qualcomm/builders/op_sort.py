# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

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
class Sort(NodeVisitor):
    """Implement aten.sort via QNN TopK with k = dim_size.

    Limitations:
        - Only supports sorting along the last dimension (channel dim in NHWC).
          Sorting on other dimensions will return None (op not delegated).
        - Sort indices are output as int32 by QNN. In quantized models, consuming
          sort indices with ops like gather (which expect int64) may cause dtype mismatches.
          For quantized use cases, prefer consuming only the sorted values output.
    """

    target = ["aten.sort.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:

        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        # aten.sort signature: sort(input, dim=-1, descending=False)
        dim = cast(int, node.args[1]) if len(node.args) > 1 else -1
        descending = cast(bool, node.args[2]) if len(node.args) > 2 else False

        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        # Sort is TopK with k = full dimension size
        k = input_tensor.shape[dim]

        if dim != len(input_tensor.shape) - 1:
            warnings.warn(
                "[QNN Delegate Op Builder]: QNN currently only supports channel as dimension for TopK/Sort.",
                stacklevel=1,
            )
            return

        sort_input_tensors = [input_tensor_wrapper]

        output_val_tensor = self.get_tensor(node, node, 0)
        output_idx_tensor = self.get_tensor(node, node, 1).to(torch.int32)

        # QNN constraint, topk/sort output_0 requires having the same quant config as input
        node.meta[QCOM_QUANT_ATTRS] = input_node.meta.get(QCOM_QUANT_ATTRS)
        output_val_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_val_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # sort output_1 is index, do not quantize it.
        node.meta.pop(QCOM_QUANT_ATTRS, None)
        output_index_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_idx_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            wrapper_idx=1,
        )
        sort_output_tensors = [output_val_tensor_wrapper, output_index_tensor_wrapper]

        topk_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTopK.op_name,
        )
        topk_op.AddInputTensors(sort_input_tensors)
        topk_op.AddOutputTensors(sort_output_tensors)

        topk_op.AddScalarParam(
            OpTopK.param_k,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(k)},
        )

        # descending=True in sort corresponds to largest=True in TopK
        topk_op.AddScalarParam(
            OpTopK.param_largest,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: descending},
        )

        return topk_op
