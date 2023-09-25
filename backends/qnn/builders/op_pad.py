# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, cast
import numpy as np
import torch

from executorch.backends.qnn.builders.node_visitor import (
    QNN_TENSOR_TYPE_MAP,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qnn.utils.qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_PAD,
    QNN_OP_PAD_PARAM_SCHEME,
    QNN_OP_PAD_SCHEME_CONSTANT,
    QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC,
    QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
    QNN_OP_PAD_SCHEME_EDGE,
    QNN_OP_PAD_PARAM_PAD_AMOUNT,
    QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
)
from executorch.backends.qnn.utils.utils import get_input_node

import executorch.backends.qnn.python.PyQnnWrapperAdaptor as PyQnnWrapper


@register_node_visitor
class Pad(NodeVisitor):
    target = "aten.constant_pad_nd.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = get_input_node(node, 0)
        input_tensor, use_memo = self.get_tensor_shape(input_node, node)
        pad_inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers if use_memo else {},
        )
        pad_input_tensors = [pad_inp_tensor_wrapper]

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        pad_output_tensors = [output_tensor_wrapper]

        pad_amount_shape = [input_tensor.dim(), 2]
        # pytorch padding start from the last index
        pad_amount = np.reshape(cast(List[int], node.args[1]), (-1, 2))[::-1].astype(np.uint32)
        # fullfill the pad amount for each idex of tensor
        if zero_amounts := pad_amount_shape[0] - pad_amount.shape[0]:
            pad_amount = np.concatenate(
                 (np.array([(0, 0)] * zero_amounts), pad_amount)
            ).astype(np.uint32)

        if "axis_order" in node.meta:
            pad_amount = np.transpose(pad_amount, node.meta["axis_order"])
        pad_amount_val = node.args[2]

        pad_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_PAD
        )
        pad_op.AddInputTensors(pad_input_tensors)
        pad_op.AddOutputTensors(pad_output_tensors)

        # For now, we only support constant (0) padding due to torch implementation
        pad_op.AddScalarParam(
            QNN_OP_PAD_PARAM_SCHEME,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(QNN_OP_PAD_SCHEME_CONSTANT)},
        )

        pad_op.AddScalarParam(
            QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
            QNN_TENSOR_TYPE_MAP[type(pad_amount_val)],
            {"data": pad_amount_val},
        )

        pad_op.AddTensorParam(
            QNN_OP_PAD_PARAM_PAD_AMOUNT,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(pad_amount_shape),
            pad_amount_shape,
            pad_amount,
            True,
        )

        return pad_op
