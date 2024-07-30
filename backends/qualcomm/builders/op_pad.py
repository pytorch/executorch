# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP, register_node_visitor
from .qnn_constants import OpPad, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Pad(NodeVisitor):
    target = ["aten.constant_pad_nd.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        pad_inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        pad_input_tensors = [pad_inp_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        pad_output_tensors = [output_tensor_wrapper]

        pad_amount_shape = [input_tensor.dim(), 2]
        # pytorch padding start from the last index
        pad_amount = np.reshape(cast(List[int], node.args[1]), (-1, 2))[::-1].astype(
            np.uint32
        )
        # fullfill the pad amount for each idex of tensor
        if zero_amounts := pad_amount_shape[0] - pad_amount.shape[0]:
            pad_amount = np.concatenate(
                (np.array([(0, 0)] * zero_amounts), pad_amount)
            ).astype(np.uint32)

        if QCOM_AXIS_ORDER in node.meta:
            pad_amount = np.transpose(pad_amount, node.meta[QCOM_AXIS_ORDER])
        pad_amount_val = node.args[2]

        pad_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPad.op_name,
        )
        pad_op.AddInputTensors(pad_input_tensors)
        pad_op.AddOutputTensors(pad_output_tensors)

        # For now, we only support constant (0) padding due to torch implementation
        pad_op.AddScalarParam(
            OpPad.param_scheme,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(OpPad.Scheme.CONSTANT)},
        )

        pad_op.AddScalarParam(
            OpPad.param_pad_constant_value,
            QNN_TENSOR_TYPE_MAP[type(pad_amount_val)],
            {QCOM_DATA: pad_amount_val},
        )

        pad_op.AddTensorParam(
            OpPad.param_pad_amount,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(pad_amount_shape),
            pad_amount_shape,
            pad_amount,
            True,
        )

        return pad_op
