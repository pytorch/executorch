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

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPad, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Pad(NodeVisitor):
    target = [
        "aten.constant_pad_nd.default",
        "aten.pad.default",               
        # Add tests before adding these two to the list  
        # "aten.reflection_pad2d.default",
        # "aten.replication_pad2d.default",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        pad_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        pad_input_tensors = [pad_inp_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        pad_output_tensors = [output_tensor_wrapper]

        # ---- Pad amount ([rank, 2], uint32) ----
        pad_amount_shape = [input_tensor.dim(), 2]
        # PyTorch pad order is from the *last* dim: e.g. 2D = [L, R, T, B]
        pad_amount = np.reshape(
            np.array(cast(List[int], node.args[1]), dtype=np.int64), (-1, 2)
        )[:: -1]  # reverse to go from last->first to first->last

        # expand to all ranks if needed
        if pad_amount_shape[0] - pad_amount.shape[0] > 0:
            zeros = np.zeros((pad_amount_shape[0] - pad_amount.shape[0], 2), dtype=np.int64)
            pad_amount = np.concatenate((zeros, pad_amount), axis=0)

        # remap rows if backend axis order is provided (backend_pos -> pt_dim)
        if QCOM_AXIS_ORDER in node.meta:
            axis_order = list(node.meta[QCOM_AXIS_ORDER])  # e.g. (0,2,3,1)
            pad_amount = pad_amount[axis_order]

        pad_amount = pad_amount.astype(np.uint32, copy=False)

        # ---- Mode/scheme ----
        if len(node.args) >= 3 and isinstance(node.args[2], str):
            mode = node.args[2]
        else:
            # default to constant
            mode = "constant"

        scheme_map = {
            "constant": OpPad.Scheme.CONSTANT,
            "reflect":  OpPad.Scheme.MIRROR_REFLECT,
            "replicate": OpPad.Scheme.EDGE, # I think this is supposed to be correct, but the result is wrong
        }
        scheme_u32 = np.uint32(scheme_map[mode])

        # ---- Build op ----
        pad_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpPad.op_name
        )
        pad_op.AddInputTensors(pad_input_tensors)
        pad_op.AddOutputTensors(pad_output_tensors)

        pad_op.AddScalarParam(
            OpPad.param_scheme,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: scheme_u32}, # scheme (UINT32)
        )

        # pad_constant_value only for constant mode
        if mode == "constant":
            pad_value = None
            if len(node.args) > 2 and not isinstance(node.args[2], str):
                pad_value = node.args[2]
            if pad_value is None:
                pad_value = 0.0
            pad_op.AddScalarParam(
                OpPad.param_pad_constant_value,
                QNN_TENSOR_TYPE_MAP[type(pad_value)],
                {QCOM_DATA: pad_value},
            )

        # pad_amount tensor param (UINT32, shape [rank, 2])
        pad_op.AddTensorParam(
            OpPad.param_pad_amount,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(pad_amount_shape),                 
            pad_amount_shape,                      
            pad_amount,                            
            True,
        )

        return pad_op
