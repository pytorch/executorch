# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpCast, OpConvert, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class To(NodeVisitor):
    target = ["aten._to_copy.default"]
    sufixed_8_offset_diff = 128
    sufixed_16_offset_diff = 32768
    epsilon = 1e-6
    sufixed_8 = {
        PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
        PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8,
    }
    sufixed_16 = {
        PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16,
        PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16,
    }

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def is_cast_node(self, node):
        input_node = node.args[0]

        # Not a case which has two quant node, no need to consider the convert op
        if not all(
            [
                input_node.meta.get(QCOM_QUANT_ATTRS),
                node.meta.get(QCOM_QUANT_ATTRS),
            ]
        ):
            return True

        input_tensor = self.get_tensor(input_node, node)
        _, inp_qconfs = self.get_quant_encoding_conf(input_node, False)
        inp_dtype = self.get_data_type(input_tensor, inp_qconfs)

        output_tensor = self.get_tensor(node, node)
        _, out_qconfs = self.get_quant_encoding_conf(node, False)
        out_dtype = self.get_data_type(output_tensor, out_qconfs)
        is_qparam_castable = (
            lambda o1, o2, s1, s2, diff: abs(s1 - s2) < self.epsilon
            and abs(o1 - o2) == diff
        )

        if {inp_dtype, out_dtype} == self.sufixed_8:
            return is_qparam_castable(
                inp_qconfs["offset"],
                out_qconfs["offset"],
                inp_qconfs["scale"],
                out_qconfs["scale"],
                self.sufixed_8_offset_diff,
            )
        elif {inp_dtype, out_dtype} == self.sufixed_16:
            return is_qparam_castable(
                inp_qconfs["offset"],
                out_qconfs["offset"],
                inp_qconfs["scale"],
                out_qconfs["scale"],
                self.sufixed_16_offset_diff,
            )
        return False

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        output_tensor = self.get_tensor(node, node)

        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        qnn_op = OpCast if self.is_cast_node(node) else OpConvert
        op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op.op_name
        )
        op.AddInputTensors([input_tensor_wrapper])
        op.AddOutputTensors([output_tensor_wrapper])

        return op
