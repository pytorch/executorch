# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpCast, OpConvert, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class To(NodeVisitor):
    target = ["aten._to_copy.default", "dim_order_ops._to_dim_order_copy.default"]
    sufixed_8_offset_diff = 128
    sufixed_16_offset_diff = 32768
    epsilon = 1e-6
    sufixed_8 = {
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8,
    }
    sufixed_16 = {
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16,
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16,
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
        # Get real quant conf of input node
        _, inp_qconfs = self.get_quant_encoding_conf(input_node, input_node)
        inp_dtype = self.get_data_type(input_tensor, inp_qconfs)

        output_tensor = self.get_tensor(node, node)
        _, out_qconfs = self.get_quant_encoding_conf(node, node)
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
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        node_input_tensors = [input_tensor_wrapper]

        # if the output / input dtype is int64, we should cast it to int32 first
        # since int32 is the only source that can be cast into int64
        # this is mainly for validation purpose, redundant cast ops will be fused
        # in preprocess stage.
        ops = []
        if (
            node.meta["val"].dtype == torch.int64
            and input_node.meta["val"].dtype != torch.int32
        ) or (
            input_node.meta["val"].dtype == torch.int64
            and node.meta["val"].dtype != torch.int32
        ):
            input_quant_encoding, input_quant_configs = self.get_quant_encoding_conf(
                input_node, node
            )
            cast_intermediate_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_cast",
                tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=QNN_TENSOR_TYPE_MAP[torch.int32],
                quant_encoding=input_quant_encoding,
                quant_configs=input_quant_configs,
                dims=input_tensor.size(),
                tensor=input_tensor,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            cast_op = PyQnnManager.PyQnnOpWrapper(
                f"{node.name}_cast",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpCast.op_name,
            )
            node_input_tensors = [cast_intermediate_tensor_wrapper]
            cast_op.AddInputTensors([input_tensor_wrapper])
            cast_op.AddOutputTensors([cast_intermediate_tensor_wrapper])
            ops.append(cast_op)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        qnn_op = OpCast if self.is_cast_node(node) else OpConvert
        op = PyQnnManager.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op.op_name
        )
        op.AddInputTensors(node_input_tensors)
        op.AddOutputTensors([output_tensor_wrapper])
        ops.append(op)

        return ops
