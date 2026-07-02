# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
)

from .node_visitor import NodeVisitor, PER_CHANNEL_ENCODING, QNN_QUANT_TYPE_MAP
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpConvert, OpGather, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class Embedding(NodeVisitor):
    target = ["aten.embedding.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        op_wrapper_list = []
        weight_node = self.get_node(node.args[0])
        is_pcq_embedding = QCOM_QUANT_ATTRS in weight_node.meta and weight_node.meta[
            QCOM_QUANT_ATTRS
        ][QCOM_ENCODING] in (PER_CHANNEL_ENCODING)
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        indices_node = node.args[1]
        indices_tensor = self.get_tensor(indices_node, node)
        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            node,
            indices_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        gather_input_tensors = []
        if is_pcq_embedding:
            act_quant_encoding, act_quant_configs = self.get_quant_encoding_conf(
                node, node
            )
            act_dtype = (
                torch.uint16
                if act_quant_configs[QCOM_DTYPE] == torch.int32
                else act_quant_configs[QCOM_DTYPE]
            )
            convert_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_convert",
                tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=QNN_QUANT_TYPE_MAP[act_dtype],
                quant_encoding=act_quant_encoding,
                quant_configs=act_quant_configs,
                dims=weight_tensor.size(),
                tensor=weight_tensor,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            convert_op = PyQnnManager.PyQnnOpWrapper(
                node.name + "_convert",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConvert.op_name,
            )
            convert_op.AddInputTensors([weight_tensor_wrapper])
            convert_op.AddOutputTensors([convert_tensor_wrapper])
            op_wrapper_list.append(convert_op)
            gather_input_tensors.append(convert_tensor_wrapper)
        else:
            gather_input_tensors.append(weight_tensor_wrapper)
        gather_input_tensors.append(indices_tensor_wrapper)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            node_name=node.name,
        )
        gather_output_tensors = [output_tensor_wrapper]

        gather_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        # For now, default axis is zero.
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(0)},
        )
        op_wrapper_list.append(gather_op)

        return op_wrapper_list
