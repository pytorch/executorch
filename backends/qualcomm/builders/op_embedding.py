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
    QCOM_QUANT_ATTRS,
    QCOM_ENCODING,
)

from .node_visitor import (
    NodeVisitor,
    PER_CHANNEL_ENCODING,
    QNN_QUANT_TYPE_MAP,
)
from .node_visitor_manager import register_node_visitor
from .qnn_constants import (
    OpConvert,
    OpGather,
    OpTranspose,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
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
        weight_node = self.get_node(node.args[0])
        is_pcq_embedding = (
            QCOM_QUANT_ATTRS in weight_node.meta and
            weight_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING] in (
                PER_CHANNEL_ENCODING
            )
        )
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
        gather_input_tensors = [weight_tensor_wrapper, indices_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        node_name = node.name
        if is_pcq_embedding:
            import copy
            node_quant_attrs = node.meta[QCOM_QUANT_ATTRS]
            new_quant_attrs = copy.deepcopy(node_quant_attrs)
            new_quant_attrs["scale"] = weight_node.meta[QCOM_QUANT_ATTRS]["scales"].max()
            new_quant_attrs["zero_point"] = 0
            new_quant_attrs["quant_max"] = 255
            new_quant_attrs["quant_min"] = 0
            new_quant_attrs["dtype"] = torch.uint8
            node.meta[QCOM_QUANT_ATTRS] = new_quant_attrs
            node_name += "_intermediate"

        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            node_name=node_name,
        )
        if is_pcq_embedding:
            node.meta[QCOM_QUANT_ATTRS] = node_quant_attrs

        gather_output_tensors = [output_tensor_wrapper]
        gather_op = PyQnnManager.PyQnnOpWrapper(
            node_name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        # For now, default axis is zero.
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: 0},
        )
        op_wrapper_list = [gather_op]

        if is_pcq_embedding:
            act_quant_encoding, act_quant_configs = self.get_quant_encoding_conf(
                node, node
            )
            act_dtype = (
                torch.uint16 if act_quant_configs[QCOM_DTYPE] == torch.int32
                else act_quant_configs[QCOM_DTYPE]
            )
            convert_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name,
                tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=QNN_QUANT_TYPE_MAP[act_dtype],
                quant_encoding=act_quant_encoding,
                quant_configs=act_quant_configs,
                dims=output_tensor.size(),
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            convert_op = PyQnnManager.PyQnnOpWrapper(
                node.name + "_convert",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConvert.op_name,
            )
            convert_op.AddInputTensors(gather_output_tensors)
            convert_op.AddOutputTensors([convert_tensor_wrapper])
            op_wrapper_list.append(convert_op)

        return op_wrapper_list
