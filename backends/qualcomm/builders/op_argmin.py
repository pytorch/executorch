# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP, register_node_visitor
from .qnn_constants import OpArgmin, OpCast, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Argmin(NodeVisitor):
    target = ["aten.argmin.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        op_wrapper_list = []
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        output_tensor = self.get_tensor(node, node)
        argmin_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        argmin_input_tensors = [argmin_inp_tensor_wrapper]

        # arg output is index, do not quantize it.
        node.meta.pop("quant_attrs", None)
        input_quant_encoding, input_quant_configs = self.get_quant_encoding_conf(
            input_node, node
        )

        argmin_intermediate_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_cast",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=QNN_TENSOR_TYPE_MAP[torch.int32],
            quant_encoding=input_quant_encoding,
            quant_configs=input_quant_configs,
            dims=output_tensor.size(),
            tensor=output_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )

        argmin_output_tensors = [argmin_intermediate_tensor_wrapper]

        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        argmin_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpArgmin.op_name,
        )
        argmin_op.AddInputTensors(argmin_input_tensors)
        argmin_op.AddOutputTensors(argmin_output_tensors)

        argmin_op.AddScalarParam(
            OpArgmin.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            argmin_op.AddScalarParam(
                OpArgmin.param_keep_dims,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: keep_dims},
            )

        op_wrapper_list.append(argmin_op)

        cast_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_cast",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpCast.op_name,
        )

        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        cast_op.AddInputTensors([argmin_intermediate_tensor_wrapper])
        cast_op.AddOutputTensors([output_tensor_wrapper])
        op_wrapper_list.append(cast_op)

        return op_wrapper_list
