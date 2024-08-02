# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpGather, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Index(NodeVisitor):
    # schema = aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
    target = ["aten.index.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

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

        if len(node.args[1]) > 1:
            # TODO consider to implement it in a recursive way.
            raise NotImplementedError("Not support tuple of tensor.")

        indices_node = node.args[1][0]
        indices_tensor = self.get_tensor(indices_node, node).to(torch.int32)
        assert indices_tensor.size(0) != 0, "Not support empty indices list"

        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            indices_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        gather_input_tensors = [input_tensor_wrapper, indices_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        gather_output_tensors = [output_tensor_wrapper]

        gather_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        # If support tuple of tensor, need to refine it based on len
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {"data": np.int32(0)},
        )

        return gather_op
