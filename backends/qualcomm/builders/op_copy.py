# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpReshape, OpTile, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Copy(NodeVisitor):
    target = ["aten.copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        # The aten copy support broadcasting, and therefore translated to
        # Reshape and Tile.
        # e.g., torch.ops.aten.copy.default(torch.rand(3,4,5), torch.rand(4,1))
        input_node = self.get_node(node.args[1])
        input_tensor = self.get_tensor(input_node, node)
        output_tensor = self.get_tensor(node, node)
        should_insert_tile = input_tensor.numel() != output_tensor.numel()
        reshape_node_name = (
            node.name + "_unsqueeze" if should_insert_tile else node.name
        )
        reshape_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        reshape_input_tensors = [reshape_inp_tensor_wrapper]
        if quant_attrs := input_node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = quant_attrs.copy()
            # Because there is no output after convert_pt2e, the QCOM_QUANT_ATTRS of node is none
            node.meta[QCOM_QUANT_ATTRS] = quant_attrs

        reshape_tensor = input_tensor
        while len(reshape_tensor.shape) < len(output_tensor.shape):
            reshape_tensor = reshape_tensor.unsqueeze(0)
        input_quant_encoding, input_quant_configs = self.get_quant_encoding_conf(
            input_node, node
        )
        reshape_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=reshape_node_name,
            tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=self.get_data_type(reshape_tensor, input_quant_configs),
            quant_encoding=input_quant_encoding,
            quant_configs=input_quant_configs,
            dims=reshape_tensor.size(),
            tensor=reshape_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        reshape_output_tensors = [reshape_tensor_wrapper]

        reshape_op = PyQnnManager.PyQnnOpWrapper(
            reshape_node_name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        reshape_op.AddInputTensors(reshape_input_tensors)
        reshape_op.AddOutputTensors(reshape_output_tensors)
        op_wrapper_list = [reshape_op]
        if should_insert_tile:
            output_tensor_wrapper = self.define_tensor(
                node,
                node,
                output_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            tile_output_tensors = [output_tensor_wrapper]
            tile_op = PyQnnManager.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpTile.op_name,
            )
            multiples = []
            for i in range(len(reshape_tensor.shape)):
                assert (
                    output_tensor.shape[i] % reshape_tensor.shape[i] == 0
                ), f"Shape mismatch at dim {i}: {output_tensor.shape[i]} not divisible by {reshape_tensor.shape[i]}"
                multiples.append(output_tensor.shape[i] // reshape_tensor.shape[i])
            tile_op.AddTensorParam(
                OpTile.param_multiples,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                1,
                [len(reshape_tensor.shape)],
                np.array(multiples, dtype=np.uint32),
                True,
            )
            tile_op.AddInputTensors(reshape_output_tensors)
            tile_op.AddOutputTensors(tile_output_tensors)
            op_wrapper_list.append(tile_op)
        return op_wrapper_list
