# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.builders.node_visitor_manager import get_node_visitors
from executorch.backends.qualcomm.builders.qnn_constants import OpTranspose
from executorch.backends.qualcomm.utils.constants import (
    QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS,
    QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS,
)
from executorch.exir import to_edge


class BuilderPipeline:
    def __init__(self, op_package_infos):
        self.op_package_infos = op_package_infos
        self.transpose_params = {}

    @staticmethod
    def export(module, sample_input):
        return to_edge(
            torch.export.export(module, sample_input, strict=True)
        ).exported_program()

    def visitors(self, edge_program):
        return get_node_visitors(edge_program, op_package_infos=self.op_package_infos)

    def export_custom(self, module, sample_input):
        edge_program = self.export(module, sample_input)
        targets = {info.custom_op_name for info in self.op_package_infos}
        nodes = [
            node
            for node in edge_program.graph.nodes
            if node.op == "call_function" and node.target.__name__ in targets
        ]
        return edge_program, nodes

    def export_chain(self, module, sample_input, producer_order, consumer_order):
        edge_program, (producer, consumer) = self.export_custom(module, sample_input)
        producer.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS] = {0: producer_order}
        consumer.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS] = {0: consumer_order}
        return edge_program, producer, consumer

    @staticmethod
    def assert_tensor(wrapper, dims, dtype=None):
        tensor = PyQnnManager.PyQnnTensorWrapper(wrapper)
        assert tuple(tensor.GetDims()) == dims
        if dtype is not None:
            assert tensor.GetDataType() == dtype

    def assert_transpose(self, operation, axis_order):
        assert operation.GetOpWrapper().GetOpType() == OpTranspose.op_name
        dtype, rank, dims, permutation = self.transpose_params[operation]
        assert dtype == PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32
        assert rank == 1
        assert dims == (len(axis_order),)
        assert permutation.dtype == np.uint32
        np.testing.assert_array_equal(permutation, axis_order)

    def build(self, edge_program):
        visitors = self.visitors(edge_program)
        wrappers = defaultdict(dict)
        operations = []
        for node in edge_program.graph.nodes:
            if node.op != "call_function":
                continue
            operation = visitors[node.target.__name__].define_node(node, wrappers)
            if isinstance(operation, list):
                operations.extend(operation)
            elif operation is not None:
                operations.append(operation)
        return operations, wrappers
