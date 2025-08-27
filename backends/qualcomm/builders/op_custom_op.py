# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict, Iterable

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)

from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP


class CustomOp(NodeVisitor):
    target = ""
    op_package_info = QnnExecuTorchOpPackageInfo()

    def __init__(self, op_package_info: QnnExecuTorchOpPackageInfo, *args) -> None:
        super().__init__(*args)
        self.target = op_package_info.custom_op_name
        self.op_package_info = op_package_info

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        custom_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            self.op_package_info.op_package_name,
            self.op_package_info.qnn_op_type_name,
        )

        custom_input_tensors = []
        custom_attr_keys = [arg.name for arg in node.target._schema.arguments]
        if len(custom_attr_keys) != len(node.args):
            warnings.warn(
                f"Number of inputs ({len(node.args)}) mismatch the number of args ({len(custom_attr_keys)}) in schema for the custom node ({self.target}).",
                stacklevel=1,
            )
            return
        for arg, arg_name in zip(node.args, custom_attr_keys):
            if arg is None:
                continue
            if isinstance(arg, torch.fx.Node):
                input_tensor = self.get_tensor(arg, node)
                input_tensor_wrapper = self.define_tensor(
                    arg,
                    node,
                    input_tensor,
                    PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    nodes_to_wrappers,
                )
                custom_input_tensors.append(input_tensor_wrapper)
            elif isinstance(arg, Iterable):
                tensor_parm_shape = [len(arg)]
                custom_op.AddTensorParam(
                    arg_name,
                    QNN_TENSOR_TYPE_MAP[type(arg[0])],
                    len(tensor_parm_shape),
                    tensor_parm_shape,
                    np.array(arg),
                    True,
                )
            else:
                custom_op.AddScalarParam(
                    arg_name,
                    QNN_TENSOR_TYPE_MAP[type(arg)],
                    {QCOM_DATA: arg},
                )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        custom_output_tensors = [output_tensor_wrapper]

        custom_op.AddInputTensors(custom_input_tensors)
        custom_op.AddOutputTensors(custom_output_tensors)
        return custom_op
