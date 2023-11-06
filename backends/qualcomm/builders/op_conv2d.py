# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpConv2d, OpDepthWiseConv2d, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class Conv2d(NodeVisitor):
    target = "aten.convolution.default"

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
        )

        filter_node = node.args[1]
        filter_tensor = get_parameter(filter_node, self.edge_program)
        filter_axis_order = (2, 3, 1, 0)
        filter_tensor = filter_tensor.permute(dims=filter_axis_order).contiguous()
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        conv_input_tensors = [input_tensor_wrapper, filter_tensor_wrapper]

        if node.args[2] is not None:
            bias_node = node.args[2]
            bias_tensor = get_parameter(bias_node, self.edge_program)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                bias_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
            conv_input_tensors.append(bias_tensor_wrapper)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        conv_output_tensors = [output_tensor_wrapper]

        stride = cast(List[int], node.args[3])
        padding = cast(List[int], node.args[4])
        dilation = cast(List[int], node.args[5])

        groups = cast(int, node.args[8])
        # Qnn filter tensor is (H, W, Cin, Cout)
        group_input_channels = filter_tensor.shape[2]
        group_output_channels = int(filter_tensor.shape[3] / groups)
        # 1) groups = input_channels (i.e. group_input_channels = 1)
        # 2) output_channels is a positive integer multiple of input channels
        # TODO: Currently, negative results will be zero with Depthwise conv2d when input_channel == groups == 1
        # and test on QNN 2.14 rc1. Need to carefully investigate.
        is_depthwise_conv = (
            (group_input_channels == 1)
            and (group_output_channels % group_input_channels == 0)
            and (groups > 2)
        )
        if len(padding) == 1:
            padding = padding + padding

        # args[6] = transposed
        if cast(bool, node.args[6]):
            print("Currently, No support for transposed convolution")
            return

        # args[7] = output padding
        if not all(out_pad == 0 for out_pad in cast(List[int], node.args[7])):
            print("QNN does not support output padding")
            return

        stride_shape = [len(stride)]
        padding_shape = [2, 2]
        dilation_shape = [len(dilation)]

        if is_depthwise_conv:
            conv_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpDepthWiseConv2d.op_name,
            )
            conv_op.AddInputTensors(conv_input_tensors)
            conv_op.AddOutputTensors(conv_output_tensors)

            conv_op.AddTensorParam(
                OpDepthWiseConv2d.param_stride,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(stride_shape),
                stride_shape,
                np.array(stride, dtype=np.uint32),
                True,
            )
            conv_op.AddTensorParam(
                OpDepthWiseConv2d.param_pad_amount,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(padding_shape),
                padding_shape,
                np.array(
                    [[padding[0], padding[0]], [padding[1], padding[1]]],
                    dtype=np.uint32,
                ),
                True,
            )
            conv_op.AddTensorParam(
                OpDepthWiseConv2d.param_dilation,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(dilation_shape),
                dilation_shape,
                np.array(dilation, dtype=np.uint32),
                True,
            )

        else:
            conv_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConv2d.op_name,
            )
            conv_op.AddInputTensors(conv_input_tensors)
            conv_op.AddOutputTensors(conv_output_tensors)
            conv_op.AddTensorParam(
                OpConv2d.param_stride,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(stride_shape),
                stride_shape,
                np.array(stride, dtype=np.uint32),
                True,
            )
            conv_op.AddTensorParam(
                OpConv2d.param_pad_amount,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(padding_shape),
                padding_shape,
                np.array(
                    [[padding[0], padding[0]], [padding[1], padding[1]]],
                    dtype=np.uint32,
                ),
                True,
            )
            conv_op.AddTensorParam(
                OpConv2d.param_dilation,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(dilation_shape),
                dilation_shape,
                np.array(dilation, dtype=np.uint32),
                True,
            )
            conv_op.AddScalarParam(
                OpConv2d.param_group,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {"data": np.uint32(groups)},
            )

        return conv_op
