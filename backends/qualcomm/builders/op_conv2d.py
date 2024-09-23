# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import (
    OpConv2d,
    OpDepthWiseConv2d,
    OpExpandDims,
    OpReshape,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from .utils import get_parameter


@register_node_visitor
class Conv2d(NodeVisitor):
    target = ["aten.convolution.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _add_conv_op_parameter(
        self,
        OP,
        conv_op,
        conv_input_tensors,
        conv_output_tensors,
        stride,
        stride_shape,
        padding,
        padding_shape,
        dilation,
        dilation_shape,
        groups=None,
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        """
        This function is shared among Conv1D, Conv2D, and DepthWise Conv2D as most of the required parameters overlaps.
        """
        conv_op.AddInputTensors(conv_input_tensors)
        conv_op.AddOutputTensors(conv_output_tensors)
        conv_op.AddTensorParam(
            OP.param_stride,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(stride, dtype=np.uint32),
            True,
        )
        conv_op.AddTensorParam(
            OP.param_pad_amount,
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
            OP.param_dilation,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(dilation_shape),
            dilation_shape,
            np.array(dilation, dtype=np.uint32),
            True,
        )
        if groups is not None:
            conv_op.AddScalarParam(
                OP.param_group,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: np.uint32(groups)},
            )

        return conv_op

    def _define_conv1d(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        """
        Conv1D is a special case for convolutional operation. QNN does not support Conv1D, therefore,
        we need to cast from input -> Conv1d -> output to input -> unsqueeze -> Conv2d -> squeeze -> output.
        """
        op_wrapper_list = []  # op_wrapper to return
        unsqueeze_input_node = node.args[0]
        input_quant_encoding, input_quant_configs = self.get_quant_encoding_conf(
            unsqueeze_input_node,
        )

        unsqueeze_input_tensor = self.get_tensor(unsqueeze_input_node, node)
        unsqueeze_input_tensor_wrapper = self.define_tensor(
            unsqueeze_input_node,
            unsqueeze_input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        unsqueeze_output_tensor = unsqueeze_input_tensor.unsqueeze(1).contiguous()
        dtype = self.get_data_type(unsqueeze_output_tensor, input_quant_configs)
        unsqueeze_output_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_unsqueeze",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=dtype,
            quant_encoding=input_quant_encoding,
            quant_configs=input_quant_configs,
            dims=unsqueeze_output_tensor.size(),
            tensor=unsqueeze_output_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        unsqueeze_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_unsqueeze",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpExpandDims.op_name,
        )
        unsqueeze_op.AddInputTensors([unsqueeze_input_tensor_wrapper])
        unsqueeze_op.AddOutputTensors([unsqueeze_output_tensor_wrapper])
        unsqueeze_op.AddScalarParam(
            OpExpandDims.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(1)},
        )
        op_wrapper_list.append(unsqueeze_op)

        filter_node = node.args[1]
        filter_tensor = (
            get_parameter(filter_node, self.edge_program).unsqueeze(2).contiguous()
        )
        filter_axis_order = (2, 3, 1, 0)
        filter_tensor = filter_tensor.permute(dims=filter_axis_order).contiguous()
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        conv_input_tensors = [unsqueeze_output_tensor_wrapper, filter_tensor_wrapper]
        if node.args[2] is not None:
            bias_node = node.args[2]
            bias_tensor = get_parameter(bias_node, self.edge_program)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                bias_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
                is_input_tensor=False,
            )
            conv_input_tensors.append(bias_tensor_wrapper)

        stride = [1] + cast(List[int], node.args[3])
        padding = [0] + cast(List[int], node.args[4])
        dilation = [1] + cast(List[int], node.args[5])
        groups = cast(int, node.args[8])

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

        conv_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_squeeze",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpConv2d.op_name,
        )
        conv_output_tensor = self.get_tensor(node, node)
        conv_output_tensor = conv_output_tensor.unsqueeze(1).contiguous()
        dtype = self.get_data_type(conv_output_tensor, input_quant_configs)
        conv_output_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_squeeze",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=dtype,
            quant_encoding=input_quant_encoding,
            quant_configs=input_quant_configs,
            dims=conv_output_tensor.size(),
            tensor=conv_output_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        conv_op = self._add_conv_op_parameter(
            OpConv2d,
            conv_op,
            conv_input_tensors,
            [conv_output_tensor_wrapper],
            stride,
            stride_shape,
            padding,
            padding_shape,
            dilation,
            dilation_shape,
            groups,
        )
        op_wrapper_list.append(conv_op)

        squeeze_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        squeeze_output_tensor = self.get_tensor(node, node)
        squeeze_output_tensor_wrapper = self.define_tensor(
            node,
            squeeze_output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
            node_name=node.name,
        )
        squeeze_op.AddInputTensors([conv_output_tensor_wrapper])
        squeeze_op.AddOutputTensors([squeeze_output_tensor_wrapper])
        op_wrapper_list.append(squeeze_op)

        return op_wrapper_list

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:

        if get_parameter(node.args[1], self.edge_program).dim() == 3:
            return self._define_conv1d(node, nodes_to_wrappers)
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        filter_node = node.args[1]
        filter_tensor = get_parameter(filter_node, self.edge_program)
        # weight of pytorch OIHW, yet QNN is HWIO
        filter_axis_order = (2, 3, 1, 0)
        filter_tensor = filter_tensor.permute(dims=filter_axis_order).contiguous()
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            filter_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=False,
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
                is_input_tensor=False,
            )
            conv_input_tensors.append(bias_tensor_wrapper)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
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
            conv_op = self._add_conv_op_parameter(
                OpDepthWiseConv2d,
                conv_op,
                conv_input_tensors,
                conv_output_tensors,
                stride,
                stride_shape,
                padding,
                padding_shape,
                dilation,
                dilation_shape,
            )

        else:
            conv_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConv2d.op_name,
            )
            conv_op = self._add_conv_op_parameter(
                OpConv2d,
                conv_op,
                conv_input_tensors,
                conv_output_tensors,
                stride,
                stride_shape,
                padding,
                padding_shape,
                dilation,
                dilation_shape,
                groups,
            )

        return conv_op
