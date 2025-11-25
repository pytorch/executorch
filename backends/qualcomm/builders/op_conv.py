# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA, QCOM_QUANT_ATTRS

from .node_visitor import NodeVisitor, PER_CHANNEL_ENCODING
from .node_visitor_manager import register_node_visitor
from .qnn_constants import (
    OpConv2d,
    OpConv3d,
    OpDepthWiseConv2d,
    OpTransposeConv2d,
    OpTransposeConv3d,
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
        output_padding=None,
        output_padding_shape=None,
        transpose_conv=False,
        groups=None,
    ) -> PyQnnManager.PyQnnOpWrapper:
        """
        This function is shared among Conv1D, Conv2D, and DepthWise Conv2D as most of the required parameters overlaps.
        """
        conv_op.AddInputTensors(conv_input_tensors)
        conv_op.AddOutputTensors(conv_output_tensors)
        conv_op.AddTensorParam(
            OP.param_stride,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(stride, dtype=np.uint32),
            True,
        )
        conv_op.AddTensorParam(
            OP.param_pad_amount,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                padding,
                dtype=np.uint32,
            ),
            True,
        )

        if transpose_conv:
            conv_op.AddTensorParam(
                OP.param_output_padding,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(output_padding_shape),
                output_padding_shape,
                np.array(output_padding, dtype=np.uint32),
                True,
            )
        else:
            conv_op.AddTensorParam(
                OP.param_dilation,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(dilation_shape),
                dilation_shape,
                np.array(dilation, dtype=np.uint32),
                True,
            )

        if groups is not None:
            conv_op.AddScalarParam(
                OP.param_group,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: np.uint32(groups)},
            )

        return conv_op

    def _reduce_bias_scales(
        self,
        node: torch.fx.Node,
        filter_node: torch.fx.Node,
        bias_node: torch.fx.Node,
        groups: int,
    ):
        """_summary_
        If transpose_conv has groups, need special handle for bias_node's per channel quant.
        Check _derived_bias_quant_spec under backends/qualcomm/quantizer/qconfig.py for more info.
        """

        filter_scales = filter_node.meta[QCOM_QUANT_ATTRS]["scales"]
        bias_scales = bias_node.meta[QCOM_QUANT_ATTRS]["scales"]
        bias_zero_points = bias_node.meta[QCOM_QUANT_ATTRS]["zero_points"]

        # Adding this condition to prevent reduce twice: op_validation and qnn_preprocess
        if filter_scales.numel() != bias_scales.numel():
            bias_scales = bias_scales.view(-1, groups)[:, 0]
            bias_zero_points = bias_zero_points.view(-1, groups)[:, 0]
            bias_node.meta[QCOM_QUANT_ATTRS]["scales"] = bias_scales
            bias_node.meta[QCOM_QUANT_ATTRS]["zero_points"] = bias_zero_points

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        assert (
            input_tensor.dim() != 3
        ), "All Conv1D should be converted to Conv2D in CanonicalizeConv,"
        assert input_tensor.dim() in {
            4,
            5,
        }, "Only Conv2d and Conv3d is supported in conv builder,"

        is_conv2d = input_tensor.dim() == 4
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        filter_node = self.get_node(node.args[1])
        filter_tensor = get_parameter(filter_node, self.edge_program)

        stride = cast(List[int], node.args[3])
        padding = cast(List[int], node.args[4])
        dilation = cast(List[int], node.args[5])
        output_padding = cast(List[int], node.args[7])
        groups = cast(int, node.args[8])

        # weight of pytorch OIHW(conv2d) / OIDHW(conv3d) or IOHW(conv_transpose2d) / IODHW(conv_transpose3d),
        # yet QNN is HWIO or DHWIO for both conv and conv_transpose.
        is_transpose_conv = cast(bool, node.args[6])
        if is_conv2d:
            filter_axis_order = (2, 3, 0, 1) if is_transpose_conv else (2, 3, 1, 0)
        else:
            filter_axis_order = (
                (2, 3, 4, 0, 1) if is_transpose_conv else (2, 3, 4, 1, 0)
            )
        filter_tensor = filter_tensor.permute(dims=filter_axis_order).contiguous()
        filter_tensor_wrapper = self.define_tensor(
            filter_node,
            node,
            filter_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        conv_input_tensors = [input_tensor_wrapper, filter_tensor_wrapper]
        if node.args[2] is not None:
            bias_node = self.get_node(node.args[2])
            # TODO: Double check on condition below once QNN supports transpose_conv with block_quant.
            # By checking node.args[1].target, only allow per_channel_quant to go through and bypass block_quant.
            if (
                is_transpose_conv
                and groups != 1
                and bias_node.meta.get(QCOM_QUANT_ATTRS) is not None
                and node.args[1].target in PER_CHANNEL_ENCODING
            ):
                self._reduce_bias_scales(node, filter_node, bias_node, groups)

            bias_tensor = get_parameter(bias_node, self.edge_program)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                node,
                bias_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
            conv_input_tensors.append(bias_tensor_wrapper)
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        conv_output_tensors = [output_tensor_wrapper]

        # Qnn filter tensor is (H, W, Cin, Cout) or (D, H, W, Cin, Cout)
        group_input_channels = filter_tensor.shape[-2]
        group_output_channels = int(filter_tensor.shape[-1] / groups)
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
        padding = [[x, x] for x in padding]

        stride_shape = [len(stride)]
        padding_shape = [len(padding), len(padding[0])]
        dilation_shape = [len(dilation)]
        output_padding_shape = [len(output_padding)]

        if is_transpose_conv:
            assert all(
                val == 1 for val in dilation
            ), "CanonicalizeConv pass should perform dilate for transpose_conv."
            op_class = OpTransposeConv2d if is_conv2d else OpTransposeConv3d
        elif is_depthwise_conv:
            assert is_conv2d, "DepthWise only supports Conv2d"
            op_class = OpDepthWiseConv2d
        else:
            op_class = OpConv2d if is_conv2d else OpConv3d

        conv_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            op_class.op_name,
        )
        conv_op = self._add_conv_op_parameter(
            op_class,
            conv_op,
            conv_input_tensors,
            conv_output_tensors,
            stride,
            stride_shape,
            padding,
            padding_shape,
            dilation,
            dilation_shape,
            output_padding,
            output_padding_shape,
            is_transpose_conv,
            None if is_depthwise_conv else groups,
        )

        return conv_op
