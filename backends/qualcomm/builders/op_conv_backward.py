# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, Dict, List, Optional, Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np
import torch

from .node_visitor import dq_ops, NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpTranspose, OpTransposeConv2d, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


logger = logging.getLogger(__name__)


@register_node_visitor
class ConvBackward(NodeVisitor):
    """Lower input-gradient Conv2d to TransposeConv2d on HTP V69+.

    Reuses the forward Conv2d weight wrapper instead of materializing another
    static weight.
    """

    target = ["aten.convolution_backward.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _get_supported_input_nodes(
        self, node: torch.fx.Node
    ) -> Optional[Tuple[torch.fx.Node, torch.Tensor, torch.fx.Node]]:
        output_mask = cast(List[bool], node.args[10])
        if output_mask != [True, False, False]:
            logger.warning(
                "ConvBackward %s is unsupported: only grad_input is supported; "
                "got output_mask=%s",
                node.name,
                output_mask,
            )
            return None
        if node.args[7]:
            logger.warning(
                "ConvBackward %s is unsupported: transposed convolutions are not supported",
                node.name,
            )
            return None

        grad_output_node = node.args[0]
        if grad_output_node.target in dq_ops:
            logger.warning(
                "ConvBackward %s is unsupported: quantized grad_output is not supported",
                node.name,
            )
            return None
        grad_output_tensor = self.get_tensor(grad_output_node, node)
        if grad_output_tensor.dim() != 4:
            logger.warning(
                "ConvBackward %s is unsupported: grad_output must be rank 4; got rank %s",
                node.name,
                grad_output_tensor.dim(),
            )
            return None

        filter_node = node.args[2]
        if filter_node.target in dq_ops:
            logger.warning(
                "ConvBackward %s is unsupported: quantized weights are not supported",
                node.name,
            )
            return None
        return grad_output_node, grad_output_tensor, filter_node

    def _get_filter_and_grad_input_tensors(
        self, node: torch.fx.Node, filter_node: torch.fx.Node
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        filter_tensor = get_parameter(filter_node, self.edge_program)
        if filter_tensor is None:
            logger.warning(
                "ConvBackward %s is unsupported: weight must be a static parameter",
                node.name,
            )
            return None
        if not filter_tensor.is_floating_point():
            logger.warning(
                "ConvBackward %s is unsupported: weight must be floating point; got %s",
                node.name,
                filter_tensor.dtype,
            )
            return None
        if filter_tensor.dim() != 4:
            logger.warning(
                "ConvBackward %s is unsupported: weight must be rank 4; got rank %s",
                node.name,
                filter_tensor.dim(),
            )
            return None

        grad_input_tensor = self.get_tensor(node, node, 0)
        if grad_input_tensor.dim() != 4:
            logger.warning(
                "ConvBackward %s is unsupported: grad_input must be rank 4; got rank %s",
                node.name,
                grad_input_tensor.dim(),
            )
            return None
        return filter_tensor, grad_input_tensor

    def _get_transpose_conv_params(
        self,
        node: torch.fx.Node,
        grad_output_node: torch.fx.Node,
        filter_tensor: torch.Tensor,
    ) -> Optional[Tuple[List[int], List[List[int]], List[int]]]:
        stride = cast(List[int], node.args[4])
        padding = cast(List[int], node.args[5])
        dilation = cast(List[int], node.args[6])
        groups = cast(int, node.args[9])
        if (
            len(stride) != 2
            or len(padding) not in (1, 2)
            or len(dilation) != 2
            or any(value != 1 for value in dilation)
        ):
            logger.warning(
                "ConvBackward %s is unsupported: requires 2D stride/padding and dilation=1; "
                "got stride=%s, padding=%s, dilation=%s",
                node.name,
                stride,
                padding,
                dilation,
            )
            return None
        if groups != 1:
            logger.warning(
                "ConvBackward %s is unsupported: grouped convolutions are not supported; "
                "got groups=%s",
                node.name,
                groups,
            )
            return None
        if len(padding) == 1:
            padding = padding + padding
        padding_2d = [[value, value] for value in padding]

        # Use unpermuted PyTorch metadata; QNN-layout tensors can place spatial
        # axes differently.
        _, _, kernel_height, kernel_width = filter_tensor.shape
        grad_output_height, grad_output_width = grad_output_node.meta["val"].shape[-2:]
        grad_input_height, grad_input_width = node.meta["val"][0].shape[-2:]
        # Derive QNN's output padding from the desired grad_input shape rather
        # than using the forward convolution's output_padding argument.
        base_height = (
            stride[0] * (grad_output_height - 1) + kernel_height - 2 * padding[0]
        )
        base_width = stride[1] * (grad_output_width - 1) + kernel_width - 2 * padding[1]
        output_padding = [
            grad_input_height - base_height,
            grad_input_width - base_width,
        ]
        if any(
            value < 0 or value >= stride[index]
            for index, value in enumerate(output_padding)
        ):
            logger.warning(
                "ConvBackward %s is unsupported: derived output_padding=%s is invalid "
                "for stride=%s",
                node.name,
                output_padding,
                stride,
            )
            return None
        return stride, padding_2d, output_padding

    def _define_filter_hwoi(
        self,
        node: torch.fx.Node,
        filter_node: torch.fx.Node,
        filter_tensor: torch.Tensor,
        nodes_to_wrappers: Dict[str, PyQnnManager.TensorWrapper],
    ) -> Tuple[PyQnnManager.TensorWrapper, List[PyQnnManager.PyQnnOpWrapper]]:
        # Reuse the forward Conv2d's static HWIO wrapper. HWOI is named from the
        # forward Conv2d view; for this TransposeConv2d it is HWIO, produced from
        # the raw OIHW weight by permute(2, 3, 0, 1).
        filter_tensor_hwio = filter_tensor.permute(2, 3, 1, 0).contiguous()
        filter_wrapper = self.define_tensor(
            filter_node,
            node,
            filter_tensor_hwio,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        filter_hwoi_name = f"{filter_node.name}_hwoi"
        if filter_hwoi_name in nodes_to_wrappers:
            return nodes_to_wrappers[filter_hwoi_name][0], []

        filter_tensor_hwoi = filter_tensor_hwio.permute(0, 1, 3, 2)
        filter_hwoi_wrapper = self.define_custom_tensor_wrapper(
            node_name=filter_hwoi_name,
            tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=self.get_data_type(filter_tensor_hwoi, {}),
            quant_encoding=PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            quant_configs={},
            dims=filter_tensor_hwoi.size(),
            tensor=filter_tensor_hwoi,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        weight_transpose_op = PyQnnManager.PyQnnOpWrapper(
            f"{node.name}_weight_to_hwoi",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTranspose.op_name,
        )
        weight_transpose_op.AddInputTensors([filter_wrapper])
        weight_transpose_op.AddOutputTensors([filter_hwoi_wrapper])
        weight_transpose_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            1,
            [4],
            np.array([0, 1, 3, 2], dtype=np.uint32),
            True,
        )
        return filter_hwoi_wrapper, [weight_transpose_op]

    def _define_transpose_conv_op(
        self,
        node: torch.fx.Node,
        grad_output_wrapper: PyQnnManager.TensorWrapper,
        filter_hwoi_wrapper: PyQnnManager.TensorWrapper,
        grad_input_wrapper: PyQnnManager.TensorWrapper,
        stride: List[int],
        padding_2d: List[List[int]],
        output_padding: List[int],
    ) -> PyQnnManager.PyQnnOpWrapper:
        transpose_conv_op = PyQnnManager.PyQnnOpWrapper(
            f"{node.name}_transpose_conv",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTransposeConv2d.op_name,
        )
        transpose_conv_op.AddInputTensors([grad_output_wrapper, filter_hwoi_wrapper])
        transpose_conv_op.AddOutputTensors([grad_input_wrapper])
        transpose_conv_op.AddTensorParam(
            OpTransposeConv2d.param_stride,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            1,
            [len(stride)],
            np.array(stride, dtype=np.uint32),
            True,
        )
        transpose_conv_op.AddTensorParam(
            OpTransposeConv2d.param_pad_amount,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            2,
            [len(padding_2d), len(padding_2d[0])],
            np.array(padding_2d, dtype=np.uint32),
            True,
        )
        transpose_conv_op.AddTensorParam(
            OpTransposeConv2d.param_output_padding,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            1,
            [len(output_padding)],
            np.array(output_padding, dtype=np.uint32),
            True,
        )
        return transpose_conv_op

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnManager.TensorWrapper],
    ) -> Optional[List[PyQnnManager.PyQnnOpWrapper]]:
        # convolution_backward signature:
        #   args[0]: grad_output
        #   args[1]: input
        #   args[2]: weight (raw parameter [Cout, Cin, kH, kW])
        #   args[3]: bias_sizes
        #   args[4]: stride
        #   args[5]: padding
        #   args[6]: dilation
        #   args[7]: transposed
        #   args[8]: output_padding
        #   args[9]: groups
        #   args[10]: output_mask [grad_input, grad_weight, grad_bias]
        supported_inputs = self._get_supported_input_nodes(node)
        if supported_inputs is None:
            logger.warning("ConvBackward %s is unsupported", node.name)
            return None
        grad_output_node, grad_output_tensor, filter_node = supported_inputs
        tensors = self._get_filter_and_grad_input_tensors(node, filter_node)
        if tensors is None:
            return None
        filter_tensor, grad_input_tensor = tensors
        params = self._get_transpose_conv_params(node, grad_output_node, filter_tensor)
        if params is None:
            return None
        stride, padding_2d, output_padding = params

        grad_output_wrapper = self.define_tensor(
            grad_output_node,
            node,
            grad_output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        filter_hwoi_wrapper, weight_transpose_ops = self._define_filter_hwoi(
            node, filter_node, filter_tensor, nodes_to_wrappers
        )

        grad_input_wrapper = self.define_tensor(
            node,
            node,
            grad_input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        transpose_conv_op = self._define_transpose_conv_op(
            node,
            grad_output_wrapper,
            filter_hwoi_wrapper,
            grad_input_wrapper,
            stride,
            padding_2d,
            output_padding,
        )
        return weight_transpose_ops + [transpose_conv_op]
