# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpReshape, OpStft, OpTranspose, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class Stft(NodeVisitor):
    """
    Lowers the real-valued form of aten.stft (return_complex=False), see
    ConvertComplexStft, as Reshape -> Stft -> Transpose [-> Reshape].

    QNN Stft takes the signal as [batch, signal_length, 1] and produces
    [batch, frames, freq, 2], while torch produces [batch, freq, frames, 2]
    (without the batch dimension for a 1D signal). The window is zero-padded on
    both sides to n_fft, as torch does when win_length < n_fft.
    """

    target = ["aten.stft.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _define_native_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        source_node: torch.fx.Node,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.TensorWrapper:
        quant_encoding, quant_configs = self.get_quant_encoding_conf(source_node, node)
        return self.define_custom_tensor_wrapper(
            node_name=name,
            tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=self.get_data_type(tensor, quant_configs),
            quant_encoding=quant_encoding,
            quant_configs=quant_configs,
            dims=tensor.size(),
            tensor=tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )

    def _define_window(
        self,
        node: torch.fx.Node,
        n_fft: int,
        win_length: int,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> List[PyQnnManager.TensorWrapper]:
        window_node = node.args[4] if len(node.args) > 4 else None
        if window_node is None:
            return []
        window_node = self.get_node(window_node)
        window = get_parameter(window_node, self.edge_program)
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        return [
            self.define_tensor(
                window_node,
                node,
                window,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
        ]

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> List[PyQnnManager.PyQnnOpWrapper]:
        args = list(node.args) + [None] * (8 - len(node.args))
        n_fft = args[1]
        hop_length = args[2] or n_fft // 4
        win_length = args[3] or n_fft
        onesided = True if args[6] is None else args[6]

        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        signal = input_tensor.reshape(-1, input_tensor.shape[-1], 1)
        signal_name = f"{node.name}_signal"
        signal_wrapper = self._define_native_tensor(
            signal_name, signal, input_node, node, nodes_to_wrappers
        )
        signal_op = PyQnnManager.PyQnnOpWrapper(
            signal_name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpReshape.op_name
        )
        signal_op.AddInputTensors([input_tensor_wrapper])
        signal_op.AddOutputTensors([signal_wrapper])

        output_tensor = self.get_tensor(node, node)
        # torch layout with the batch dimension: [batch, freq, frames, 2]
        spectrum = output_tensor.reshape(-1, *output_tensor.shape[-3:])
        frames_name = f"{node.name}_frames"
        frames_wrapper = self._define_native_tensor(
            frames_name, spectrum.transpose(1, 2), node, node, nodes_to_wrappers
        )
        stft_op = PyQnnManager.PyQnnOpWrapper(
            frames_name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpStft.op_name
        )
        stft_op.AddInputTensors(
            [signal_wrapper]
            + self._define_window(node, n_fft, win_length, nodes_to_wrappers)
        )
        stft_op.AddOutputTensors([frames_wrapper])
        stft_op.AddScalarParam(
            OpStft.param_frame_step,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(hop_length)},
        )
        stft_op.AddScalarParam(
            OpStft.param_frame_length,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(n_fft)},
        )
        stft_op.AddScalarParam(
            OpStft.param_onesided,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: onesided},
        )

        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        is_batched = output_tensor.dim() == 4
        transpose_name = node.name if is_batched else f"{node.name}_transpose"
        transpose_wrapper = (
            output_tensor_wrapper
            if is_batched
            else self._define_native_tensor(
                transpose_name, spectrum, node, node, nodes_to_wrappers
            )
        )
        perm = [0, 2, 1, 3]
        transpose_op = PyQnnManager.PyQnnOpWrapper(
            transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpTranspose.op_name
        )
        transpose_op.AddInputTensors([frames_wrapper])
        transpose_op.AddOutputTensors([transpose_wrapper])
        transpose_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            1,
            [len(perm)],
            np.array(perm, dtype=np.uint32),
            True,
        )
        op_wrapper_list = [signal_op, stft_op, transpose_op]

        if not is_batched:
            unbatch_op = PyQnnManager.PyQnnOpWrapper(
                node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpReshape.op_name
            )
            unbatch_op.AddInputTensors([transpose_wrapper])
            unbatch_op.AddOutputTensors([output_tensor_wrapper])
            op_wrapper_list.append(unbatch_op)
        return op_wrapper_list
