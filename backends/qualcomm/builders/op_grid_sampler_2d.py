# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch

from executorch.backends.qualcomm.utils.constants import QCOM_DATA, QCOM_DTYPE

from .node_visitor import NodeVisitor, QNN_QUANT_TYPE_MAP, QNN_TENSOR_TYPE_MAP
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpGridSample, OpTranspose, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class GridSample(NodeVisitor):
    target = ["aten.grid_sampler_2d.default", "aten.grid_sampler_3d.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        grid_sample_op_list = []
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        grid_node = self.get_node(node.args[1])
        grid_tensor = self.get_tensor(grid_node, node)
        grid_tensor_wrapper = self.define_tensor(
            grid_node,
            node,
            grid_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        input_shape = input_node.meta["val"].shape
        input_rank = len(input_shape)
        if input_rank not in [4, 5]:
            warnings.warn(
                "[QNN Delegate Op Builder]: The shape is not supported, fallback op",
                stacklevel=1,
            )
            return

        # About this operator, in ATen, the layout of input_tensor and of grid_tensor are not identical.
        # But in HW they are all NHWC or NDHWC. So, we make shape transformation again.
        if input_rank == 4:
            dims_shape_back = (0, 3, 1, 2)
        elif input_rank == 5:
            dims_shape_back = (0, 4, 1, 2, 3)
        else:
            warnings.warn(
                f"[QNN Delegate Op Builder]: Not support rank {input_rank}, fallback op",
                stacklevel=1,
            )
            return

        grid_quant_encoding, grid_quant_configs = self.get_quant_encoding_conf(
            grid_node, node
        )
        grid_dtype = (
            QNN_TENSOR_TYPE_MAP[grid_tensor.dtype]
            if grid_quant_encoding
            == PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED
            else QNN_QUANT_TYPE_MAP[
                (
                    torch.uint16
                    if grid_quant_configs[QCOM_DTYPE] == torch.int32
                    else grid_quant_configs[QCOM_DTYPE]
                )
            ]
        )
        # transpose
        permute_output_tensor = grid_tensor.permute(dims=dims_shape_back)
        transpose_output_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_transpose",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=grid_dtype,
            quant_encoding=grid_quant_encoding,
            quant_configs=grid_quant_configs,
            dims=permute_output_tensor.size(),
            tensor=permute_output_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )

        permute_order = cast(List[int], dims_shape_back)
        permute_order_shape = [len(permute_order)]
        transpose_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTranspose.op_name,
        )
        transpose_op.AddInputTensors([grid_tensor_wrapper])
        transpose_op.AddOutputTensors([transpose_output_tensor_wrapper])
        transpose_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(permute_order_shape),
            permute_order_shape,
            np.array(permute_order, dtype=np.uint32),
            True,
        )
        grid_sample_op_list.append(transpose_op)

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        align_corners = node.args[4] if len(node.args) > 4 else False
        padding_mode = node.args[3] if len(node.args) > 3 else 0
        interpo_mode = node.args[2] if len(node.args) > 2 else 0

        grid_sample_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGridSample.op_name,
        )
        grid_sample_op.AddInputTensors(
            [input_tensor_wrapper, transpose_output_tensor_wrapper]
        )
        grid_sample_op.AddOutputTensors([output_tensor_wrapper])
        grid_sample_op.AddScalarParam(
            OpGridSample.param_align_corners,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: align_corners},
        )
        grid_sample_op.AddScalarParam(
            OpGridSample.param_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(interpo_mode)},
        )
        grid_sample_op.AddScalarParam(
            OpGridSample.param_padding_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(padding_mode)},
        )
        grid_sample_op_list.append(grid_sample_op)
        return grid_sample_op_list
