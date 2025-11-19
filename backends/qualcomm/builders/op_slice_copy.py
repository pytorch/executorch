# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpStridedSlice, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class StrideSlice(NodeVisitor):
    target = ["aten.slice_copy.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            tensor_type,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER][dim]

        # --- parse & normalize pytorch dim ---
        pytorch_dim = int(node.args[1])
        rank = len(input_tensor.shape)
        if pytorch_dim < 0:
            pytorch_dim = pytorch_dim % rank

        # --- map pytorch dim -> QNN dim ---
        qnn_dim = pytorch_dim
        if QCOM_AXIS_ORDER in node.meta:
            axis_order = node.meta[QCOM_AXIS_ORDER] 
            qnn_dim = axis_order.index(pytorch_dim)     

        # --- size on the QNN axis ---
        size = int(input_tensor.shape[qnn_dim])

        # --- get start/end/step ---
        start = 0 if len(node.args) <= 2 or node.args[2] is None else int(node.args[2])
        end   = size
        if len(node.args) > 3 and node.args[3] is not None:
            end = int(node.args[3])
        step  = 1 if len(node.args) <= 4 or node.args[4] is None else int(node.args[4])

        # --- normalize negatives ---
        if start < 0: 
            start = start % size
        if end < 0: 
            end = end % size

        # --- clamp into valid range ---
        start = max(0, min(start, size))
        end = max(0, min(end,   size))

        # --- canonicalize for positive step ---
        if step == 0:
            step = 1
        if step > 0 and start > end:
            # empty slice (like Python []): make it start=end
            start = end
        elif step < 0:
            raise NotImplementedError("Negative step not supported in QNN StridedSlice")

        # --- build ranges in QNN axes ---
        ranges = []
        for q in range(rank):
            if q == qnn_dim:
                ranges.extend([start, end, step])
            else:
                ranges.extend([0, int(input_tensor.shape[q]), 1])

        range_shape = [rank, 3]

        stride_slice_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpStridedSlice.op_name,
        )
        stride_slice_op.AddInputTensors([input_tensor_wrapper])
        stride_slice_op.AddOutputTensors([output_tensor_wrapper])

        stride_slice_op.AddTensorParam(
            OpStridedSlice.param_ranges,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            len(range_shape),
            range_shape,
            np.array(ranges, dtype=np.int32),
            True,
        )

        return stride_slice_op
