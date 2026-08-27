# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_DATA,
    QCOM_QUANT_ATTRS,
)

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpScatterElements, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class ScatterElements(NodeVisitor):
    target = [
        "aten.scatter.src",
        "aten.scatter.value",
        "aten.scatter_add.default",
        "aten.scatter_reduce.two",
    ]

    # aten reduce string -> QNN reduction mode. "mean" / "amax" / "amin"
    # are intentionally absent: QNN ScatterElements cannot express them.
    reduce_str_to_reduction = {
        "sum": OpScatterElements.Reduction.ADD,
        "prod": OpScatterElements.Reduction.MUL,
    }

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _get_reduction(
        self, node: torch.fx.Node
    ) -> Optional[OpScatterElements.Reduction]:
        """
        Resolve the QNN reduction mode for this node, or None if the node
        cannot be represented by QNN ScatterElements (caller falls back to CPU).
        """
        op_name = node.target.__name__

        if op_name == "aten.scatter_add.default":
            return OpScatterElements.Reduction.ADD

        if op_name == "aten.scatter_reduce.two":
            # include_self is keyword-only in aten.scatter_reduce.two
            include_self = node.kwargs.get("include_self", True)
            if not include_self:
                # QNN always accumulates onto the existing values of 'self'
                return None

            reduce_str = (
                node.args[4] if len(node.args) > 4 else node.kwargs.get("reduce")
            )
            return self.reduce_str_to_reduction.get(reduce_str)

        # aten.scatter.src: plain overwrite
        return OpScatterElements.Reduction.NONE

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        reduction = self._get_reduction(node)
        if reduction is None:
            # unsupported reduce mode or include_self=False -> fall back to CPU
            return None

        # NOTE: QNN HTP only supports reduction != NONE in quantized mode. We
        # intentionally do not gate on that here: backend capability is resolved
        # by IsNodeSupportedByBackend during partitioning, so this builder stays
        # backend-agnostic and picks up capability changes across SDK versions
        # automatically. The fp case is covered in the rework tests.
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        index_node = self.get_node(node.args[2])
        index_tensor = self.get_tensor(index_node, node)
        index_tensor_wrapper = self.define_tensor(
            index_node,
            node,
            index_tensor.to(torch.int32),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # Handle both scatter.src (args[3] is a Node/tensor) and
        # scatter.value (args[3] is a scalar)
        if isinstance(node.args[3], torch.fx.Node):
            # scatter.src path: args[3] is a tensor node
            updates_node = self.get_node(node.args[3])
            updates_tensor = self.get_tensor(updates_node, node)
            updates_tensor_wrapper = self.define_tensor(
                updates_node,
                node,
                updates_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
        else:
            # scatter.value: expand scalar to a float static tensor matching index
            # shape. For quantized models, use input_node as quant source so
            # define_tensor derives correct encoding and handles quantization via
            # get_quant_tensor_value. For fp, use index_node to avoid tensor
            # aliasing issues with the input at runtime.
            scalar_val = node.args[3]
            updates_tensor = torch.full(
                index_tensor.shape, scalar_val, dtype=input_tensor.dtype
            )
            quant_source = (
                input_node if input_node.meta.get(QCOM_QUANT_ATTRS) else index_node
            )

            updates_tensor_wrapper = self.define_tensor(
                quant_source,
                node,
                updates_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
                node_name=f"{node.name}_updates_value",
                wrapper_idx=2,
            )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        dim = node.args[1]
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        scatter_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterElements.op_name,
        )
        scatter_op.AddInputTensors(
            [
                input_tensor_wrapper,
                index_tensor_wrapper,
                updates_tensor_wrapper,
            ]
        )
        scatter_op.AddOutputTensors([output_tensor_wrapper])

        scatter_op.AddScalarParam(
            OpScatterElements.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        scatter_op.AddScalarParam(
            OpScatterElements.param_reduction,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(reduction)},
        )

        return scatter_op
