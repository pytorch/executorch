from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch

from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpScatterNd, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class SliceScatterVisitor(NodeVisitor):
    target = ["aten.slice_scatter.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        value_node = self.get_node(node.args[1])
        value_tensor = self.get_tensor(value_node, node)
        value_tensor_wrapper = self.define_tensor(
            value_node,
            node,
            value_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dim = cast(int, node.args[2]) if len(node.args) > 2 else 0
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        start = (
            cast(int, node.args[3])
            if len(node.args) > 3 and node.args[3] is not None
            else 0
        )
        if start < 0:
            start = start % input_tensor.shape[dim]

        if len(node.args) > 4:
            end = min(cast(int, node.args[4]), input_tensor.shape[dim])
            if end < 0:
                end = end % input_tensor.shape[dim]
        else:
            end = input_tensor.shape[dim]

        step = node.args[5] if len(node.args) > 5 else 1

        target_index_shape = []
        ranges = []
        # Collect the index
        for i in range(dim + 1):
            if i == dim:
                target_range = torch.tensor(range(start, end, step), dtype=torch.int32)
                target_index_shape.append(target_range.size(-1))
                ranges.append(target_range)
            else:
                size = input_tensor.size(i)
                target_index_shape.append(size)
                ranges.append(torch.arange(size, dtype=torch.int32))
        # last dim means x-tuple index
        target_index_shape.append(dim + 1)
        target_index_tensor = (
            torch.cartesian_prod(*ranges).reshape(target_index_shape).contiguous()
        )

        target_index_node = torch.fx.Node(
            node.graph,
            node.name + "_target_index",
            "call_function",
            exir_ops.edge.aten.tensor.default,
            (),  # args
            {},  # kwargs
        )
        target_index_tensor_wrapper = self.define_tensor(
            target_index_node,
            node,
            target_index_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        slice_scatter_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterNd.op_name,
        )
        slice_scatter_op.AddInputTensors(
            [
                input_tensor_wrapper,
                target_index_tensor_wrapper,
                value_tensor_wrapper,
            ]
        )
        slice_scatter_op.AddOutputTensors([output_tensor_wrapper])

        return slice_scatter_op
