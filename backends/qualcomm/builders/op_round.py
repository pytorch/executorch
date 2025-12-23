import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor

from .qnn_constants import OpElementWiseRound, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Round(NodeVisitor):
    target = ["aten.round.default"]

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

        if len(node.args) > 1:
            warnings.warn(
                "[QNN Delegate Op Builder]: QNN dose not support decimals",
                stacklevel=1,
            )
            return None

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        round_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseRound.op_name,
        )
        round_op.AddInputTensors([input_tensor_wrapper])
        round_op.AddOutputTensors([output_tensor_wrapper])
        return round_op
