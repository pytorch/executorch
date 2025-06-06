from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpScatterNd, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class IndexPutVisitor(NodeVisitor):
    target = ["aten.index_put.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        indicies_node = node.args[1]
        indices_list = [
            self.get_tensor(idx, idx) for idx in indicies_node if idx is not None
        ]

        # Unpack the tuple
        indices_unpacked = [torch.flatten(idx) for idx in indices_list]

        # Convert to 2-D tensor
        indices_qnn = torch.cat(indices_unpacked).unsqueeze(0)
        indice_node = [n for n in indicies_node if isinstance(n, torch.fx.Node)]
        # TODO consider to write a pass to combine to one input tensor for indices
        assert len(indice_node) == 1, "Not support multiple indices tensor"

        indices_tensor_wrapper = self.define_tensor(
            indice_node[0],
            node,
            indices_qnn,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        value_node = self.get_node(node.args[2])

        value_tensor = self.get_tensor(value_node, node)

        value_tensor_wrapper = self.define_tensor(
            value_node,
            node,
            value_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
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

        index_put_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterNd.op_name,
        )
        index_put_op.AddInputTensors(
            [input_tensor_wrapper, indices_tensor_wrapper, value_tensor_wrapper]
        )
        index_put_op.AddOutputTensors([output_tensor_wrapper])

        return index_put_op
