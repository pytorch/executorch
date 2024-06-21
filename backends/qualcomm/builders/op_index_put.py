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
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        indicies_node = node.args[1]

        print("indicies_node: ", indicies_node)
        indices_list = [self.get_tensor(idx, idx) for idx in indicies_node if idx is not None]

        # Unpack the tuple
        indices_unpacked = [torch.flatten(idx) for idx in indices_list]

        # Convert to 2-D tensor
        indices_qnn = torch.cat(indices_unpacked).unsqueeze(0)
        print("indices_qnn: ", indices_qnn, " indices_qnn.shape: ", indices_qnn.shape)
        indice_node = None
        for candidate_index_node in indicies_node:
            if candidate_index_node is not None and isinstance(candidate_index_node, torch.fx.Node):
                indice_node = candidate_index_node
                break
        indices_tensor_wrapper = self.define_tensor(
            indice_node,
            indices_qnn,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        value_node = node.args[2]

        value_tensor = self.get_tensor(value_node, node)

        value_tensor_wrapper = self.define_tensor(
            value_node,
            value_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        index_put_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterNd.op_name,
        )
        # axis = 0
        # if len(node.args) == 2:
        #     axis = cast(int, node.args[1])

        # if axis < 0:
        #     axis += node.meta["val"].dim()
        index_put_op.AddInputTensors([input_tensor_wrapper, indices_tensor_wrapper, value_tensor_wrapper])
        index_put_op.AddOutputTensors([output_tensor_wrapper])
        # index_put_op.AddScalarParam(
        #     OpScatterElements.param_axis,
        #     PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
        #     {"data": np.uint32(axis)},
        # )

        return index_put_op

