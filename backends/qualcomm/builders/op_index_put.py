import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch

from executorch.backends.qualcomm.utils.constants import QCOM_DATA, QCOM_QUANT_ATTRS
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP
from .node_visitor_manager import register_node_visitor
from .qnn_constants import (
    OpConcat,
    OpReshape,
    OpScatterNd,
    OpTile,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)


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
        op_wrapper_list = []
        input_node = self.get_node(node.args[0])
        # Because the args[0] of index_put op doesn't annotate, need to fill in the quant_attr with the node here.
        if quant_attrs := node.meta.get(QCOM_QUANT_ATTRS):
            quant_attrs = quant_attrs.copy()
            input_node.meta[QCOM_QUANT_ATTRS] = quant_attrs
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        indicies_node = node.args[1]
        index_node_dim = None
        index_nodes = []
        index_tensors = []
        target_index = []
        # If there is None in a list, it means all range at that dimension
        # E.g., indicies_node: [None, None, aten__to_copy_default_1]
        if isinstance(indicies_node, list):
            for index, idx_node in enumerate(indicies_node):
                # First, collect the indice_node and index of None to construct the shape of index node
                # E.g., shape of input: [1, 1024, 12, 64]
                # For "None" axis (assume indicies_node: [None, None, aten__to_copy_default_1]),
                # target_index: [1, 1024, x], x is the shape of index_tensor, index_node_dim: 2
                if isinstance(idx_node, torch.fx.Node):
                    index_nodes.append(idx_node)
                    index_tensors.append(self.get_tensor(idx_node, idx_node))
                    target_index.extend(index_tensors[-1].size())
                    index_node_dim = index
                elif idx_node is None and index_node_dim is None:
                    # E.g., indicies_node: [None, aten__to_copy_default_1, None]
                    # Don't need to consider "None" after index_node.
                    target_index.append(input_tensor.size(index))
                else:
                    warnings.warn(
                        f"[QNN Delegate Op Builder]: Get the index {idx_node} that is neither a node nor None",
                        stacklevel=1,
                    )
                    return
        # Assume that there is only one node in list
        assert len(index_nodes) == 1, "Not support multiple indices tensor"
        indice_node = index_nodes[0]
        indice_tensor = index_tensors[0]
        indices_tensor_wrapper = self.define_tensor(
            indice_node,
            node,
            indice_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # Need to reconstruct the index tensor.
        # E.g., based on ScatterND Op Def in QNN Docs.
        # Torch:
        #   Given that
        #     shape of input: [1, 12, 1024, 64]
        #     indicies_node: [None, None, aten__to_copy_default_1]
        #     shape of aten__to_copy_default_1: [1]
        # QNN:
        #   Index tensor:
        #     Shape: [1, 12, 1, 3]
        #     Value: [[[0,0,x]],[[0,1,x]],...,[[0,11,x]]]
        # The index tensor is treated as 4-dimensional tensor of 3-tuples,
        # where each 3-tuple is a partial-index into input
        # Reference code for QNN ScatterNd:
        #   output = np.copy(input)
        #   update_indices = indices.shape[:-1]
        #   for idx in np.ndindex(update_indices):
        #       output[indices[idx]] = updates[idx]

        # Append one dimension to specify x-tuple
        index_shape = target_index + [1]
        # Reshape the index_node for tile op
        reshape_shape = [
            shape if id == index_node_dim else 1 for id, shape in enumerate(index_shape)
        ]
        reshape_output_tensor = indice_tensor.reshape(reshape_shape)
        reshape_output_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_reshape",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=QNN_TENSOR_TYPE_MAP[reshape_output_tensor.dtype],
            quant_encoding=PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            quant_configs={},
            dims=reshape_output_tensor.size(),
            tensor=reshape_output_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        reshape_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        reshape_op.AddInputTensors([indices_tensor_wrapper])
        reshape_op.AddOutputTensors([reshape_output_tensor_wrapper])
        op_wrapper_list.append(reshape_op)
        index_put_index_input_tensor_wrapper = reshape_output_tensor_wrapper

        # Tile the index_node and concat the target index
        if None in indicies_node:
            tile_output_tensor = reshape_output_tensor.expand(index_shape)
            # Tile the index_node to align with the shape of target_index
            # Only need to tile the dim of None axis
            # E.g., indicies_node: [None, None, aten__to_copy_default_1]
            # Should tile the first two dimension.
            multiples = [
                shape if id != index_node_dim else 1
                for id, shape in enumerate(index_shape)
            ]
            multiples_shape = [len(index_shape)]
            tile_output_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_tile",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=QNN_TENSOR_TYPE_MAP[tile_output_tensor.dtype],
                quant_encoding=PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                quant_configs={},
                dims=tile_output_tensor.size(),
                tensor=tile_output_tensor,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            tile_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpTile.op_name,
            )
            tile_op.AddInputTensors([reshape_output_tensor_wrapper])
            tile_op.AddOutputTensors([tile_output_tensor_wrapper])
            tile_op.AddTensorParam(
                OpTile.param_multiples,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                len(multiples_shape),
                multiples_shape,
                np.array(multiples, dtype=np.uint32),
                True,
            )
            op_wrapper_list.append(tile_op)

            # Repeat index for "None" axis in indicies_node
            ranges = [
                torch.arange(dim, dtype=indice_tensor.dtype)
                for dim in target_index[:-1]
            ]
            target_index_shape = target_index + [len(ranges)]
            target_index_tensor = torch.cartesian_prod(*ranges)
            reshape_target_index_shape = [
                shape if id != index_node_dim else 1
                for id, shape in enumerate(target_index_shape)
            ]
            target_index_tensor = target_index_tensor.reshape(
                reshape_target_index_shape
            )
            target_index_tensor = target_index_tensor.expand(
                target_index_shape
            ).contiguous()
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
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )

            # Concat target_index and tile output to reconstruct index_node
            # Cannot use QNN Pack (stack) since QNN Pack is not support int32 dtype
            concat_output_tensor = torch.concat(
                (target_index_tensor, tile_output_tensor), dim=-1
            )
            concat_output_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_concat",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=QNN_TENSOR_TYPE_MAP[concat_output_tensor.dtype],
                quant_encoding=PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                quant_configs={},
                dims=concat_output_tensor.size(),
                tensor=concat_output_tensor,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            concat_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConcat.op_name,
            )
            concat_op.AddInputTensors(
                [target_index_tensor_wrapper, tile_output_tensor_wrapper]
            )
            concat_op.AddOutputTensors([concat_output_tensor_wrapper])
            concat_op.AddScalarParam(
                OpConcat.param_axis,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: np.uint32(concat_output_tensor.dim() - 1)},
            )
            op_wrapper_list.append(concat_op)
            index_put_index_input_tensor_wrapper = concat_output_tensor_wrapper

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
            [
                input_tensor_wrapper,
                index_put_index_input_tensor_wrapper,
                value_tensor_wrapper,
            ]
        )
        index_put_op.AddOutputTensors([output_tensor_wrapper])
        op_wrapper_list.append(index_put_op)
        return op_wrapper_list
