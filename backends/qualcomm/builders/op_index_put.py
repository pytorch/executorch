import warnings
from collections import OrderedDict
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch

from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_DTYPE,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor, QNN_QUANT_TYPE_MAP, QNN_TENSOR_TYPE_MAP
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

    def define_node(  # noqa: C901
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

        indices_nodes = (
            node.args[1] if isinstance(node.args[1], list) else [node.args[1]]
        )
        target_index = []
        all_range_index = OrderedDict()
        index_dtype = [
            node.meta["val"].dtype for node in indices_nodes if node is not None
        ][0]

        # preprocess:
        # - broadcast dimension for multiple specified index
        # - broadcast specified index if dimensions are not matched
        max_indices_in_specified_index = 0
        for index, idx_node in enumerate(indices_nodes):
            if isinstance(idx_node, torch.fx.Node):
                last_specified_index_node = index
                if max_indices_in_specified_index < idx_node.meta["val"].nelement():
                    max_indices_in_specified_index = idx_node.meta["val"].nelement()
        # If there is None in a list, it means all range at that dimension
        for index, idx_node in enumerate(indices_nodes):
            # First, collect the index_node and index of None to construct the shape of index node
            # E.g., shape of input: [1, 1024, 12, 64]
            # For "None" axis (assume indices_node: [None, None, aten__to_copy_default_1]),
            # target_index: [1, 1024, x], x is the shape of index_tensor, index_node_dim: 2
            if isinstance(idx_node, torch.fx.Node):
                # e.g. for case [index_node_0, None, index_node_1], nodes will have the same number of indices
                target_index.append(
                    self.get_tensor(idx_node, idx_node).nelement()
                    if last_specified_index_node == index
                    else 1
                )
            elif idx_node is None:
                # E.g., indices_node: [None, None, aten__to_copy_default_1]
                all_range_index[index] = torch.arange(
                    input_tensor.size(index), dtype=index_dtype
                )
                target_index.append(input_tensor.size(index))
            else:
                warnings.warn(
                    f"[QNN Delegate Op Builder]: Get the index {idx_node} that is neither a node nor None",
                    stacklevel=1,
                )
                return

        # preprocess all range indices if any
        if None in indices_nodes:
            all_range_tensor = torch.cartesian_prod(*all_range_index.values())
            # repeat all_range_tensor interleavely for future concatenation
            # e.g. input_node = [5, 4, 3, 2], indices = [index_0_node, None, index_2_node]
            #      index_0.shape == index_2.shape == 2 (will guarantee this condition)
            #      where user specified (3, 4) for index_0, (0, 1) for index_2
            # ---
            # we should have all_range_tensor: [0, 1, 2, 3]
            # repeat interleavely with 2 to match future tiled index_0_node & index_2_node
            # we'll have 1(index_0 -> same as index_2)*4(index_1)*2(index_2) indices in total:
            # | index_0_node | None | index_2_node |
            # | 3            | 0    | 0            |
            # | 4            | 0    | 1            |
            # | 3            | 1    | 0            |
            # | 4            | 1    | 1            |
            # | 3            | 2    | 0            |
            # | 4            | 2    | 1            |
            # | 3            | 3    | 0            |
            # | 4            | 3    | 1            |
            all_range_tensor_aug = all_range_tensor.repeat_interleave(
                max_indices_in_specified_index, dim=0
            )
            for index in all_range_index.keys():
                # Repeat index for "None" axis in indices_nodes
                range_index_node = torch.fx.Node(
                    node.graph,
                    node.name + f"_all_range_index_{index}",
                    "call_function",
                    exir_ops.edge.aten.tensor.default,
                    (),  # args
                    {},  # kwargs
                )
                range_indices = (
                    (
                        all_range_tensor_aug[:, index]
                        if all_range_tensor_aug.dim() > 1
                        else
                        # if there is only one None
                        all_range_tensor_aug
                    )
                    .reshape(-1, 1)
                    .contiguous()
                )
                target_index_tensor_wrapper = self.define_tensor(
                    range_index_node,
                    node,
                    range_indices,
                    PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                    nodes_to_wrappers,
                )
                # store it for future concatenation
                all_range_index[index] = (range_indices, target_index_tensor_wrapper)

        # Need to reconstruct the index tensor.
        # E.g., based on ScatterND Op Def in QNN Docs.
        # Torch:
        #   Given that
        #     shape of input: [1, 12, 1024, 64]
        #     indices_node: [None, None, aten__to_copy_default_1]
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
        specified_index = OrderedDict()
        for i, indices_node in enumerate(indices_nodes):
            if indices_node is None:
                continue

            indices_tensor = self.get_tensor(indices_node, indices_node)
            indices_tensor_wrapper = self.define_tensor(
                indices_node,
                node,
                indices_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            if indices_tensor.nelement() < max_indices_in_specified_index:
                # broadcast the specified index
                indices_tensor = indices_tensor.repeat(max_indices_in_specified_index)
                indices_multiples = [max_indices_in_specified_index]
                indices_multiples_shape = [len(indices_multiples)]
                indices_tile_tensor_wrapper = self.define_custom_tensor_wrapper(
                    node_name=node.name + f"_indices_tile_{i}",
                    tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    dtype=QNN_TENSOR_TYPE_MAP[indices_tensor.dtype],
                    quant_encoding=PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                    quant_configs={},
                    dims=indices_tensor.size(),
                    tensor=indices_tensor,
                    is_fake_tensor=True,
                    nodes_to_wrappers=nodes_to_wrappers,
                )
                tile_op = PyQnnWrapper.PyQnnOpWrapper(
                    node.name,
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    OpTile.op_name,
                )
                tile_op.AddInputTensors([indices_tensor_wrapper])
                tile_op.AddOutputTensors([indices_tile_tensor_wrapper])
                tile_op.AddTensorParam(
                    OpTile.param_multiples,
                    PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                    len(indices_multiples_shape),
                    indices_multiples_shape,
                    np.array(indices_multiples, dtype=np.uint32),
                    True,
                )
                op_wrapper_list.append(tile_op)
                indices_tensor_wrapper = indices_tile_tensor_wrapper

            # Append one dimension to specify x-tuple
            # Reshape the index_node for tile op
            reshape_shape = list(indices_tensor.shape) + [1]
            reshape_output_tensor = indices_tensor.reshape(reshape_shape)
            reshape_output_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + f"_reshape_{i}",
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
            index_tensor_wrapper = reshape_output_tensor_wrapper
            index_tensor = reshape_output_tensor

            # Tile the index_node and concat the target index
            if None in indices_nodes:
                tile_output_tensor = reshape_output_tensor.repeat(
                    all_range_tensor.size(0), 1
                )
                # Tile the index_node to align with the shape of target_index
                # Only need to tile the dim of None axis
                # E.g., indices_node: [None, None, aten__to_copy_default_1]
                # Should tile the number of indices combination of first two dimension
                # times number of indices specified by aten__to_copy_default_1
                multiples = [all_range_tensor.size(0), 1]
                multiples_shape = [len(multiples)]
                tile_output_tensor_wrapper = self.define_custom_tensor_wrapper(
                    node_name=node.name + f"_tile_{i}",
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
                index_tensor_wrapper = tile_output_tensor_wrapper
                index_tensor = tile_output_tensor

            specified_index[i] = (index_tensor, index_tensor_wrapper)

        # Concat target_index and tile output to reconstruct index_node
        # Cannot use QNN Pack (stack) since QNN Pack is not support int32 dtype
        index_tensors, index_tensor_wrappers = [], []
        for i, arg in enumerate(indices_nodes):
            tensor, tensor_wrapper = (
                all_range_index[i] if arg is None else specified_index[i]
            )
            index_tensors.append(tensor)
            index_tensor_wrappers.append(tensor_wrapper)

        if len(index_tensor_wrappers) > 1:
            concat_output_tensor = torch.concat(index_tensors, dim=-1)
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
            concat_op.AddInputTensors(index_tensor_wrappers)
            concat_op.AddOutputTensors([concat_output_tensor_wrapper])
            concat_op.AddScalarParam(
                OpConcat.param_axis,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: np.uint32(concat_output_tensor.dim() - 1)},
            )
            op_wrapper_list.append(concat_op)

        value_node = self.get_node(node.args[2])
        value_tensor = self.get_tensor(value_node, node)
        value_tensor_wrapper = self.define_tensor(
            value_node,
            node,
            value_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        # handle broadcast scenario
        # e.g. input_tensor: (1, 12, 1024, 64), value_tensor: (1, 64)
        #      => value_reshape_tensor: (1, 1, 1, 64)
        new_value_shape = (
            *([1] * (input_tensor.dim() - value_tensor.dim())),
            *value_tensor.shape,
        )
        # reshape the value_node for tile op
        value_quant_encoding, value_quant_configs = self.get_quant_encoding_conf(
            value_node, node
        )
        value_dtype = (
            QNN_TENSOR_TYPE_MAP[value_tensor.dtype]
            if value_quant_encoding
            == PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED
            else QNN_QUANT_TYPE_MAP[
                (
                    torch.uint16
                    if value_quant_configs[QCOM_DTYPE] == torch.int32
                    else value_quant_configs[QCOM_DTYPE]
                )
            ]
        )
        value_reshape_tensor = value_tensor.reshape(new_value_shape)
        value_reshape_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_value_reshape",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=value_dtype,
            quant_encoding=value_quant_encoding,
            quant_configs=value_quant_configs,
            dims=value_reshape_tensor.size(),
            tensor=value_reshape_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        value_reshape_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        value_reshape_op.AddInputTensors([value_tensor_wrapper])
        value_reshape_op.AddOutputTensors([value_reshape_tensor_wrapper])
        op_wrapper_list.append(value_reshape_op)

        # e.g. input_tensor: (1, 12, 1024, 64), index_tensor: (None, None, 2), value_tensor: (1, 64)
        #      => multiples: [1, 12, 2, 1]
        value_multiples = []
        for i in range(input_tensor.dim() - 1, -1, -1):
            if i in specified_index:
                # all user specified index node wil have the same dimension
                multiplier = (
                    indices_nodes[i].meta["val"].nelement() // new_value_shape[i]
                    if i == last_specified_index_node
                    else 1
                )
            else:
                multiplier = input_tensor.shape[i] // new_value_shape[i]
            value_multiples.insert(0, multiplier)

        value_tile_tensor = value_reshape_tensor.repeat(value_multiples)
        value_multiples_shape = [len(value_multiples)]
        value_tile_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_value_tile",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=value_dtype,
            quant_encoding=value_quant_encoding,
            quant_configs=value_quant_configs,
            dims=value_tile_tensor.size(),
            tensor=value_tile_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        value_tile_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTile.op_name,
        )
        value_tile_op.AddInputTensors([value_reshape_tensor_wrapper])
        value_tile_op.AddOutputTensors([value_tile_tensor_wrapper])
        value_tile_op.AddTensorParam(
            OpTile.param_multiples,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(value_multiples_shape),
            value_multiples_shape,
            np.array(value_multiples, dtype=np.uint32),
            True,
        )
        op_wrapper_list.append(value_tile_op)

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
        # accumulation
        if len(node.args) > 3 and node.args[3]:
            index_put_op.AddScalarParam(
                OpScatterNd.param_reduction,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
                {QCOM_DATA: 1},
            )

        # check final index_input tensor
        index_input_tensor, index_input_tensor_wrapper = (
            (concat_output_tensor, concat_output_tensor_wrapper)
            if len(index_tensor_wrappers) > 1
            else specified_index[last_specified_index_node]
        )
        target_index_reshape_tensor = index_input_tensor.reshape((*target_index, -1))
        target_index_reshape_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_target_index_reshape",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=QNN_TENSOR_TYPE_MAP[target_index_reshape_tensor.dtype],
            quant_encoding=PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            quant_configs={},
            dims=target_index_reshape_tensor.size(),
            tensor=target_index_reshape_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        target_index_reshape_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReshape.op_name,
        )
        target_index_reshape_op.AddInputTensors([index_input_tensor_wrapper])
        target_index_reshape_op.AddOutputTensors([target_index_reshape_tensor_wrapper])
        op_wrapper_list.append(target_index_reshape_op)

        index_put_op.AddInputTensors(
            [
                input_tensor_wrapper,
                target_index_reshape_tensor_wrapper,
                value_tile_tensor_wrapper,
            ]
        )
        index_put_op.AddOutputTensors([output_tensor_wrapper])
        op_wrapper_list.append(index_put_op)
        return op_wrapper_list
