# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.exir.dialects._ops import ops as exir_ops

QNN_TENSOR_TYPE_MAP = {
    torch.float32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
    torch.int8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
    torch.int16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16,
    torch.int32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32,
    # Note that there is no int64 tensor data type in Qnn.
    torch.int64: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UNDEFINED,
    torch.uint8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8,
    float: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
}
QNN_SCALAR_TYPE_MAP = {
    torch.float32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
    torch.int8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_8,
    torch.int16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_16,
    torch.int32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
    torch.int64: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_64,
    torch.uint8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_8,
    float: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
}

PER_CHANNEL_ENCODING_MAPPING = {
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default: PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
}

PER_TENSOR_ENCODING_MAPPING = {
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor: PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor: PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
}


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph
    """

    def __init__(self, external_ids, compile_mode) -> None:
        self.external_ids = external_ids or {}
        self.compile_mode = compile_mode

    def is_graph_input(self, tensor: torch.fx.Node) -> bool:
        """
        Check if the given tensor is a graph input

        Args:
            tensor: EdgeIR Tensor that is being checked for graph input
        """
        return tensor.op == "placeholder"

    def is_graph_output(self, tensor: torch.fx.Node) -> bool:
        """
        Check if the given tensor is used as a graph output

        Args:
            tensor: EdgeIR Tensor that is being checked for graph input
        """

        for user in tensor.users.keys():
            if user.op == "output":
                return True
        return False

    def get_tensor_shape(self, input_node, op_node, idx=None):
        """
        Get tensor shape with axis_order
        """

        def get_tensor(node, index):
            if index is not None:
                assert isinstance(index, int)
                return node.meta["val"][index]
            return node.meta["val"]

        def get_shape_c(index):
            tensor, use_memo = get_tensor(input_node, index), True
            if "axis_order" in op_node.meta:
                tensor = tensor.permute(dims=op_node.meta["axis_order"]).contiguous()
            return tensor, use_memo

        def get_shape_p(index):
            tensor, use_memo = get_tensor(input_node, index), True
            res = ["axis_order" in meta for meta in [input_node.meta, op_node.meta]]
            if res == [False, True]:
                tensor = tensor.permute(dims=op_node.meta["axis_order"]).contiguous()
                use_memo = False
            elif res == [True, False]:
                use_memo = False
            elif res == [True, True]:
                tensor = tensor.permute(dims=op_node.meta["axis_order"]).contiguous()
            return tensor, use_memo

        tensor, use_memo = get_tensor(input_node, idx), True
        if len(tensor.shape) == 0:
            return tensor, use_memo

        return get_shape_c(idx) if self.compile_mode else get_shape_p(idx)

    def get_quant_encoding_conf(self, node: torch.fx.Node) -> Tuple[Any, Dict]:
        if not node.meta.get("quant_attrs", None):
            return (
                PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                {},
            )

        quant_attrs = node.meta["quant_attrs"]
        encoding = quant_attrs["encoding"]
        quant_config = {}
        if encoding in PER_CHANNEL_ENCODING_MAPPING:
            scales = quant_attrs["scales"]
            zero_points = quant_attrs["zero_points"]
            assert len(scales) == len(
                zero_points
            ), f"Per channel encoding of node {node}, has differnt size fo scales {len(scales)} and zero_points {len(zero_points)}"

            scale_offset = []
            for i in range(len(scales)):
                scale_offset.append(
                    PyQnnWrapper.Qnn_ScaleOffset_t(scales[i], -zero_points[i])
                )

            user_0 = list(node.users)[0]
            # Memory layout of QNN conv is NHW"C", need to set axis as 3
            if (
                type(user_0.target) != str
                and user_0.target.__name__ in ["aten.convolution.default"]
                and list(node.users)[0].args[1] == node
            ):
                quant_config["axis"] = 3
            else:
                quant_config["axis"] = quant_attrs["axis"]

            quant_config["scale_offset"] = scale_offset
            quant_config["dtype"] = quant_attrs["dtype"]
            return PER_CHANNEL_ENCODING_MAPPING[encoding], quant_config

        # per tensor situation
        quant_config["scale"] = quant_attrs["scale"]
        # check Qnn_ScaleOffset_t in QNN/include/QnnTypes.h
        quant_config["offset"] = -quant_attrs["zero_point"]
        quant_config["dtype"] = quant_attrs["dtype"]
        return PER_TENSOR_ENCODING_MAPPING[encoding], quant_config

    def get_quant_tensor_value(
        self, node: torch.fx.Node, tensor: torch.Tensor
    ) -> torch.Tensor:
        quant_attrs = node.meta["quant_attrs"]
        encoding = quant_attrs["encoding"]
        if encoding in PER_CHANNEL_ENCODING_MAPPING:
            scales = quant_attrs["scales"]
            offsets = quant_attrs["zero_points"]
            return tensor.div(scales).add(offsets).round().to(quant_attrs["dtype"])

        # per tensor situation
        scale = quant_attrs["scale"]
        offset = quant_attrs["zero_point"]
        return tensor.div(scale).add(offset).round().to(quant_attrs["dtype"])

    def get_tensor_type(
        self,
        node: torch.fx.Node,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
    ) -> PyQnnWrapper.Qnn_TensorType_t:
        is_input = self.is_graph_input(node)
        is_output = self.is_graph_output(node)
        # handle logic for input/output tensors
        if is_input or is_output:
            assert (
                node in self.external_ids.keys()
            ), f"Node {node}, is_input: {is_input}, is_output: {is_output}, ext_ids: {self.external_ids.keys()}"
            if is_input:
                return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE
            if is_output:
                return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ
        return tensor_type

    def define_scalar(
        self,
        node: torch.fx.Node,
        tensor: torch.Tensor,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.TensorWrapper:
        if node in nodes_to_wrappers:
            return nodes_to_wrappers[node]
        dims = tensor.size()
        tensor_type = self.get_tensor_type(node, tensor_type)
        quant_encoding, quant_configs = self.get_quant_encoding_conf(node)
        dtype = tensor.dtype if not quant_configs else quant_configs["dtype"]

        if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                node.name,
                tensor_type,
                QNN_SCALAR_TYPE_MAP[dtype],
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                np.array([]),
                False,
            )
        else:
            if quant_configs:
                tensor = self.get_quant_tensor_value(node, tensor)
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                node.name,
                tensor_type,
                QNN_SCALAR_TYPE_MAP[dtype],
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                tensor.detach().numpy(),
                True,
            )
        nodes_to_wrappers[node] = tensor_wrapper
        return tensor_wrapper

    def define_tensor(
        self,
        node: torch.fx.Node,
        tensor: torch.Tensor,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.TensorWrapper:
        """
        Covert torch.Tensor to TensorWrapper

        Args:
            tensor: EdgeIR Tensor
            nodes_to_wrappers: Set contains edge_graph values(node targets)
        """

        if node in nodes_to_wrappers:
            return nodes_to_wrappers[node]
        # For constant input, the size of tensor is torch.Size([])
        dims = [1] if len(tensor.size()) == 0 else tensor.size()
        tensor_type = self.get_tensor_type(node, tensor_type)
        quant_encoding, quant_configs = self.get_quant_encoding_conf(node)
        dtype = tensor.dtype if not quant_configs else quant_configs["dtype"]

        if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                node.name,
                tensor_type,
                QNN_TENSOR_TYPE_MAP[dtype],
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                np.array([]),
                False,
            )
        else:
            if quant_configs:
                tensor = self.get_quant_tensor_value(node, tensor)
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                node.name,
                tensor_type,
                QNN_TENSOR_TYPE_MAP[dtype],
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                tensor.detach().numpy(),
                True,
            )
        nodes_to_wrappers[node] = tensor_wrapper
        return tensor_wrapper

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        """Convert torch.fx.Node to OpWrapper"""
        raise NotImplementedError("NodeVisitor must be extended!")


# This will hold mapping of all node names to the visitor class
_node_visitor_dict = {}


def register_node_visitor(visitor):
    """Register node visitor into _node_visitor_dict"""
    assert (
        isinstance(visitor, type)
        and issubclass(visitor, NodeVisitor)
        and hasattr(visitor, "target")
    ), f"Illformed NodeVisitor subclass, can't register!, got: {visitor}"
    _node_visitor_dict[visitor.target] = visitor


def generate_node_to_external_map(
    edge_graph_module: torch.fx.GraphModule,
) -> Dict[torch.fx.Node, int]:
    node_to_external_map = {}
    for node in edge_graph_module.graph.nodes:
        # The order in which we visit the placeholder node is same as the *args
        # order for the forward(*args) signature for this gm. Using the order of
        # the nodes as external_id to extract the right arg from *args at runtime
        if node.op == "placeholder":
            node_to_external_map[node] = len(node_to_external_map)
    for node in edge_graph_module.graph.nodes:
        if node.op == "output":
            for output_nodes in node.args:
                for output_node in output_nodes:
                    node_to_external_map[output_node] = len(node_to_external_map)
    return node_to_external_map


def get_node_visitors(
    edge_graph_module: torch.fx.GraphModule,
    compile_mode=True,
) -> Dict[str, NodeVisitor]:
    """Create a new class instance at runtime, and put them in a dict"""
    node_to_external_map = generate_node_to_external_map(edge_graph_module)
    node_visitors = {}
    for target, visitor in _node_visitor_dict.items():
        assert callable(
            visitor
        ), f"Expeting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(node_to_external_map, compile_mode)
    return node_visitors
