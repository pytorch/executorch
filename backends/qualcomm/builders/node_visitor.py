# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Tuple

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS,
    QCOM_AXIS_ORDER,
    QCOM_BITWIDTH,
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_OFFSET,
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_REQUANTIZE,
    QCOM_SCALE,
    QCOM_SCALE_OFFSET,
    QCOM_SCALES,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)

from executorch.exir.dialects._ops import ops as exir_ops

from .utils import (
    deduce_dtype,
    get_parameter,
    is_graph_input,
    is_graph_output,
    is_parameter,
)


QNN_QUANT_TYPE_MAP = {
    torch.int8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
    torch.int16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16,
    torch.int32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32,
    # Note that there is no int64 tensor data type in Qnn.
    torch.int64: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UNDEFINED,
    torch.uint8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8,
    torch.uint16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16,
}
QNN_TENSOR_TYPE_MAP = {
    torch.bool: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
    torch.float32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
    torch.int8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_8,
    torch.int16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_16,
    torch.int32: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
    torch.int64: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_64,
    torch.uint8: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_8,
    torch.uint16: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_16,
    float: PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
}

PER_CHANNEL_ENCODING = {
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
}

PER_TENSOR_ENCODING = {
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
}


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph
    """

    def __init__(
        self,
        external_ids,
        edge_program: torch.export.ExportedProgram,
        enable_tensor_dump,
    ) -> None:
        self.external_ids = external_ids or {}
        self.edge_program = edge_program
        self.enable_tensor_dump = enable_tensor_dump

    def get_tensor(self, input_node, op_node, idx=None):
        """
        Get tensor value/shape with axis_order
        """

        def _get_tensor(node, index):
            if index is not None:
                assert isinstance(index, int)
                if is_parameter(node, self.edge_program):
                    return get_parameter(node, self.edge_program)[index]
                return node.meta["val"][index]

            if is_parameter(node, self.edge_program):
                return get_parameter(node, self.edge_program)
            return node.meta["val"]

        tensor = _get_tensor(input_node, idx)
        if len(tensor.shape) != 0 and QCOM_AXIS_ORDER in op_node.meta:
            tensor = tensor.permute(dims=op_node.meta[QCOM_AXIS_ORDER]).contiguous()
        return tensor

    def make_qnn_per_channel_config(self, node: torch.fx.Node, quant_attrs: Dict):
        quant_config = copy.deepcopy(quant_attrs)

        scales = quant_attrs[QCOM_SCALES]
        zero_points = quant_attrs[QCOM_ZERO_POINTS]
        assert len(scales) == len(
            zero_points
        ), f"Per channel encoding of node {node}, has different size for scales {len(scales)} and zero_points {len(zero_points)}"

        scale_offset = []
        for i in range(len(scales)):
            # check Qnn_ScaleOffset_t in QNN/include/QnnTypes.h
            scale_offset.append(
                PyQnnWrapper.Qnn_ScaleOffset_t(scales[i], -zero_points[i])
            )

        user_0 = list(node.users)[0]
        # Memory layout of QNN conv weight always ends in Output. Like conv2d is HWIO
        if (
            "convolution" in user_0.target.__name__
            and list(node.users)[0].args[1] == node
        ):
            quant_config[QCOM_AXIS] = 3

        else:
            quant_config[QCOM_AXIS] = quant_attrs[QCOM_AXIS]

        quant_config[QCOM_SCALE_OFFSET] = scale_offset
        # special case for 4 bits
        if (
            quant_config[QCOM_DTYPE] == torch.int8
            and quant_config[QCOM_QUANT_MAX] - quant_config[QCOM_QUANT_MIN] <= 15
        ):
            quant_config[QCOM_BITWIDTH] = 4
            return (
                PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET,
                quant_config,
            )
        return (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
            quant_config,
        )

    def make_qnn_per_tensor_config(self, quant_attrs: Dict):
        quant_config = copy.deepcopy(quant_attrs)
        # check Qnn_ScaleOffset_t in QNN/include/QnnTypes.h
        quant_config[QCOM_OFFSET] = -quant_attrs[QCOM_ZERO_POINT]
        # special case for 4 bits
        if (
            quant_config[QCOM_DTYPE] == torch.int8
            and quant_config[QCOM_QUANT_MAX] - quant_config[QCOM_QUANT_MIN] <= 15
        ):
            quant_config[QCOM_BITWIDTH] = 4
            return (
                PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET,
                quant_config,
            )
        return (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
            quant_config,
        )

    def get_quant_encoding_conf(
        self, node: torch.fx.Node, is_input_tensor: bool = False
    ) -> Tuple[Any, Dict]:
        if not node.meta.get(QCOM_QUANT_ATTRS, None):
            return (
                PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                {},
            )
        quant_attrs = (
            node.meta[QCOM_REQUANTIZE]
            if QCOM_REQUANTIZE in node.meta and is_input_tensor
            else node.meta[QCOM_QUANT_ATTRS]
        )
        if quant_attrs[QCOM_ENCODING] in PER_CHANNEL_ENCODING:
            return self.make_qnn_per_channel_config(node, quant_attrs)

        return self.make_qnn_per_tensor_config(quant_attrs)

    def get_quant_tensor_value(
        self, tensor: torch.Tensor, quant_attrs: Dict, quant_configs: Dict
    ) -> torch.Tensor:
        if quant_attrs[QCOM_ENCODING] in PER_TENSOR_ENCODING:
            scale = quant_attrs[QCOM_SCALE]
            zero_point = quant_attrs[QCOM_ZERO_POINT]
        else:  # per channel case
            scale = quant_attrs[QCOM_SCALES]
            zero_point = quant_attrs[QCOM_ZERO_POINTS]

        dtype = quant_configs[QCOM_DTYPE]

        tensor = tensor.div(scale).add(zero_point).round().to(dtype)
        # Make the backends access data correctly
        if quant_configs.get(QCOM_BITWIDTH) == 4:
            mask = torch.full(tensor.size(), 0x0F, dtype=torch.int8)
            tensor = torch.bitwise_and(mask, tensor)
        return tensor

    def get_tensor_type(
        self,
        node: torch.fx.Node,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
    ) -> PyQnnWrapper.Qnn_TensorType_t:
        is_input = is_graph_input(node, self.edge_program)
        is_output = is_graph_output(node)
        # handle logic for input/output tensors
        if is_input or is_output:
            assert (
                node in self.external_ids
            ), f"Node {node}, is_input: {is_input}, is_output: {is_output}, ext_ids: {self.external_ids.keys()}"
            if is_input:
                return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_WRITE
            if is_output:
                return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ

        if is_parameter(node, self.edge_program):
            return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC

        # dump all tensor, set to app read
        if self.enable_tensor_dump:
            return PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_APP_READ
        return tensor_type

    def get_data_type(
        self,
        tensor: torch.Tensor,
        quant_config: Dict,
    ) -> PyQnnWrapper.Qnn_TensorType_t:
        if quant_config:
            quant_config[QCOM_DTYPE] = deduce_dtype(tensor, quant_config)
            return QNN_QUANT_TYPE_MAP[quant_config[QCOM_DTYPE]]

        return QNN_TENSOR_TYPE_MAP[tensor.dtype]

    def define_custom_tensor_wrapper(
        self,
        node_name: str,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
        dtype: PyQnnWrapper.Qnn_DataType_t,
        quant_encoding: PyQnnWrapper.Qnn_QuantizationEncoding_t,
        quant_configs: dict,
        dims: torch.Size,
        tensor: torch.Tensor,
        is_fake_tensor: bool,
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnWrapper.TensorWrapper]],
        wrapper_idx: int = 0,
    ) -> PyQnnWrapper.TensorWrapper:
        if cached := nodes_to_wrappers[node_name].get(wrapper_idx, None):
            return cached
        if is_fake_tensor:
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                node_name,
                tensor_type,
                dtype,
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                np.array([]),
                False,
            )
        else:
            # Can implement non-fake tensor when there is a need
            return None
        nodes_to_wrappers[node_name][wrapper_idx] = tensor_wrapper
        return tensor_wrapper

    def define_tensor(
        self,
        node: torch.fx.Node,
        tensor: torch.Tensor,
        tensor_type: PyQnnWrapper.Qnn_TensorType_t,
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnWrapper.TensorWrapper]],
        is_input_tensor: bool,
        node_name: str = None,
        wrapper_idx: int = 0,
    ) -> PyQnnWrapper.TensorWrapper:
        """
        Covert torch.Tensor to TensorWrapper

        Args:
            node: EdgeIR Node
            tensor: EdgeIR Tensor
            tensor_type: QNN tensor type
            nodes_to_wrappers: Set contains edge_graph values(node targets)
            is_input_tensor: Whether tensor is a fake input tensor relatively to
                             the op builder that is calling this function
        """
        if node_name is None:
            node_name = node.name

        if cached := nodes_to_wrappers[node_name].get(wrapper_idx, None):
            return cached

        tensor_name = f"{node.name}_{wrapper_idx}"
        if is_graph_input(node, self.edge_program):
            tensor_name = "input_" + str(self.external_ids[node]) + "_" + tensor_name
        if is_graph_output(node):
            tensor_name = "output_" + tensor_name
        dims = [1] if len(tensor.size()) == 0 else tensor.size()
        tensor_type = self.get_tensor_type(node, tensor_type)
        quant_encoding, quant_configs = self.get_quant_encoding_conf(
            node, is_input_tensor
        )
        dtype = self.get_data_type(tensor, quant_configs)
        if isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor):
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                tensor_name,
                tensor_type,
                dtype,
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                np.array([]),
                False,
            )
        else:
            if quant_configs:
                tensor = self.get_quant_tensor_value(
                    tensor,
                    node.meta[QCOM_QUANT_ATTRS],
                    quant_configs,
                )
            tensor_wrapper = PyQnnWrapper.TensorWrapper(
                tensor_name,
                tensor_type,
                dtype,
                quant_encoding,
                quant_configs,
                len(dims),
                dims,
                tensor.detach().numpy(),
                True,
            )
        nodes_to_wrappers[node_name][wrapper_idx] = tensor_wrapper
        return tensor_wrapper

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnWrapper.TensorWrapper]],
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
    for target in visitor.target:
        _node_visitor_dict[target] = visitor


def generate_node_to_external_map(
    edge_program: torch.export.ExportedProgram,
) -> Dict[torch.fx.Node, int]:
    node_to_external_map = {}
    for node in edge_program.graph_module.graph.nodes:
        # The order in which we visit the placeholder node is same as the *args
        # order for the forward(*args) signature for this gm. Using the order of
        # the nodes as external_id to extract the right arg from *args at runtime
        if is_graph_input(node, edge_program):
            node_to_external_map[node] = len(node_to_external_map)
    for node in edge_program.graph_module.graph.nodes:
        if is_graph_output(node):
            node_to_external_map[node] = len(node_to_external_map)
    return node_to_external_map


def get_node_visitors(
    edge_program: torch.export.ExportedProgram,
    enable_tensor_dump=False,
) -> Dict[str, NodeVisitor]:
    """Create a new class instance at runtime, and put them in a dict"""
    node_to_external_map = generate_node_to_external_map(edge_program)
    node_visitors = {}
    for target, visitor in _node_visitor_dict.items():
        assert callable(
            visitor
        ), f"Expeting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(
            node_to_external_map, edge_program, enable_tensor_dump
        )
    return node_visitors
