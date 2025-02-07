#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import ctypes
import logging

from typing import Dict, List, Tuple, Union

import torch

from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    Buffer,
    MPSCast,
    MPSDataType,
    MPSGraph,
    MPSNode,
    MPSNodeUnion,
    MPSTensor,
)

from executorch.backends.apple.mps.utils.mps_utils import (
    edge_dtype_to_mps_dtype,
    get_input_node,
    get_param_tensor,
    get_scalar_val,
    is_parameter,
)

from executorch.backends.transforms import get_shape
from executorch.exir.sym_util import eval_shape

from torch.export.exported_program import ExportedProgram


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph and
    serializing them using the mps serialization schema.
    """

    _tensor_to_id: Dict[torch.fx.Node, int] = {}
    _convert_model_to_fp16: bool = True

    def __init__(
        self, exported_program: ExportedProgram, convert_model_to_fp16: bool = True
    ):
        self._exported_program = exported_program
        self._convert_model_to_fp16 = convert_model_to_fp16

    @property
    def tensor_to_id(self) -> Dict[torch.fx.Node, int]:
        return self._tensor_to_id

    @property
    def convert_model_to_fp16(self) -> bool:
        return self._convert_model_to_fp16

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        raise NotImplementedError("NodeVisitor must be extended!")

    def define_tensor(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
        mps_data_type: MPSDataType = None,
    ) -> int:
        """Defines a tensor value into the MPSGraph serialization schema

        Args:
            node (torch.fx.Node): EdgeIR tensor to define into mps_graph
            mps_graph (MPSGraph): MPSGraph object for serializing into flatbuffer
        """

        if node is None:
            return -1

        if node in self.tensor_to_id:
            return self.tensor_to_id[node]

        # Get a unique id for the node.
        id = self.get_serialized_id(node, mps_graph)
        cb_size, constant_buffer, mps_data_type = self.get_serialized_buffer(
            node, mps_graph, id, mps_data_type
        )
        dims = get_shape(node)

        logging.debug(
            f"Serializing: {node}, data type: {node.meta['val'].dtype}, dims: {dims}"
        )
        mps_tensor = MPSTensor(
            datatype=mps_data_type,
            num_dims=len(dims),
            dims=dims,
            constant_buffer_size=cb_size,
            constant_buffer=constant_buffer,
        )

        mps_graph.mps_values.append(mps_tensor)
        return id

    def define_tensor_list(self, node: torch.fx.Node, mps_graph: MPSGraph) -> List[int]:
        """_summary_

        Args:
            node (torch.fx.Node): _description_
            mps_graph (MPSGraph): _description_
        """
        if node is None:
            return -1

        if node in self.tensor_to_id:
            return self.tensor_to_id[node]

        self.tensor_to_id[node] = []
        for i in range(len(node.meta["val"])):
            id = len(mps_graph.mps_values)
            self.tensor_to_id[node].append(id)

            tensor = node.meta["val"][i]
            dims = eval_shape(tensor.shape)
            mps_data_type = edge_dtype_to_mps_dtype(tensor.dtype)
            logging.debug(
                f"Serializing: [{i}]: {node}, data type: {tensor.dtype}, dims: {dims}"
            )

            mps_tensor = MPSTensor(
                datatype=mps_data_type,
                num_dims=len(dims),
                dims=dims,
                constant_buffer_size=0,
                constant_buffer=Buffer(storage=b""),
            )
            logging.debug(f"  Serialized tensor: {mps_tensor}")
            mps_graph.mps_values.append(mps_tensor)
        return self.tensor_to_id[node]

    def hash_tensor(self, tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))

    def define_constant(
        self,
        constant_tensor: torch.tensor,
        mps_graph: MPSGraph,
    ):
        """Defines a scalar value into the MPSGraph serialization schema

        Args:
            constant_tensor (torch.fx.Node): EdgeIR tensor to define into mps_graph
            mps_graph (MPSGraph): MPSGraph object for serializing into flatbuffer
        """
        constant_tensor = constant_tensor.contiguous()
        hash = self.hash_tensor(constant_tensor)
        if hash in self.tensor_to_id:
            return self.tensor_to_id[hash]

        id = self.get_serialized_id(constant_tensor, mps_graph, hash)

        mps_data_type = edge_dtype_to_mps_dtype(constant_tensor.dtype)
        constant_buffer_size, constant_buffer, mps_data_type = self.get_serialized_data(
            constant_tensor, mps_graph, mps_data_type, id
        )
        dims = list(constant_tensor.shape)

        mps_tensor = MPSTensor(
            datatype=mps_data_type,
            num_dims=len(dims),
            dims=dims,
            constant_buffer_size=constant_buffer_size,
            constant_buffer=constant_buffer,
        )

        mps_graph.mps_values.append(mps_tensor)
        return id

    def define_scalar(
        self,
        val: Union[float, int],
        mps_data_type: MPSDataType,
        mps_graph: MPSGraph,
    ):
        """Defines a scalar value into the MPSGraph serialization schema

        Args:
            mps_graph (MPSGraph): MPSGraph object for serializing into flatbuffer
        """
        assert isinstance(val, int) or isinstance(val, float)

        if val in self.tensor_to_id:
            return self.tensor_to_id[val]

        id = self.get_serialized_id(val, mps_graph, val)

        tensor = torch.tensor(val)
        constant_buffer_size, constant_buffer, mps_data_type = self.get_serialized_data(
            tensor, mps_graph, mps_data_type, id
        )

        mps_tensor = MPSTensor(
            datatype=mps_data_type,
            num_dims=1,
            dims=[1],
            constant_buffer_size=constant_buffer_size,
            constant_buffer=constant_buffer,
        )

        mps_graph.mps_values.append(mps_tensor)
        return id

    def get_serialized_buffer(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
        node_id: int,
        mps_data_type: MPSDataType = None,
    ) -> Tuple[int, Buffer, MPSDataType]:
        """
        If tensor holds some constant data, serialize it and return the
        index of its placement in the constant buffer

        Args:
            node (torch.fx.Node): _description_
            mps_graph (MPSGraph): _description_

        Returns:
            _type_: _description_
        """
        mps_data_type = (
            self.get_serialized_dtype(node) if mps_data_type is None else mps_data_type
        )

        # Check if this node is a lifted parameter
        if not is_parameter(self.exported_program, node):
            return 0, Buffer(storage=b""), mps_data_type

        tensor = get_param_tensor(self.exported_program, node)
        assert tensor is not None and isinstance(tensor, torch.Tensor)
        tensor = tensor.contiguous()

        return self.get_serialized_data(tensor, mps_graph, mps_data_type, node_id)

    def get_serialized_data(
        self,
        tensor: torch.tensor,
        mps_graph: MPSGraph,
        mps_data_type: MPSDataType,
        id: int,
    ) -> Tuple[int, Buffer, MPSDataType]:
        if (
            self.convert_model_to_fp16
            and mps_data_type == MPSDataType.mps_data_type_float32
        ):
            tensor = tensor.half()
            mps_data_type = MPSDataType.mps_data_type_float16

        if id not in mps_graph.constant_ids:
            mps_graph.constant_ids.append(id)

        if (
            mps_data_type is MPSDataType.mps_data_type_int4
            and tensor.dtype is torch.int8
        ):
            if tensor.dim() != 2:
                raise RuntimeError(f"Unexpected tensor shape {tensor.shape}")

            tensor = tensor.to(dtype=torch.int32)
            tensor = (((tensor[::, ::2] & 0x0F) << 4) | (tensor[::, 1::2] & 0x0F)).to(
                torch.uint8
            )
            tensor = (
                torch._convert_weight_to_int4pack(tensor.to("mps"), 2)
                .cpu()
                .view(dtype=torch.uint8)
            )
        array_type = ctypes.c_char * tensor.untyped_storage().nbytes()
        array = ctypes.cast(
            tensor.untyped_storage().data_ptr(),
            ctypes.POINTER(array_type),
        ).contents
        buffer = Buffer(storage=bytes(array))

        return tensor.untyped_storage().nbytes(), buffer, mps_data_type

    def get_serialized_id(
        self, node: Union[torch.fx.Node, float, int], mps_graph: MPSGraph, hash=None
    ) -> int:
        """
        Map a tensor to a unique id. If the tensor was already mapped, return
        the existent id.

        Args:
            node (Union[torch.fx.Node, float]): _description_
            mps_graph (MPSGraph): _description_

        Returns:
            int: _description_
        """
        if hash is not None and hash in self.tensor_to_id:
            return self.tensor_to_id[hash]
        elif node in self.tensor_to_id:
            return self.tensor_to_id[node]

        id = len(mps_graph.mps_values)
        if hash is not None:
            self.tensor_to_id[hash] = id
        else:
            self.tensor_to_id[node] = id

        return id

    def torch_dtype_to_mps_dtype(self, torch_dtype: torch.dtype) -> MPSDataType:
        return edge_dtype_to_mps_dtype(torch_dtype)

    def get_serialized_dtype(
        self,
        node: torch.fx.Node,
    ) -> MPSDataType:
        return self.torch_dtype_to_mps_dtype(node.meta["val"].dtype)

    def create_tertiary_node(
        self, node: torch.fx.Node, mps_graph: MPSGraph, tertiary_op: MPSNodeUnion
    ):
        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        input2_id = self.define_tensor(get_input_node(node, 1), mps_graph)
        input3_id = self.define_tensor(get_input_node(node, 2), mps_graph)
        output_id = self.define_tensor(node, mps_graph)
        return MPSNode(
            mpsnode_union=tertiary_op(
                input1_id=input1_id,
                input2_id=input2_id,
                input3_id=input3_id,
                output_id=output_id,
            )
        )

    def create_binary_node(
        self, node: torch.fx.Node, mps_graph: MPSGraph, binary_op: MPSNodeUnion
    ) -> MPSNode:
        input1_node = get_input_node(node, 0)
        input1_id = self.define_tensor(input1_node, mps_graph)

        # Handle both tensor and scalar variants of the op.
        # In case of scalar ops, manually define a constant and serialize it in the FlatBuffer.
        if isinstance(node.args[1], torch.fx.Node):
            # Second argument is a node.
            input2_id = self.define_tensor(get_input_node(node, 1), mps_graph)
        else:
            # Second argument is a scalar.
            scalar_val = get_scalar_val(node, 1)
            if input1_node.meta["val"].dtype == torch.float32:
                scalar_val = float(scalar_val)
            input2_id = self.define_scalar(
                scalar_val, self.get_serialized_dtype(input1_node), mps_graph
            )

        output_id = self.define_tensor(node, mps_graph)
        return MPSNode(
            mpsnode_union=binary_op(
                input1_id=input1_id, input2_id=input2_id, output_id=output_id
            )
        )

    def create_unary_node(
        self, node: torch.fx.Node, mps_graph: MPSGraph, unary_op: MPSNodeUnion
    ) -> MPSNode:
        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        output_id = self.define_tensor(node, mps_graph)
        return MPSNode(mpsnode_union=unary_op(input1_id=input1_id, output_id=output_id))


# This will hold mapping of all node names to the visitor class.
_node_visitor_dict = {}


def register_node_visitor(visitor):
    assert (
        isinstance(visitor, type)
        and issubclass(visitor, NodeVisitor)
        and hasattr(visitor, "target")
    ), f"Illformed NodeVisitor subclass, can't register!, got: {visitor}"
    if isinstance(visitor.target, list):
        for elem in visitor.target:
            _node_visitor_dict[elem] = visitor
    else:
        _node_visitor_dict[visitor.target] = visitor


def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    node_visitors = {}
    """
    Create a new class instance at runtime, and put them in a dict
    """
    for target, visitor in _node_visitor_dict.items():
        assert callable(
            visitor
        ), f"Expecting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(*args)

    placeholder_output_visitor = NodeVisitor(*args)
    node_visitors["placeholder"] = placeholder_output_visitor
    node_visitors["output"] = placeholder_output_visitor
    return node_visitors


def process_placeholder_nodes(
    exported_program: ExportedProgram,
    edge_graph_module: torch.fx.GraphModule,
    mps_graph: MPSGraph,
    placeholder_visitor: NodeVisitor,
) -> None:
    # Visit the placeholder nodes in the same order they are passed to the
    # forward function - forward(*args). When lifted graphs are being used,
    # parameters/buffers are lifted as placeholders and the order of the args
    # is not matching anymore with the original graph. We can retrieve the
    # original order by parsing all the placeholder nodes, and check if they are
    # constant tensors.
    #
    # Constant tensors will be bundled directly in the FlatBuffer and they won't be
    # provided by ExecuTorch during runtime.

    for node in edge_graph_module.graph.nodes:
        if node.op == "placeholder" and not is_parameter(
            exp_prog=exported_program, node=node
        ):
            if node.meta["val"] is None:
                continue

            input_id = placeholder_visitor.define_tensor(node, mps_graph)
            mps_graph.input_ids.append(input_id)

            if (
                placeholder_visitor.convert_model_to_fp16
                and node.meta["val"].dtype == torch.float32
            ):
                mps_node = MPSNode(
                    mpsnode_union=MPSCast(
                        input1_id=input_id,
                        output_id=input_id,
                        dtype=MPSDataType.mps_data_type_float16,
                    )
                )
                mps_graph.mps_nodes.append(mps_node)


def process_output_node(
    output_node,
    mps_graph: MPSGraph,
    output_visitor: NodeVisitor,
) -> None:
    output_id = output_visitor.define_tensor(output_node, mps_graph)
    mps_graph.output_ids.append(output_id)

    if (
        output_visitor.convert_model_to_fp16
        and output_node.meta["val"].dtype == torch.float32
    ):
        mps_node = MPSNode(
            mpsnode_union=MPSCast(
                input1_id=output_id,
                output_id=output_id,
                dtype=MPSDataType.mps_data_type_float32,
            )
        )
        mps_graph.mps_nodes.append(mps_node)
