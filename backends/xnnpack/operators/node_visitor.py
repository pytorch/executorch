import ctypes
from typing import cast, Dict, Optional, Tuple

import torch
from executorch.backends.transforms import get_shape

from executorch.backends.xnnpack.passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    Buffer,
    PerChannelQuant,
    PerTensorQuant,
    XNNDatatype,
    XNNGraph,
    XNNQuantizedTensorValue,
    XNNQuantParams,
    XNNTensorValue,
    XValue,
)

from executorch.backends.xnnpack.utils.quant_utils import QuantParams
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node

from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
)
from executorch.exir.dialects._ops import ops as exir_ops

XNN_TYPE_MAP = {
    torch.float32: XNNDatatype.xnn_datatype_fp32,
}
PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


class InputTypeToIndex:
    """
    Mapping from input type to the arg index of a node
    """

    node_input: int
    node_weight: int

    def __init__(self, node_input: int, node_weight: int, node_bias=None):
        self.node_input = node_input
        self.node_weight = node_weight
        self.node_bias = node_bias


def get_tensor_value(xvalue: XValue) -> XNNTensorValue:
    val_union = xvalue.xvalue_union
    if isinstance(val_union, XNNTensorValue):
        return val_union
    else:
        # it is XNNQuantizedTensorValue
        q_tensor = val_union
        return q_tensor.tensor_value


class NodeVisitor:
    """
    Node visitor pattern for visiting nodes in an edge IR graph and
    serializing them using the xnnpack serialization schema defined
    """

    def __init__(self, external_ids) -> None:
        self.external_ids = external_ids or {}

    def is_graph_input(self, tensor: torch.fx.Node) -> bool:
        """
        Checks if the given tensor is a graph input

        Args:
            tensor: EdgeIR Tensor that is being checked for graph input
        """
        return tensor.op == "placeholder"

    def is_graph_output(self, tensor: torch.fx.Node) -> bool:
        """
        Checks if the given tensor is used as a graph output

        Args:
            tensor: EdgeIR Tensor that is being checked for graph input
        """

        for user in tensor.users.keys():
            if user.op == "output":
                return True
        return False

    def gen_ids_and_flags(
        self,
        tensor: torch.fx.Node,
        xnn_graph: XNNGraph,
        quant_params: Optional[QuantParams],
    ) -> Tuple[int, int, int]:
        """
        Generate new id, external id, and flag values for tensor info

        Args:
           tensor: EdgeIR Tensor that is being defined into xnn_graph
           xnn_graph: XNNGraph object for serializing into flatbuffer
           quant_params: QuantParams object representing the q params of this tensor
                    is none if not quantized

        Returns:
            tuple of external_id, id_out and external input/output flags
        """
        id_out = len(xnn_graph.xvalues)
        ext_id = XNN_INVALID_VALUE_ID
        flag = 0

        # Dynamic quant isn't really a quant
        if quant_params is not None and quant_params.is_dynamic:
            tensor = quant_params.q_input

        # TODO tensor here for [placeholder -> q -> dq -> op] must be the placeholder node
        # This will break if we change the way q/dq are partitioned

        # Tensor can still be input if its quantizing node is an input
        is_input = (self).is_graph_input(tensor)
        # Tensor can still be output if its quantizing node is an output
        is_output = self.is_graph_output(tensor)
        # handle logic for input/output tensors
        if is_input or is_output:
            assert (
                tensor in self.external_ids.keys()
            ), f"Tensor {tensor}, is_input: {is_input}, is_output: {is_output}, ext_ids: {self.external_ids.keys()}"
            ext_id = self.external_ids[tensor].external_id
            if is_input:
                xnn_graph.input_ids.append(id_out)
                flag = XNN_VALUE_FLAG_EXTERNAL_INPUT
            if is_output:
                xnn_graph.output_ids.append(id_out)
                flag = XNN_VALUE_FLAG_EXTERNAL_OUTPUT

        return ext_id, id_out, flag

    def get_serialized_dtype(
        self, quant_params: Optional[QuantParams]
    ) -> Tuple[XNNDatatype, XNNDatatype]:
        dtype, dq_dtype = (
            XNNDatatype.xnn_datatype_fp32,
            XNNDatatype.xnn_datatype_invalid,
        )
        if quant_params is not None:
            if quant_params.is_dynamic:
                dq_dtype = XNNDatatype.xnn_datatype_qint8
            else:
                if quant_params.per_channel:
                    dtype = (
                        XNNDatatype.xnn_datatype_qcint32
                        if quant_params.dtype == torch.int32
                        else XNNDatatype.xnn_datatype_qcint8
                    )
                else:
                    dtype = (
                        XNNDatatype.xnn_datatype_qint32
                        if quant_params.dtype == torch.int32
                        else XNNDatatype.xnn_datatype_qint8
                    )

        return (dtype, dq_dtype)

    def get_quant_params(self, quant_params: QuantParams) -> XNNQuantParams:
        if quant_params.per_channel:
            scale = cast(torch.Tensor, quant_params.scale)
            return PerChannelQuant(
                scale=scale.tolist(),
                channel_dim=quant_params.axis,
            )

        return PerTensorQuant(
            scale=cast(float, quant_params.scale),
            zero_point=cast(int, quant_params.zp),
        )

    def define_tensor(
        self,
        tensor: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        convert_to_nhwc: bool = False,
        swap_nc_for_depthwise_weights: bool = False,
        quant_params: Optional[QuantParams] = None,
    ) -> None:
        """
        Defines an tensor value into the XNNGraph

        Args:
            tensor: EdgeIR Tensor that is being defined into xnn_graph
            xnn_graph: XNNGraph object for serializing into flatbuffer
            vals_to_ids: dictionary mapping edge_graph values(node targets) to
                        their corresponding ids in XNNGraph
            convert_to_nhwc: bool to indicate whether tensor shape should be permuted to
                        reflect the nhwc memory format.
            swap_nc_for_depthwise_weights: bool to indicate whether tensor shape
                        should be permuted such that the N and C dimensions are
                        swapped, which should be used for depthwise convolution
                        weights. This is only valid for tensors which hold
                        constant data. If used along with convert_to_nhwc, this
                        swap will happen before converting to nhwc.
            quant_params: Quantization meta data for this tensor, None if it is not quantized
        """

        if tensor in vals_to_ids:
            return

        if quant_params is not None:
            if quant_params.q_input in vals_to_ids:
                vals_to_ids[tensor] = vals_to_ids[quant_params.q_input]
                return
        # Tag added by ChannelsLastTaggedReshapePass
        convert_to_nhwc |= tensor.meta.get(
            ChannelsLastTaggedReshapePass.XNN_NHWC_NODE, False
        )

        # Get new xnn id for tensor value
        ext_id, id_out, flag = self.gen_ids_and_flags(tensor, xnn_graph, quant_params)
        dims = get_shape(tensor)

        # constant values serialize data
        buffer_idx = self.get_serialized_buffer(
            tensor,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc,
            swap_nc_for_depthwise_weights,
            quant_params,
        )

        # convert tensor shape must reflect memory format, default is contiguous, so
        # only permute shape if we are converting the tensor to nhwc format
        if tensor.target == exir_ops.edge.aten.t_copy.default:
            # We ignore transpose nodes and reverse the dims to before it
            dims = dims[::-1]
        if swap_nc_for_depthwise_weights:
            dims = [dims[1], dims[0]] + dims[2:]
        if convert_to_nhwc:
            check_or_raise(len(dims) == 4, "Converting to nhwc requires 4d tensor")
            dims = [dims[i] for i in [0, 2, 3, 1]]

        dtype, dq_dtype = self.get_serialized_dtype(quant_params)

        tvalue = XNNTensorValue(
            datatype=dtype,
            num_dims=len(dims),
            dims=dims,
            external_id=ext_id,
            constant_buffer_idx=buffer_idx,
            flags=flag,
            id_out=id_out,
            dq_datatype=dq_dtype,
        )

        ser_val = (
            XValue(xvalue_union=tvalue)
            if quant_params is None or quant_params.is_dynamic
            else XValue(
                xvalue_union=XNNQuantizedTensorValue(
                    tensor_value=tvalue,
                    quant_params=self.get_quant_params(quant_params),
                )
            )
        )

        xnn_graph.xvalues.append(ser_val)
        vals_to_ids[tensor] = id_out
        if quant_params is not None:
            vals_to_ids[quant_params.q_input] = id_out

    def get_serialized_buffer(
        self,
        tensor: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        convert_to_nhwc: bool,
        swap_nc_for_depthwise_weights: bool,
        quant_params: Optional[QuantParams],
    ) -> int:
        """
        If tensor holds some constant data, serialize it and return the
        index of its placement in the constant buffer

        Args:
            tensor: EdgeIR Tensor that is being defined into xnn_graph
            xnn_graph: XNNGraph object for serializing into flatbuffer
            vals_to_ids: dictionary apping edge_graph values(node targets) to
                        their corresponding ids in XNNGraph
            convert_to_nhwc: bool to indicate whether tensor shape should be permuted to
                        reflect the nhwc memory format.
            swap_nc_for_depthwise_weights: bool to indicate whether tensor shape
                        should be permuted such that the N and C dimensions are
                        swapped, which should be used for depthwise convolution
                        weights. This is only valid for tensors which hold
                        constant data. If used along with convert_to_nhwc, this
                        swap will happen before converting to nhwc.
            quant_params: Quantization meta data for this tensor, None if it is not quantize

        Returns:
            buffer_idx: idx of the serialized data. 0 If not associated constant
                        data
        """
        # The get_attr node is the input to quant_params.
        get_attr_node = tensor if quant_params is None else quant_params.q_input
        if get_attr_node.op != "get_attr":
            check_or_raise(
                not swap_nc_for_depthwise_weights,
                "Swapping N and C dimensions is only valid for constant data tensors",
            )
            return 0

        check_or_raise(
            len(xnn_graph.constant_buffer) == len(xnn_graph.mem_buffer_sizes),
            "Internal Error: const_buffer and buffer_sizes length mismatch",
        )
        buffer_idx = len(xnn_graph.constant_buffer)
        const_val = getattr(
            get_attr_node.graph.owning_module, get_attr_node.target
        ).contiguous()
        # Quantize buffer if static data is indeed quantized
        if quant_params is not None and not quant_params.is_dynamic:
            const_val = quant_params.quantize_tensor(const_val).contiguous()
        if swap_nc_for_depthwise_weights:
            const_val = const_val.permute(
                dims=((1, 0) + tuple(range(2, const_val.dim())))
            ).contiguous()
        if convert_to_nhwc:
            const_val = const_val.to(memory_format=torch.channels_last)

        # pyre-ignore
        array_type = ctypes.c_char * const_val.untyped_storage().nbytes()
        array = ctypes.cast(
            const_val.untyped_storage().data_ptr(),
            ctypes.POINTER(array_type),
        ).contents
        buffer = Buffer(storage=bytes(array))
        xnn_graph.constant_buffer.append(buffer)
        xnn_graph.mem_buffer_sizes.append(const_val.untyped_storage().nbytes())

        return buffer_idx

    def define_nodes_tensor_inputs_outputs(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        convert_to_nhwc: bool = False,
        input_type_map: Optional[InputTypeToIndex] = None,
    ) -> None:
        # serialize node outputs if not already defined
        self.define_tensor(
            node,
            xnn_graph,
            vals_to_ids,
            quant_params=QuantParams.from_outputs(node),
            convert_to_nhwc=convert_to_nhwc,
        )

        if input_type_map is None:
            # serialize node inputs if not already defined
            for inp in node.all_input_nodes:
                self.define_tensor(
                    inp,
                    xnn_graph,
                    vals_to_ids,
                    quant_params=QuantParams.from_inputs(inp),
                    convert_to_nhwc=convert_to_nhwc,
                )
        else:
            num_inputs = 3 if input_type_map.node_bias is not None else 2
            check_or_raise(
                num_inputs == len(node.all_input_nodes),
                f"Invalid input type map given, {input_type_map}, {num_inputs}, {node.all_input_nodes}",
            )
            # Define Input Node
            input_node = get_input_node(node, input_type_map.node_input)
            input_quant_params = QuantParams.from_inputs(input_node)
            self.define_tensor(
                input_node,
                xnn_graph,
                vals_to_ids,
                quant_params=input_quant_params,
                convert_to_nhwc=convert_to_nhwc,
            )
            # Define Weight Node
            weight_node = get_input_node(node, input_type_map.node_weight)
            weight_quant_params = QuantParams.from_weights(weight_node)
            self.define_tensor(
                weight_node,
                xnn_graph,
                vals_to_ids,
                quant_params=weight_quant_params,
                convert_to_nhwc=convert_to_nhwc,
            )
            # Define Bias Node
            if input_type_map.node_bias is not None:
                bias_node = get_input_node(node, input_type_map.node_bias)
                bias_quant_params = QuantParams.from_bias(
                    bias_node, weight_quant_params, input_quant_params
                )
                self.define_tensor(
                    bias_node,
                    xnn_graph,
                    vals_to_ids,
                    quant_params=bias_quant_params,
                    convert_to_nhwc=False,  # Bias is generally 1d and can not be in NHWC
                )

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        raise NotImplementedError("NodeVisitor must be extended!")


# This will hold mapping of all node names to the visitor class that will define
# the torch.fx.Node object into the XNNGraph. Don't use it directly!
_node_visitor_dict = {}


def register_node_visitor(visitor):
    assert (
        isinstance(visitor, type)
        and issubclass(visitor, NodeVisitor)
        and hasattr(visitor, "target")
    ), f"Illformed NodeVisitor subclass, can't register!, got: {visitor}"
    _node_visitor_dict[visitor.target] = visitor


# @lru_cache - TODO enable caching - ATM dict being non hashable is causing issues with LRU cache
def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    node_visitors = {}
    """
    Create a new class instance at runtime, and put them in a dict
    """
    for target, visitor in _node_visitor_dict.items():
        assert callable(
            visitor
        ), f"Expeting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(*args)
    return node_visitors
