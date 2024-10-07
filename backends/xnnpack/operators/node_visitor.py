# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes

from typing import cast, Dict, List, Optional, Tuple

import torch
from executorch.backends.transforms import get_shape

from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)

from executorch.backends.xnnpack.operators.quant_params import QuantParams

from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    ConstantDataOffset,
    PerChannelGroupQuant,
    PerChannelQuant,
    PerTensorQuant,
    PerTokenDynamicQuant,
    XNNDatatype,
    XNNGraph,
    XNNQuantizedTensorValue,
    XNNQuantParams,
    XNNTensorValue,
    XValue,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_input_node,
    get_param_tensor,
    is_param_node,
    PERM_NCHW_TO_NHWC,
)

from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID
from torch.export import ExportedProgram

XNN_TYPE_MAP = {
    torch.float32: XNNDatatype.xnn_datatype_fp32,
}

from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    _aligned_size,
    _pad_to,
    CONSTANT_TENSOR_ALIGNMENT,
)


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

    def __init__(
        self,
        exported_program: ExportedProgram,
        external_ids: Dict,
        constant_data_bytes: bytearray,
    ) -> None:
        self._external_ids = external_ids or {}
        self._exported_program = exported_program or None
        self._constant_data_bytes = constant_data_bytes

    @property
    def external_ids(self) -> Dict:
        return self._external_ids

    @property
    def exported_program(self) -> ExportedProgram:
        return self._exported_program

    def is_graph_input(self, tensor: torch.fx.Node) -> bool:
        """
        Checks if the given tensor is a graph input

        Args:
            tensor: EdgeIR Tensor that is being checked for graph input
        """
        return tensor.op == "placeholder" and not is_param_node(
            self.exported_program, tensor
        )

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
        is_input = self.is_graph_input(tensor) or (
            quant_params.is_input
            and not is_param_node(self.exported_program, quant_params.q_input)
            if quant_params
            else False
        )

        # Tensor can still be output if its quantizing node is an output
        is_output = self.is_graph_output(tensor) or (
            quant_params.is_output if quant_params else False
        )

        if is_input:
            tensor_input = tensor
            if (
                quant_params
                and quant_params.is_input
                and not is_param_node(self.exported_program, quant_params.q_input)
                and not self.is_graph_input(tensor)
            ):
                tensor_input = quant_params.q_input

            assert (
                tensor_input in self.external_ids.keys()
            ), f"Tensor {tensor_input}, is_input. ext_ids: {self.external_ids.keys()}"

            ext_id = self.external_ids[tensor_input].external_id
            xnn_graph.input_ids.append(id_out)
            flag = self.external_ids[tensor_input].io_type

        elif is_output:
            tensor_output = tensor
            if (
                quant_params
                and quant_params.is_output
                and not self.is_graph_output(tensor)
            ):
                tensor_output = list(tensor.users)[0]

            assert (
                tensor_output in self.external_ids.keys()
            ), f"Tensor {tensor_output} is_output. ext_ids: {self.external_ids.keys()}"

            ext_id = self.external_ids[tensor_output].external_id
            xnn_graph.output_ids.append(id_out)
            flag = self.external_ids[tensor_output].io_type

        return ext_id, id_out, flag

    def get_serialized_dtype(
        self,
        quant_params: Optional[QuantParams],
        node: torch.fx.Node,
        fp32_static_weight: bool = False,
    ) -> XNNDatatype:
        # Default initialization
        dtype = XNNDatatype.xnn_datatype_fp32

        def get_node_dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
            """
            Extract the tensor.dtype from the node meta data if possible
            """
            node_val = node.meta.get("val", None)
            if node_val is not None:
                if isinstance(node_val, torch.Tensor):
                    return node_val.dtype

        # only for static quant
        def get_per_channel_dtype(
            quant_params: QuantParams,
        ) -> XNNDatatype:
            if quant_params.dtype == torch.int32:
                return XNNDatatype.xnn_datatype_qcint32
            elif quant_params.dtype == torch.int8:
                if quant_params.is_per_channel_group:
                    # 4-bit per channel group quantized weights
                    # No 8-bit support yet
                    assert (
                        quant_params.is_qc4w is True
                    ), "Only 4-bit per channel group quantization is supported"
                    return XNNDatatype.xnn_datatype_qbint4
                else:
                    # 4/8-bit per channel quantized weights
                    return (
                        XNNDatatype.xnn_datatype_qcint4
                        if quant_params.is_qc4w
                        else XNNDatatype.xnn_datatype_qcint8
                    )
            else:
                raise RuntimeError(
                    f"Unable to resolve static quantized tensor dtype using quant params dtype: {quant_params.dtype}, [qmin, qmax]: {quant_params.qmin}, {quant_params.qmax} for per channel quantization"
                )

        if quant_params is not None:
            if quant_params.is_dynamic:
                dtype = XNNDatatype.xnn_datatype_qdint8
            else:
                if quant_params.per_channel:
                    dtype = get_per_channel_dtype(quant_params)
                else:
                    dtype = (
                        XNNDatatype.xnn_datatype_qint32
                        if quant_params.dtype == torch.int32
                        else XNNDatatype.xnn_datatype_qint8
                    )
        else:
            node_dtype = get_node_dtype(node)
            if node_dtype is not None and node_dtype == torch.float16:
                dtype = (
                    XNNDatatype.xnn_datatype_fp32
                    if fp32_static_weight
                    else XNNDatatype.xnn_datatype_fp16
                )

        return dtype

    def get_quant_params(self, quant_params: QuantParams) -> XNNQuantParams:
        if quant_params.per_channel:
            scale = cast(torch.Tensor, quant_params.scale)
            if quant_params.is_per_channel_group:
                return PerChannelGroupQuant(
                    scale=scale.flatten().tolist(),
                    channel_dim=quant_params.axis,
                    group_size=quant_params.group_size,
                )
            else:  # per_channel quant
                return PerChannelQuant(
                    scale=scale.tolist(),
                    channel_dim=quant_params.axis,
                )
        elif quant_params.is_dynamic:
            # NB:
            # We use per_token quantization for per_tensor quantization
            # Beacuase that's the only option in XNNPACK in absance of per_tensor dynamic quantization
            # TODO: Upstream support for per_tensor dynamic quantization or broadcasting same scale value internally
            return PerTokenDynamicQuant(
                num_nonbatch_dims=quant_params.num_nonbatch_dims,
            )

        return PerTensorQuant(
            scale=cast(float, quant_params.scale),
            zero_point=cast(int, quant_params.zp),
        )

    @staticmethod
    def _check_per_channel_group_params(
        quant_params: QuantParams, dims: List[int]
    ) -> None:
        # Make sure things are lining up for per_channel_group quantization case
        # Has to be done this late because we don't have clean access to the actual tensor
        assert quant_params.is_per_channel_group, "Not per_channel_group quantization"
        # linear weights will be in [oc, ic]. And per_channel quantization must be on axis 0
        num_groups = cast(torch.Tensor, quant_params.scale).shape[1]
        assert (
            quant_params.axis == 0
        ), "For per_channel_group quant, axis must be 0, but got {axis}"
        assert (
            len(dims) == 2
        ), "For per_channel_group quant, expecting linear weights to be 2d, but got {len(dims)}"
        assert (
            num_groups > 0 and quant_params.group_size > 0
        ), "For per_channel_group quant, num_groups and group_size must be > 0, but got num_groups: {num_groups}, group_size: {quant_params.group_size}"
        output_channels = dims[quant_params.axis]
        input_channels = dims[quant_params.axis ^ 1]
        assert (
            output_channels == cast(torch.Tensor, quant_params.scale).shape[0]
        ), "For per_channel_group quant, expecting output channels to match scale.shape[0], gut got: {output_channels}, scale.shape[0]: {quant_params.scale.shape[0]}"
        assert (
            input_channels % num_groups == 0
        ), "For per_channel_group quant, expecting input channels to be divisible by num_groups, but got ic: {input_channels}, num_groups: {num_groups}"
        assert (
            input_channels % quant_params.group_size == 0
        ), "For per_channel_group quant, expecting input channels to be divisible by group_size, but got ic: {input_channels}, group_size: {quant_params.group_size}"
        assert (
            input_channels / quant_params.group_size == num_groups
        ), "For per_channel_group quant, expecting input channels // group_size == num_groups, but got ic: {input_channels}, group_size: {quant_params.group_size}, num_groups: {num_groups}"

        # For now group quantization is only supported for 4b weights
        assert quant_params.is_qc4w, "Only 4b group quantization is supported"

    def define_tensor(
        self,
        tensor: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        convert_to_nhwc: bool = False,
        swap_nc_for_depthwise_weights: bool = False,
        quant_params: Optional[QuantParams] = None,
        fp32_static_weights: bool = False,
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
            fp32_static_weights: XNN_FLAG_FP32_STATIC_WEIGHTS for fp16 conv
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
        dims = [1] if len(dims) == 0 else dims

        # check for per_channel_group quantization
        if quant_params and quant_params.per_channel_group:
            self._check_per_channel_group_params(quant_params, dims)

        # constant values serialize data
        buffer_idx = self.get_serialized_buffer_index(
            tensor,
            xnn_graph,
            vals_to_ids,
            convert_to_nhwc,
            swap_nc_for_depthwise_weights,
            quant_params,
            fp32_static_weights,
        )

        # convert tensor shape must reflect memory format, default is contiguous, so
        # only permute shape if we are converting the tensor to nhwc format
        if swap_nc_for_depthwise_weights:
            dims = [dims[1], dims[0]] + dims[2:]
        if convert_to_nhwc:
            check_or_raise(len(dims) == 4, "Converting to nhwc requires 4d tensor")
            dims = [dims[i] for i in PERM_NCHW_TO_NHWC]

        dtype = self.get_serialized_dtype(
            quant_params, tensor, fp32_static_weight=fp32_static_weights
        )

        tvalue = XNNTensorValue(
            datatype=dtype,
            num_dims=len(dims),
            dims=dims,
            external_id=ext_id,
            constant_buffer_idx=buffer_idx,
            flags=flag,
            id_out=id_out,
        )

        # Override the quant params axis since we have
        # updated the weights for depthwise, with that the out_channels dim
        # will be dims[3] instead of dims[0]. Let's update the per_channel
        # quant axis to match the new weight tensor before serializing
        if swap_nc_for_depthwise_weights and (
            quant_params and quant_params.per_channel
        ):
            if quant_params.axis == 0:
                quant_params.axis = len(dims) - 1
            else:
                assert f"Unsupported weight per channel quantization axis for depthwise conv2d: {quant_params.axis}, expecting 0."

        # Serialize tensor value
        ser_val = (
            XValue(xvalue_union=tvalue)
            if quant_params is None
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

    @staticmethod
    def convert_to_qc4w(inp: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to a quantized channelwise tensor 4bit tensor
        """

        import torch.nn.functional as F

        # Assert we got a properly quantized tensor.
        min, max = inp.min().item(), inp.max().item()
        assert (
            max <= 7 and min >= -8
        ), f"convert_to_qc4w: [min,max] out of [-8, 7] range, got [{min}, {max}]"

        # Assuming we have a 2d tensor
        if inp.ndim != 2:
            inp = inp.squeeze()
        assert (
            inp.ndim == 2
        ), f"convert_to_qc4w: expecting input tensor to be 2d, got {inp.ndim}"

        # pad ic
        if inp.shape[-1] % 2 != 0:
            inp = F.pad(input=inp, pad=(0, 1, 0, 0), mode="constant", value=0)

        # Shape after padding
        oc, ic = inp.shape
        assert ic % 2 == 0, "convert_to_qc4w: expecting ic to be even"

        # Adjust inp tensor for zp
        inp = inp.to(dtype=torch.uint8) + 8

        # Prepare the Result tensor
        inp = inp.contiguous().view(-1)
        return (inp[1::2] << 4 | inp[::2]).view(oc, int(ic / 2))

    def get_serialized_buffer_index(
        self,
        tensor: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        convert_to_nhwc: bool,
        swap_nc_for_depthwise_weights: bool,
        quant_params: Optional[QuantParams],
        fp32_static_weights: bool = False,
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
            fp32_static_weights: bool to indicate whether tensor is fp32 static weights

        Returns:
            buffer_idx: idx of the serialized data. 0 If not associated constant
                        data
        """
        # The get_attr node is the input to quant_params.
        get_attr_node = tensor if quant_params is None else quant_params.q_input
        if not is_param_node(self.exported_program, get_attr_node):
            check_or_raise(
                not swap_nc_for_depthwise_weights,
                "Swapping N and C dimensions is only valid for constant data tensors",
            )
            return 0

        buffer_idx = len(xnn_graph.constant_data)
        const_val = get_param_tensor(self.exported_program, get_attr_node)
        assert const_val is not None and isinstance(const_val, torch.Tensor)
        const_val = const_val.contiguous()

        # Quantize buffer if static data is indeed quantized
        if quant_params is not None and not quant_params.is_dynamic:
            const_val = quant_params.quantize_tensor(const_val).contiguous()
        elif const_val.dtype != torch.float16 or fp32_static_weights:
            # ensure that the const is fp32
            const_val = const_val.to(dtype=torch.float32).contiguous()

        if swap_nc_for_depthwise_weights:
            const_val = const_val.permute(
                dims=((1, 0) + tuple(range(2, const_val.dim())))
            ).contiguous()

        if convert_to_nhwc:
            const_val = const_val.to(memory_format=torch.channels_last)

        if quant_params is not None and quant_params.is_qc4w:
            const_val = self.convert_to_qc4w(const_val)

        array_type = ctypes.c_char * const_val.untyped_storage().nbytes()
        array = ctypes.cast(
            const_val.untyped_storage().data_ptr(),
            ctypes.POINTER(array_type),
        ).contents

        offset = len(self._constant_data_bytes)
        size = const_val.untyped_storage().nbytes()
        xnn_graph.constant_data.append(ConstantDataOffset(offset=offset, size=size))
        self._constant_data_bytes.extend(
            _pad_to(bytes(array), _aligned_size(size, CONSTANT_TENSOR_ALIGNMENT))
        )

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
                    quant_params=QuantParams.from_inputs(inp, self._exported_program),
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
            input_quant_params = QuantParams.from_inputs(
                input_node, self._exported_program
            )
            self.define_tensor(
                input_node,
                xnn_graph,
                vals_to_ids,
                quant_params=input_quant_params,
                convert_to_nhwc=convert_to_nhwc,
            )
            # Define Weight Node
            weight_node = get_input_node(node, input_type_map.node_weight)
            weight_quant_params = QuantParams.from_weights(
                weight_node, self._exported_program
            )
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
        ), f"Expecting a callable class, but got {visitor} of type {type(visitor)}"
        node_visitors[target] = visitor(*args)
    return node_visitors
