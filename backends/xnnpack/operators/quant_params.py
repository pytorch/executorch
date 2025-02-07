# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, Optional, Union

import torch
from executorch.backends.xnnpack._passes.tag_implicit_q_dq_pass import (
    TagImplicitQDqPass,
)
from executorch.backends.xnnpack.utils.quant_utils import (
    extract_qdq_affine_op_args_for_decomposed_ops,
    is_affine_qdq,
    is_dequant,
    is_dynamic_qdq,
    is_per_channel,
    is_per_channel_group,
    is_quant,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


class QuantParams:
    """
    QuantParams class, to represent the paramaters and meta data needed
    to quantize a tensor. The metadata can technically all be encapsulated
    within the quant torch.fx.Node, however, there are some cases in which
    nodes which are meant to be quantized for XNNPACK are not quantized
    in PyTorch IR, specifically bias nodes. In this case, we can still build
    quantizer class to serialize the quantized attributes needed for XNNPACK.

    Attributes:
        per_channel: Whether this quantization is per channel or per tensor
        q_input: node that is the input to this quantization
        scale: tensor or float that is used as the quantization scale
        zp: tensor or float that is used as the quantization zero point
        axis: used for per_channel quantizaiton, representing the axis
        dtype: dtype of the type being quantized to
        qmin: quantization minimum
        qmax: quantization maximum
        is_output: whether this is an output node or not
        is_input: whether this is an input node or not
    """

    def __init__(
        self,
        per_channel: bool,
        q_input: torch.fx.Node,
        scale: Union[torch.Tensor, float],
        zp: Union[torch.Tensor, float],
        axis: int,
        dtype: torch.dtype,
        qmax: int,
        qmin: int,
        is_output: bool,
        is_input: bool,
        is_dynamic: bool = False,
        num_nonbatch_dims: int = 1,
        group_size: int = 0,
    ) -> None:
        self.per_channel = per_channel
        self.q_input = q_input
        self.scale = scale
        self.zp = zp
        self.axis = axis
        self.dtype = dtype
        self.qmax = qmax
        self.qmin = qmin
        self.is_output = is_output
        self.is_input = is_input
        self.is_dynamic = is_dynamic
        self.num_nonbatch_dims = num_nonbatch_dims
        self.is_qc4w = (
            self.per_channel
            and not self.is_dynamic
            and self.qmin == -8
            and self.qmax == 7
            and self.dtype == torch.int8
        )

        # Groupwise quantization for weight
        self.per_channel_group = False
        self.group_size = group_size
        if self.group_size > 0:
            assert (
                self.per_channel is True
            ), "Only per channel quantization supports groupwise quantization"
            assert (
                cast(torch.Tensor, scale).ndim == 2
            ), "Scale must be 2D for per channel groupwise quant"
            self.per_channel_group = True
            assert group_size > 0, "Group size must be greater than 0"
        self.is_per_channel_group = self.per_channel and self.group_size > 0

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # Do nothing if already quantized by the Quantizer
        if tensor.dtype == self.dtype:
            return tensor

        if self.per_channel:
            assert (
                self.per_channel_group is False
            ), f"Not expecting per channel group quantization, got q dtype: {self.dtype}, tensor.dtype {tensor.dtype}"
            assert (
                tensor.shape[self.axis] == cast(torch.Tensor, self.scale).shape[0]
            ), f"Invalid size of per channel quantization scales, axis: {self.axis}, scale size: {self.scale.shape}, tensor shape: {tensor.shape}"

            assert (
                tensor.shape[self.axis] == cast(torch.Tensor, self.zp).shape[0]
            ), f"Invalid size of per channel quantization zero-points, axis: {self.axis}, zp size: {self.zp.shape}, tensor shape: {tensor.shape}"

            # Assuming folded quant weights
            # TODO Add support for unfolded weights
            assert not self.is_qc4w, "Not expecting QC4W per channel tensor"

            return exir_ops.edge.quantized_decomposed.quantize_per_channel.default(
                tensor, self.scale, self.zp, self.axis, self.qmin, self.qmax, self.dtype
            )
        else:
            return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                tensor, self.scale, self.zp, self.qmin, self.qmax, self.dtype
            )

    @classmethod
    def _from_dynamic_input_node(cls, quant_node: torch.fx.Node) -> QuantParams:
        q_input = quant_node.args[0]  # fp32 input
        assert isinstance(q_input, torch.fx.Node)
        # TODO - materialize this from the quant_node scale count and val shape
        num_nonbatch_dims = 1

        return cls(
            per_channel=False,  # True is not valid
            q_input=q_input,
            scale=0.0,  # no need
            zp=0.0,  # no need
            axis=0,  # no need
            dtype=torch.float32,  # will be quantized at runtime
            qmax=0,  # no need
            qmin=0,  # no need
            is_output=False,
            is_input=q_input.op == "placeholder",
            is_dynamic=True,
            num_nonbatch_dims=num_nonbatch_dims,
        )

    @classmethod
    def from_q_dq_node(
        cls, quant_node: torch.fx.Node, ep: Optional[ExportedProgram] = None
    ) -> QuantParams:
        check_or_raise(
            is_quant(quant_node) or is_dequant(quant_node),
            f"building quantizer from q/dq node but was given node:{quant_node}",
        )
        q_input = quant_node.all_input_nodes[0]

        # TODO: Use presence of choose_qparam node to determine if this is a dynamic quantization
        if is_dynamic_qdq(quant_node):
            return cls._from_dynamic_input_node(quant_node)

        per_channel = is_per_channel(quant_node)

        _groupwise = is_per_channel_group(quant_node)
        quant_node_args = quant_node.args
        if _groupwise and is_affine_qdq(quant_node):
            quant_node_args = extract_qdq_affine_op_args_for_decomposed_ops(quant_node)

        scale = quant_node_args[1]
        zp = quant_node_args[2]
        axis = 0
        if per_channel:
            assert isinstance(scale, torch.fx.Node) and isinstance(scale.target, str)
            assert isinstance(zp, torch.fx.Node) and isinstance(zp.target, str)
            assert (
                ep is not None
            ), "ExportedProgram must be provided to extract per channel params"

            def _get_tensor(node):
                param = get_param_tensor(ep, node)
                assert param is not None, f"Expected to find param tensor for {node}"
                return cast(torch.Tensor, param)

            scale = _get_tensor(scale)
            zp = _get_tensor(zp)
            axis = cast(int, quant_node_args[3])

            if _groupwise:
                scale_tensor = cast(torch.Tensor, scale)
                if scale_tensor.ndim == 1:
                    scale_tensor = scale_tensor.reshape(-1, 1)
                    zp = zp.reshape(-1, 1)
                    scale = scale_tensor

                assert (
                    scale_tensor.ndim == 2
                ), "Weight scale must be 2D for per_channel_group [de]quant node, got {scale.ndim}D"
                axis = 0  # axis is ignored for groupwise quantization

        check_or_raise(
            bool(
                quant_node_args[-1] != torch.uint8
                or quant_node_args[-1] != torch.quint8
            ),
            "XNNPACK does not support unsigned quantization",
        )

        if _groupwise:
            _ = quant_node_args[-1]  # output dtype - not used
            group_size = cast(int, quant_node_args[-2])
            dtype = cast(torch.dtype, quant_node_args[-3])
            qmax = cast(int, quant_node_args[-4])
            qmin = cast(int, quant_node_args[-5])
        else:
            group_size = 0
            dtype = cast(torch.dtype, quant_node_args[-1])
            qmax = cast(int, quant_node_args[-2])
            qmin = cast(int, quant_node_args[-3])

        is_output = any(
            user_node.op == "output" for user_node in quant_node.users.keys()
        )
        is_input = q_input.op == "placeholder"
        return cls(
            per_channel,
            q_input,
            scale,
            zp,
            axis,
            dtype,
            qmax,
            qmin,
            is_output,
            is_input,
            group_size=group_size,
        )

    @classmethod
    def from_weights(
        cls, tensor_node: torch.fx.Node, ep: Optional[ExportedProgram] = None
    ) -> Optional[QuantParams]:
        if not is_dequant(tensor_node):
            return None

        # source node for quant params
        src = tensor_node

        # is input of dq is q?
        dq_input = src.all_input_nodes[0]
        if is_quant(dq_input):
            src = dq_input

        # replace this with pointing to the actual weight value.
        # if no one else uses this weight value then take it out of the toplevel module
        check_or_raise(
            src.all_input_nodes[0].op in ["get_attr", "placeholder"],
            f"q->dq->permute_copy not derived from static weight, input to the q or dq (for folded quant) node: {src.all_input_nodes[0]}",
        )

        return cls.from_q_dq_node(src, ep)

    @classmethod
    def from_inputs(
        cls, tensor_node: torch.fx.Node, ep: ExportedProgram
    ) -> Optional[QuantParams]:
        # tensor_node is quantized if it is produced by a dequant node
        if is_dequant(tensor_node) and TagImplicitQDqPass.is_tagged_as_implicit_q_dq(
            tensor_node
        ):
            dq_input = cast(torch.fx.Node, tensor_node.args[0])
            if is_quant(dq_input):
                q_input = cast(torch.fx.Node, dq_input.args[0])
                if is_param_node(ep, q_input):
                    return cls.from_q_dq_node(dq_input)
            return cls.from_q_dq_node(tensor_node)

        return None

    @classmethod
    def from_outputs(cls, tensor_node: torch.fx.Node) -> Optional[QuantParams]:
        # tensor_node can also be quantized if it is used as in q -> dq
        if len(tensor_node.users) == 1:
            q = list(tensor_node.users.keys())[0]
            # Check if user is a q node
            if is_quant(q) and TagImplicitQDqPass.is_tagged_as_implicit_q_dq(q):
                return cls.from_q_dq_node(q)

        return None

    @classmethod
    def from_bias(
        cls,
        bias: torch.fx.Node,
        weight_quantizer: Optional[QuantParams],
        input_quantizer: Optional[QuantParams],
    ) -> Optional[QuantParams]:
        if weight_quantizer is None or input_quantizer is None:
            check_or_raise(
                weight_quantizer is None and input_quantizer is None,
                "Weight and Input should both be quantized",
            )
            return None

        if input_quantizer.is_dynamic:
            # No need to quantize bias for dyanamic quantization
            return None

        check_or_raise(
            not input_quantizer.per_channel,
            "Input can not be quantized per channel",
        )

        # Only per_tensor quantization is supported for input here
        check_or_raise(
            isinstance(input_quantizer.scale, float),
            f"q_input scale should be float, but got {input_quantizer.scale}",
        )
        return cls(
            per_channel=weight_quantizer.per_channel,
            q_input=bias,
            scale=weight_quantizer.scale * cast(float, input_quantizer.scale),
            zp=weight_quantizer.zp * 0,
            axis=0,  # not using weight_quantizer.axis because bias is always of shape [out_channels] i.e. 1D
            dtype=torch.int32,
            qmin=-(2**31),
            qmax=(2**31) - 1,
            is_output=False,
            is_input=False,
        )
