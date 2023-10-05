# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, Optional, Union

import torch
from executorch.backends.xnnpack.passes.tag_implicit_q_dq_pass import TagImplicitQDqPass
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.backends.xnnpack.utils.utils import check_or_raise, is_param_node
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

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
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
        )

    @classmethod
    def from_q_dq_node(cls, quant_node: torch.fx.Node) -> QuantParams:
        check_or_raise(
            is_quant(quant_node) or is_dequant(quant_node),
            f"building quantizer from q/dq node but was given node:{quant_node}",
        )
        q_input = quant_node.all_input_nodes[0]

        if quant_node.target in [
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        ]:
            return cls._from_dynamic_input_node(q_input)

        per_channel = quant_node.target in [
            exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        ]
        scale = quant_node.args[1]
        zp = quant_node.args[2]
        axis = 0
        if per_channel:
            assert isinstance(scale, torch.fx.Node) and isinstance(scale.target, str)
            assert isinstance(zp, torch.fx.Node) and isinstance(zp.target, str)
            # TODO: use get_param_tensor()
            scale = getattr(quant_node.graph.owning_module, scale.target)
            zp = getattr(quant_node.graph.owning_module, zp.target)
            axis = cast(int, quant_node.args[3])

        check_or_raise(
            bool(
                quant_node.args[-1] != torch.uint8
                or quant_node.args[-1] != torch.quint8
            ),
            "XNNPACK does not support unsigned quantization",
        )
        dtype = cast(torch.dtype, quant_node.args[-1])
        qmax = cast(int, quant_node.args[-2])
        qmin = cast(int, quant_node.args[-3])
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
        )

    @classmethod
    def from_weights(cls, tensor_node: torch.fx.Node) -> Optional[QuantParams]:
        # Ignore transpose for weights
        # TODO:T148540997 remove the t_copy/permute_copy check when convert addmm to linear
        dq = (
            tensor_node.all_input_nodes[0]
            if tensor_node.target
            in (
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.aten.t_copy.default,
            )
            else tensor_node
        )
        # check input of t_copy/permute_copy is dequant
        if not is_dequant(dq):
            return None

        # input of dq is q
        check_or_raise(is_quant(dq.all_input_nodes[0]), "expected input to dq to be q")

        q = dq.all_input_nodes[0]

        # replace this with pointing to the actual weight value.
        # if no one else uses this weight value then take it out of the toplevel module
        check_or_raise(
            q.all_input_nodes[0].op in ["get_attr", "placeholder"],
            f"q->dq->permute_copy not derived from static weight, input to the q node: {q.all_input_nodes[0]}",
        )

        return cls.from_q_dq_node(q)

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

        check_or_raise(
            isinstance(input_quantizer.scale, float),
            f"q_input scale should be float, but got {input_quantizer.scale}",
        )
        return cls(
            per_channel=weight_quantizer.per_channel,
            q_input=bias,
            scale=weight_quantizer.scale * cast(float, input_quantizer.scale),
            zp=weight_quantizer.zp * 0,
            axis=weight_quantizer.axis,
            dtype=torch.int32,
            qmin=-(2**31),
            qmax=(2**31) - 1,
            is_output=False,
            is_input=False,
        )
