# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, Tuple

import executorch.backends.vulkan.utils as utils
import torch

import torch.nn.functional as F

from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

#################
## linear_qcnw ##
#################


def matches_linear_qcnw_pattern(  # noqa: C901
    program: ExportedProgram, node: torch.fx.Node
) -> Optional[Tuple[torch.qscheme, int]]:
    """
    Checks if the nodes surrounding a linear node matches the pattern for weight only
    quantized linear, where the weight is quantized channelswise to n bits.

    If the graph pattern matches, then return a tuple of (quantization_method, nbits)
    describing the type of quantization used for the weights. Otherwise, return None.
    """
    if not utils.is_linear_node(node):
        return None

    input_node = node.args[0]
    weight_node = node.args[1]

    # Type checking
    if not isinstance(weight_node, torch.fx.Node):
        return None
    if not isinstance(input_node, torch.fx.Node):
        return None

    # The input arg should not be a dequant node; if it is, then it is indicative that
    # dynamically quantized linear should be used instead
    if utils.is_dequant_node(input_node):
        return None

    # The weight arg should be a dequant node dequantizing the quantized weight
    # Furthermore, the op expects per channel quantization of the weight
    if not utils.is_dequant_per_channel_node(weight_node):
        return None

    orig_weight = weight_node.args[0]
    zeros = weight_node.args[2]

    # Type checking
    if not isinstance(orig_weight, torch.fx.Node):
        return None
    if not is_param_node(program, orig_weight):
        return None
    if not isinstance(zeros, torch.fx.Node):
        return None
    if not is_param_node(program, zeros):
        return None

    zeros_tensor = get_param_tensor(program, zeros)
    if not isinstance(zeros_tensor, torch.Tensor):
        return None

    quant_method = torch.per_channel_affine
    # Check for symmetric quantization, where the zeros used for dequantization will
    # actually be all zeros.
    if torch.all(zeros_tensor == 0):
        quant_method = torch.per_channel_symmetric

    orig_weight_tensor = get_param_tensor(program, orig_weight)
    if not isinstance(orig_weight_tensor, torch.Tensor):
        return None
    # Sanity check the dtype of the quantized weight
    if orig_weight_tensor.dtype != torch.int8:
        return None

    quant_min = orig_weight_tensor.min().item()
    quant_max = orig_weight_tensor.max().item()
    # Determine the number of bits the weight has been quantized to
    if quant_min >= -8 and quant_max <= 7:
        return quant_method, 4
    elif quant_min >= -128 and quant_max <= 127:
        return quant_method, 8

    return None


def pack_4bit_weight_tensor(inp: torch.Tensor) -> torch.Tensor:
    """
    Given a 8-bit weight tensor containing values quantized to 4 bits, create a packed
    weight tensor by packing 2 4-bit values in one unsigned 8-bit value.

    An input weight tensor of shape (M, K) will produce a packed weight tensor of shape
    (M, K / 2).
    """

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
    return (inp[::2] << 4 | inp[1::2]).view(oc, int(ic / 2))


def fuse_into_linear_qcnw_node(
    program: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    linear_node: torch.fx.Node,
    quant_method: torch.qscheme,
    nbits: int,
) -> None:
    """
    The weight_int8pack_mm operator represents a weight only quantized linear operator,
    where the weight tensor has been quantized channelswise to nbits bits.

      After the PT2E quantization flow, the expected graph pattern is

          dq_weight = dequantize(weight, scales)
          out = linear(activation, dq_weight, bias?)

      The goal of this function is to condense that sequence into

          out = quantized_linear(activation, dq_weight, scales)
          out = out + bias
    """
    activation = linear_node.args[0]
    dq_weight_node = linear_node.args[1]
    assert isinstance(activation, torch.fx.Node)
    assert isinstance(dq_weight_node, torch.fx.Node)

    bias = None
    if len(linear_node.args) > 2:
        bias = linear_node.args[2]
        assert isinstance(bias, torch.fx.Node)

    orig_weight = dq_weight_node.args[0]
    scale = dq_weight_node.args[1]

    # For 4 bit quantization, pack the weight tensor
    if nbits == 4:
        assert isinstance(orig_weight, torch.fx.Node)
        orig_weight_tensor = get_param_tensor(program, orig_weight)
        assert isinstance(orig_weight_tensor, torch.Tensor)
        packed_weight_tensor = pack_4bit_weight_tensor(orig_weight_tensor)
        utils.update_program_state_dict(
            program,
            orig_weight.name,
            packed_weight_tensor,
        )
        orig_weight.meta["val"] = orig_weight.meta["val"][:, ::2].to(torch.uint8)

    if nbits == 8 and quant_method == torch.per_channel_symmetric:
        op_target = exir_ops.edge.aten._weight_int8pack_mm.default
    elif nbits == 4 and quant_method == torch.per_channel_symmetric:
        op_target = exir_ops.edge.et_vk.linear_qcs4w.default
    else:
        raise NotImplementedError(
            "only 4 and 8 bits per channel symmetric quant supported for linear_qcnw"
        )

    with graph_module.graph.inserting_before(linear_node):
        weight_int8pack_mm_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            (activation, orig_weight, scale),
        )
        if bias:
            add_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.add.Tensor,
                (weight_int8pack_mm_node, bias),
            )
            linear_node.replace_all_uses_with(add_node)
        else:
            linear_node.replace_all_uses_with(weight_int8pack_mm_node)
        graph_module.graph.erase_node(linear_node)
        graph_module.graph.erase_node(dq_weight_node)


class FuseQuantizedOpsTransform(ExportPass):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.program = exported_program

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            # Check for linear_qcnw pattern (weight-only quantization)
            qcnw_details = matches_linear_qcnw_pattern(self.program, node)
            if qcnw_details is not None:
                qcnw_method, qcnw_nbits = qcnw_details
                fuse_into_linear_qcnw_node(
                    self.program, graph_module, node, qcnw_method, qcnw_nbits
                )
                continue

        graph_module.recompile()
        dead_code_elimination_pass(graph_module)

        # Re-trace the graph since new nodes were (potentially) inserted
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
