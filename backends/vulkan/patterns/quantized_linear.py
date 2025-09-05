# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Callable, List, Optional

import executorch.backends.vulkan.utils as utils

import torch
import torch.nn.functional as F

from executorch.backends.transforms.utils import get_param_tensor, is_param_node

from executorch.backends.vulkan.patterns.pattern_registry import (
    register_pattern_graph,
    register_pattern_replacement,
)

from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

from torch.export import export
from torch.fx.passes.utils.matcher_utils import InternalMatch

from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
from torchao.utils import unwrap_tensor_subclass


class TorchAOWeightOnlyQuantizedLinearPattern(torch.nn.Module):
    """
    Quantized linear pattern produced when quantizing linear layers using
    `torchao.quantization.quant_api.quantize_()` with IntxWeightOnlyConfig.
    """

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 256,
        bias: bool = False,
        group_size: int = 64,
        weight_bits: int = 4,
        granularity_class: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.group_size = group_size
        self.weight_bits = weight_bits

        if self.weight_bits == 4:
            # pyre-ignore[16]
            self.weight_dtype = torch.int4
        else:
            self.weight_dtype = torch.int8

        if granularity_class is not None:
            self.quant_granularity = granularity_class(self.group_size)
        else:
            self.quant_granularity = PerGroup(self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def apply_quantization(self):
        q_config = IntxWeightOnlyConfig(
            weight_dtype=self.weight_dtype,
            granularity=self.quant_granularity,
        )
        quantize_(self, q_config)
        unwrap_tensor_subclass(self)
        return self


@lru_cache(maxsize=None)
@register_pattern_graph("torchao_wo_quantized_linear")
def get_torchao_wo_quantized_linear_graphs() -> List[torch.fx.GraphModule]:
    graphs = []

    # Different configurations to test
    configs = [
        # gemv pattern
        (1, 1, 128, 128, False, 64, 4, PerGroup),
        # gemm pattern
        (1, 8, 128, 128, False, 64, 4, PerGroup),
    ]

    for (
        batch_size,
        seq_len,
        in_features,
        out_features,
        bias,
        group_size,
        weight_bits,
        granularity_class,
    ) in configs:
        for dtype in [torch.float32]:
            xs = []
            xs.append(torch.randn(batch_size, seq_len, in_features, dtype=dtype))
            if batch_size == 1:
                xs.append(torch.randn(seq_len, in_features, dtype=dtype))

            for x in xs:
                # Create and quantize the pattern
                pattern = TorchAOWeightOnlyQuantizedLinearPattern(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    group_size=group_size,
                    weight_bits=weight_bits,
                    granularity_class=granularity_class,
                )

                # Apply quantization
                pattern = pattern.apply_quantization()

                # Export the quantized pattern
                edge = to_edge(
                    export(
                        pattern,
                        (x,),
                    ),
                    compile_config=EdgeCompileConfig(_check_ir_validity=False),
                )
                gm = edge.exported_program().graph_module
                graphs.append(gm)

    return graphs


def pack_4bit_weight_tensor(inp: torch.Tensor) -> torch.Tensor:
    """
    Given a 8-bit weight tensor containing values quantized to 4 bits, create a packed
    weight tensor by packing 2 4-bit values in one unsigned 8-bit value.

    An input weight tensor of shape (M, K) will produce a packed weight tensor of shape
    (M, K / 2).

    The packing implemented here is the same as the packing produced by
    backends/vulkan/_passes/int4_weight_only_quantizer.py
    """

    # Assert we got a properly quantized tensor.
    min, max = inp.min().item(), inp.max().item()
    assert (
        max <= 7 and min >= -8
    ), f"pack_4bit_weight_tensor: [min,max] out of [-8, 7] range, got [{min}, {max}]"

    # Assuming we have a 2d tensor
    if inp.ndim != 2:
        inp = inp.squeeze()
    assert (
        inp.ndim == 2
    ), f"pack_4bit_weight_tensor: expecting input tensor to be 2d, got {inp.ndim}"

    # pad ic
    if inp.shape[-1] % 2 != 0:
        inp = F.pad(input=inp, pad=(0, 1, 0, 0), mode="constant", value=0)

    # Shape after padding
    oc, ic = inp.shape
    assert ic % 2 == 0, "convert_to_qc4w: expecting ic to be even"

    # Adjust inp tensor for zp
    inp = inp.to(dtype=torch.uint8) + 8
    # Pack each 4-bit value into a single 8-bit value
    return inp[::, ::2] << 4 | inp[::, 1::2]


def make_combined_scales_and_zeros_tensor(
    scales: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """
    Given a scales and zeros tensor, create a combined tensor by stacking them into a
    single tensor.

    The scales and zeros tensors are expected to be 2D tensors of shape
    (OUTPUT_CHANNELS, NUM_GROUPS). The combined tensor will have the shape
    (NUM_GROUPS, OUTPUT_CHANNELS, 2).

    This is the scales and zeros format produced by
    backends/vulkan/_passes/int4_weight_only_quantizer.py, which in turn is the scales
    and zeros format expected by the _weight_int4pack_mm op in ATen.
    """
    scales_reshaped = scales.transpose(0, 1).unsqueeze(2)
    zeros_reshaped = zeros.transpose(0, 1).unsqueeze(2)

    zeros_scaled = zeros_reshaped * scales_reshaped * -1
    return torch.cat((scales_reshaped, zeros_scaled), dim=2)


def identify_wo_quantized_linear_io_nodes(  # noqa: C901
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
) -> Optional[List[torch.fx.Node]]:
    dequant_node = None
    # First, find the dequant node
    for node in match.nodes_map.values():
        if utils.is_dequant_node(node):
            dequant_node = node
            break

    if dequant_node is None:
        return None

    quantized_weight = dequant_node.args[0]
    quant_scales = dequant_node.args[2]
    quant_zeros = dequant_node.args[3]

    if not isinstance(quantized_weight, torch.fx.Node) or not is_param_node(
        ep, quantized_weight
    ):
        return None
    if not isinstance(quant_scales, torch.fx.Node) or not is_param_node(
        ep, quant_scales
    ):
        return None
    if not isinstance(quant_zeros, torch.fx.Node) or not is_param_node(ep, quant_zeros):
        return None

    input_nodes = match.placeholder_nodes
    if len(input_nodes) != 4:
        return None

    in_tensor_node = None
    for node in input_nodes:
        if node not in dequant_node.args:
            in_tensor_node = node
            break

    if in_tensor_node is None:
        return None

    output_nodes = match.returning_nodes

    if len(output_nodes) != 1:
        return None

    out_tensor_node = output_nodes[0]
    if not isinstance(out_tensor_node, torch.fx.Node):
        return None

    return [
        in_tensor_node,
        quantized_weight,
        quant_scales,
        quant_zeros,
        out_tensor_node,
    ]


# wo = "weight only"
@register_pattern_replacement("torchao_wo_quantized_linear")
def create_wo_quantized_linear_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
):
    io_nodes = identify_wo_quantized_linear_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 5
    in_tensor, quantized_weight, quant_scales, quant_zeros, out_tensor = io_nodes

    quantized_weight_tensor = get_param_tensor(ep, quantized_weight)
    if not isinstance(quantized_weight_tensor, torch.Tensor):
        return
    packed_quantized_weight_tensor = pack_4bit_weight_tensor(quantized_weight_tensor)
    utils.update_program_state_dict(
        ep, quantized_weight.name, packed_quantized_weight_tensor
    )
    quantized_weight.meta["val"] = quantized_weight.meta["val"][:, ::2].to(torch.uint8)

    quant_scales_tensor = get_param_tensor(ep, quant_scales)
    quant_zeros_tensor = get_param_tensor(ep, quant_zeros)

    assert quantized_weight_tensor is not None
    assert quant_scales_tensor is not None
    assert quant_zeros_tensor is not None

    group_size = quantized_weight_tensor.shape[1] // quant_scales_tensor.shape[1]

    combined_scales_zeros_tensor = make_combined_scales_and_zeros_tensor(
        quant_scales_tensor, quant_zeros_tensor
    )

    combined_scales_zeros_name = f"{quantized_weight.name}_scales_zeros"
    graph_module.register_parameter(
        combined_scales_zeros_name, torch.nn.Parameter(combined_scales_zeros_tensor)
    )

    with graph_module.graph.inserting_before(out_tensor):
        combined_scales_zeros = graph_module.graph.get_attr(combined_scales_zeros_name)
        wo_qlinear = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_weight_int4.default,
            args=(in_tensor, quantized_weight, group_size, combined_scales_zeros, 1),
        )

    if hasattr(out_tensor, "meta") and "val" in out_tensor.meta:
        wo_qlinear.meta["val"] = out_tensor.meta["val"]

    out_tensor.replace_all_uses_with(wo_qlinear)
