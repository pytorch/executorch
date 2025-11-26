# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401

import torch
from executorch.backends.arm._passes.quant_args import QuantArgs

from executorch.backends.cortex_m.passes.passes_utils import quantize_val

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class DecomposeHardswishPass(ExportPass):
    """
    Decomposes hardswish like

        hardswish(x) = x * (clamp(x, -3, 3) + 3)/6

    where the add and division is implemented by modifying the quantization parameters similar
    to hardsigmoid in the activation_fusion_pass. Note that this pass assumes
    that the output range of the preceding op is already clamped to [-3, inf] during
    quantization by the clamp_hardswish_pass, removing the need for the negative clamp.
    """

    TARGETS = {
        exir_ops.edge.aten.hardswish.default,
    }

    FUSE_OPS = {
        exir_ops.edge.aten.linear.default,
        exir_ops.edge.aten.convolution.default,
    }

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list[Node] = []

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in self.TARGETS:
                continue

            input_node = node.args[0]
            if (
                input_node.op != "call_function"
                or input_node.target not in self.FUSE_OPS
            ):
                logger.warning(
                    f"Cannot fuse activation {node.name} as input node {input_node.name} is not a supported fused activation op."
                )
                continue
            if len(input_node.users.values()) > 1:
                logger.warning(
                    f"Cannot fuse activation {node.name} as input node {input_node.name} has multiple users."
                )
                continue

            input_quant_dict = input_node.meta.get("output_qparams", [None])[
                0
            ]._asdict()
            scale = input_quant_dict["scale"]
            zero_point = input_quant_dict["zp"]
            qmin = input_quant_dict["qmin"]
            qmax = input_quant_dict["qmax"]

            # Create min node
            with graph_module.graph.inserting_after(input_node):
                clamp_node = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.aten.minimum.default,
                    args=(
                        input_node,
                        torch.tensor(
                            quantize_val(3, scale, zero_point, qmin, qmax),
                            dtype=torch.int8,
                        ),
                    ),
                    kwargs={},
                )
                clamp_node.meta = input_node.meta.copy()

            # Create mul node
            with graph_module.graph.inserting_after(clamp_node):
                mul_node = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.aten.mul.Tensor,
                    args=(input_node, clamp_node),
                    kwargs={},
                )
                mul_node.meta = node.meta.copy()

            mul_quant_dict = node.meta["input_qparams"][0]._asdict()

            mul_quant_dict_shifted = mul_quant_dict.copy()
            mul_quant_dict_shifted["zp"] = mul_quant_dict_shifted["zp"] - round(
                3 / (mul_quant_dict_shifted["scale"])
            )

            output_quant_dict = node.meta["output_qparams"][0]._asdict()
            output_quant_dict["scale"] = output_quant_dict["scale"] * 6

            node.meta["input_qparams"][0] = QuantArgs(**mul_quant_dict)
            mul_node.meta["input_qparams"][1] = QuantArgs(**mul_quant_dict_shifted)
            mul_node.meta["output_qparams"][0] = QuantArgs(**output_quant_dict)

            node.replace_all_uses_with(mul_node)
            nodes_to_erase.append(node)
            modified = True

        for node in nodes_to_erase:
            graph_module.graph.erase_node(node)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
