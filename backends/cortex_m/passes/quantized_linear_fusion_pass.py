# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import executorch.backends.cortex_m.ops.operators  # noqa

import torch
import torch.fx
from executorch.backends.cortex_m.passes.passes_utils import quantize_multiplier_aot

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult


class QuantizedLinearFusionPass(XNNPACKPass):
    """
    Cortex-M backend pass that fuses quantized linear-like patterns.
    Fuses: dequantize -> [linear/addmm/fc_ops] -> quantize
    Into: cortex_m.quantized_linear.default with direct parameters.

    Note that the optimzed implementation makes use of the following rewrite:

    Let
    - yi be the output activations (y1, ... yn)
    - xj be the input activations (x1, ... xm)
    - wij be the weights (w11, ... wnm)
    - a be the input offset
    - b be the weight offset
    - ci be the bias

    Then the linear operation can be written as:
    yi = sum_j((xj + a) * (wij + b)) + ci
       = sum_j(xj*wij + xj*b + a*wij + a*b) + ci
       = sum_j(xj*wij) + sum_j(xj)*b + (a * sum_j(wij + b) + ci)
       = sum_j(xj*wij) + sum_j(xj)*b + kernel_sum

    where kernel_sum is precomputed aot.
    """

    def _compute_kernel_sum(self, weights, bias, input_offset, weight_offset):
        """
        Computes the precomputed kernel sum term (bias optional)
            a * sum_j(wij + b) + ci

        as defined above, for i = (1, ..., n) where j indexes the input activations.
        """
        weights_transposed = weights.T
        weights_int32 = weights_transposed.to(torch.int32)
        offset_weights = weights_int32 + weight_offset
        kernel_sum = torch.sum(offset_weights, dim=0, keepdim=True, dtype=torch.int32)
        kernel_sum_offset = kernel_sum * input_offset

        if bias is not None:
            kernel_sum_offset += bias

        return kernel_sum_offset

    def _get_linear_replacement(self, args, meta, node):
        input_scale = meta["input_qparams"][0].scale
        input_zp = meta["input_qparams"][0].zp
        weight_scale = meta["input_qparams"][1].scale
        weight_zp = meta["input_qparams"][1].zp
        output_scale = meta["output_qparams"][0].scale
        output_zp = meta["output_qparams"][0].zp
        output_min = meta["output_qparams"][0].qmin
        output_max = meta["output_qparams"][0].qmax

        quantized_multiplier, quantized_shift = quantize_multiplier_aot(
            (input_scale * weight_scale) / output_scale
        )

        # TODO: Add support for configuring the backend to support other extensions.
        # Kernel sum is only used in the CMSIS-NN implementation for the MVE extension,
        # so this should be optional.
        weights = args[1]
        weights_tensor = get_param_tensor(self.exported_program, weights)
        bias_tensor = (
            get_param_tensor(self.exported_program, args[2]) if len(args) > 2 else None
        )
        kernel_sum_tensor = self._compute_kernel_sum(
            weights_tensor, bias_tensor, -input_zp, -weight_zp
        )
        with node.graph.inserting_after(weights):
            kernel_sum = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_kernel_sum",
                InputKind.PARAMETER,
                kernel_sum_tensor,
            )

        args = (
            args[0],
            weights,
            None,
            kernel_sum,
            -input_zp,
            -weight_zp,
            output_zp,
            [quantized_multiplier],
            [quantized_shift],
            output_max,
            output_min,
        )

        return args

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.linear.default:
                continue
            if (
                node.meta.get("input_qparams", {}) == {}
                or node.meta.get("output_qparams", {}) == {}
            ):
                continue

            args = self._get_linear_replacement(node.args, node.meta, node)
            with graph_module.graph.inserting_before(node):
                cortex_m_linear = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.cortex_m.quantized_linear.default,
                    args=args,
                    kwargs={},
                )

                node.replace_all_uses_with(cortex_m_linear)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
