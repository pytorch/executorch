# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_quant,
    tag_as_implicit_q_dq,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class InsertPadQDQPass(XNNPACKPass):
    """
    Inserts implicit quantize/dequantize pairs after constant_pad_nd nodes
    that sit in a quantized context (input is a dequantize node), so the pad
    can be serialized as a quantized static pad op.

    Skips pads whose output is already quantized (idempotent).

    Without this pass, a zero-valued constant_pad_nd between a dequantize and
    a convolution would serialize as fp32 while the conv expects quantized
    activation, causing a mismatch.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.constant_pad_nd.default
            ):
                continue

            pad_input = node.args[0]
            if not (isinstance(pad_input, torch.fx.Node) and is_dequant(pad_input)):
                continue

            pad_value = cast(float, node.args[2]) if len(node.args) > 2 else 0.0
            if pad_value != 0.0:
                continue

            pad_amounts = cast(List[int], node.args[1])
            if any(p < 0 for p in pad_amounts):
                continue

            if any(is_quant(user) for user in node.users):
                continue

            q_params = pad_input.args[1:]

            with graph.inserting_after(node):
                q = graph.create_node(
                    "call_function",
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                    args=(node,) + q_params,
                )
                q.meta = node.meta.copy()
                tag_as_implicit_q_dq(q)

            with graph.inserting_after(q):
                dq = graph.create_node(
                    "call_function",
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                    args=(q,) + q_params,
                )
                dq.meta = q.meta.copy()
                tag_as_implicit_q_dq(dq)

            node.replace_all_uses_with(dq)
            q.args = (node,) + q_params

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
