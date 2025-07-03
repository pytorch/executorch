# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_quant,
    tag_as_implicit_q_dq,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DecomposeConcatenate(ExportPass):
    """
    XNNPACK's Concatenate operation only supports concatenation for <= 5 tensors
    at a time. As a result, to support concatenates with > 5 tensors, we can decompose
    concatenates into sequences of cats each with <= 5 tensors.

    Example:
    Before Pass:
        cat: "f32" = torch.ops.aten.cat.default([t1, t2, t3, t4, t5, t6], 1);

    After Pass:
        cat: "f32" = torch.ops.aten.cat.default([t1, t2, t3, t4, t5], 1);
        cat_1: "f32" = torch.ops.aten.cat.default([cat, t6], 1);
    """

    def call(self, graph_module: torch.fx.GraphModule):
        gm = graph_module
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target.__name__ == "aten.cat.default"
            ):
                concat_args = node.args
                nodes_to_concat = node.args[0]
                if len(nodes_to_concat) <= 5:
                    continue

                is_quantized = all(
                    is_dequant(node) for node in nodes_to_concat
                ) and all(is_quant(node) for node in node.users.keys())

                # replace the cat args with the same args but only with the first 5 nodes
                new_concat_args = (nodes_to_concat[:5],) + concat_args[1:]
                node.args = new_concat_args

                remainder_nodes_to_concat = nodes_to_concat[5:]
                with gm.graph.inserting_after(node):
                    logger.debug(f"Decomposing cat node {node}")
                    remainder_concat_node = gm.graph.create_node(
                        "call_function",
                        target=exir_ops.edge.aten.cat.default,
                        args=([],),  # we will replace this remainder_nodes later
                        kwargs=node.kwargs,
                    )
                    node.replace_all_uses_with(remainder_concat_node)
                if is_quantized:
                    # if quantized we need to enforce the q/dq pattern for the newly inserted
                    # concat node
                    q_params = nodes_to_concat[0].args[1:]
                    q_kwargs = nodes_to_concat[0].kwargs
                    # Quantizer enforces all the inputs and output to a concat node must share
                    # the same qparams, this means the newly inserted q/dq pair must share the
                    # same qparams as the first quantized input in the concat node.
                    with gm.graph.inserting_after(node):
                        logger.debug(
                            f"Inserting Q/DQ pair for new cat node {remainder_concat_node}"
                        )
                        q_node = gm.graph.create_node(
                            "call_function",
                            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                            args=(node,) + q_params,
                            kwargs=q_kwargs,
                        )
                        tag_as_implicit_q_dq(q_node)
                    with gm.graph.inserting_after(q_node):
                        dq_node = gm.graph.create_node(
                            "call_function",
                            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                            args=(q_node,) + q_params,
                            kwargs=q_kwargs,
                        )
                        tag_as_implicit_q_dq(dq_node)
                    remainder_concat_node.args = (
                        [dq_node] + remainder_nodes_to_concat,
                    ) + node.args[1:]
                else:
                    remainder_concat_node.args = (
                        [node] + remainder_nodes_to_concat,
                    ) + node.args[1:]

        gm.recompile()
        new_gm = super().call(gm).graph_module
        return PassResult(new_gm, True)
