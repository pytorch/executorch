# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx

from .utils import merge_decomposed_graph


class DecomposeMatmul(ExportPass):
    """
    Decompose matmul for quantization annotation to work properly.
    """

    targeted_ops: list[OpOverload] = [
        torch.ops.aten.matmul.default,
    ]

    def __init__(self) -> None:
        super().__init__()

    def _replace_output(
        self, node: torch.fx.Node, output_node: torch.fx.Node, remap: Dict
    ):
        for user in node.users.copy():
            user.replace_input_with(
                node,
                remap[output_node.args[0]],
            )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            input_tensors = []
            for arg in node.args:
                if hasattr(arg, "meta"):
                    input_tensors.append(arg.meta["val"])
                elif isinstance(arg, int):
                    input_tensors.append(arg)

            # TODO Add support for multiplication with vectors
            if (
                len(input_tensors) > 1 and input_tensors[1].dim() == 1
            ) or input_tensors[0].dim() == 1:
                continue

            decomposed_module = make_fx(
                node.target,
                tracing_mode="fake",
            )(*input_tensors)

            with graph.inserting_before(node):
                # Create mapping from placeholder names to original nodes
                remap = {"arg0_1": node.args[0], "arg1_1": node.args[1]}
                merge_decomposed_graph(
                    remap=remap,
                    target_node=node,
                    target_graph=graph,
                    decomposed_graph_module=decomposed_module,
                    output_processor=self._replace_output,
                )
                graph.erase_node(node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
