# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import _operator
from typing import Dict, Tuple

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx.experimental.proxy_tensor import make_fx

from .utils import merge_decomposed_graph


class DecomposeWrapWithAutocast(ExportPass):
    """
    Decompose the _higher_order_ops WrapWithAutocast
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_submod(
        self, gm: torch.fx.GraphModule, node: torch.fx.Node
    ) -> Tuple[torch.fx.GraphModule, str]:
        for a in node.args:
            if isinstance(a, torch.fx.Node) and "submod" in a.target:
                return getattr(gm, a.target), a.target

    def _replace_output(
        self, wwac_node: torch.fx.Node, output_node: torch.fx.Node, remap: Dict
    ):
        for user in wwac_node.users.copy():
            arg_idx = 0
            is_user_getitem = False

            if user.target == _operator.getitem:
                arg_idx = user.args[1]
                is_user_getitem = True

            user.replace_input_with(
                wwac_node,
                remap[output_node.args[0][arg_idx]],
            )

            if is_user_getitem:
                for user_user in user.users.copy():
                    user_user.replace_input_with(user, user.args[0])

    def _replace(self, gm: torch.fx.GraphModule) -> None:
        graph = gm.graph
        for node in graph.nodes:
            if isinstance(node.target, torch._higher_order_ops.wrap.WrapWithAutocast):
                submod, _ = self._get_submod(gm, node)
                n_args = node.args
                device_type, dtype, enabled, cache_enabled = n_args[:4]
                if enabled:
                    # Retrace the region under its autocast, as torch.export does,
                    # so the casts autocast applies become ops. Nested regions are
                    # traced through with their own autocast.
                    with torch.autocast(device_type, dtype, enabled, cache_enabled):
                        submod = make_fx(submod, tracing_mode="fake")(
                            *[arg.meta["val"] for arg in n_args[5:]]
                        )
                else:
                    # Inline nested regions first. Their submodule names only make
                    # sense inside this submodule.
                    self._replace(submod)
                input_submod = n_args[4]
                decomposed_module = submod
                with graph.inserting_before(node):
                    # remap maps each placeholder of the submodule to the operand
                    # passed in its position. Their names need not match.
                    placeholders = [
                        n for n in submod.graph.nodes if n.op == "placeholder"
                    ]
                    remap = {p.name: arg for p, arg in zip(placeholders, n_args[5:])}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                        output_processor=self._replace_output,
                    )
                    graph.erase_node(node)

                graph.erase_node(input_submod)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._replace(graph_module)
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
