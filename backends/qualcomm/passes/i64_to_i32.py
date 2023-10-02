# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class I64toI32(ExportPass):
    """
    Try cast unsuuported int64 datatype into int32.
    Currently supported patterns are:

    1. placeholder (int64) -> placeholder + cast (int32)
    """

    def __init__(self):
        super(I64toI32, self).__init__()

    def _create_call_function_node(
        self,
        graph_module: torch.fx.GraphModule,
        target: torch.fx.node.Target,
        args: Tuple[torch.fx.node.Argument, ...],
        kwargs: Dict[str, torch.fx.node.Argument],
    ):
        return graph_module.graph.create_node(
            "call_function",
            target=target,
            args=args,
            kwargs=kwargs,
        )

    def _insert_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.node,
    ) -> None:
        with graph_module.graph.inserting_after(node):
            users = node.users.copy()
            cast = self._create_call_function_node(
                graph_module,
                exir_ops.edge.aten._to_copy.default,
                (node,),
                {"dtype": torch.int32},
            )

            for user in users:
                user.replace_input_with(node, cast)

    def _update_meta(self, node: torch.fx.node) -> None:
        meta_val = node.meta["val"]
        if isinstance(meta_val, tuple):
            node.meta["val"] = (
                fake_tensor.to(torch.int32)
                if fake_tensor.dtype == torch.int64
                else fake_tensor
                for fake_tensor in meta_val
            )
        else:
            if meta_val.dtype == torch.int64:
                node.meta["val"] = meta_val.to(torch.int32)

    def _cast_to_int32(self, graph_module: torch.fx.GraphModule):
        for n in graph_module.graph.nodes:
            if n.target == exir_ops.edge.aten._to_copy.default:
                continue

            meta_val = n.meta["val"]

            if (n.op == "placeholder") and meta_val.dtype == torch.int64:
                self._insert_node(graph_module, n)

    def call(self, graph_module: torch.fx.GraphModule):
        self._cast_to_int32(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
