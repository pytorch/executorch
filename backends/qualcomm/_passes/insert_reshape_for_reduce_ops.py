# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class InsertReshapeForReduceOps(ExportPass):
    """
    Rewrite `aten.argmax.default` with `dim=None` into
    a reshape-to-1D followed by argmax(dim=0).

    PyTorch semantics:
      torch.argmax(x, dim=None) -> flatten(x) then argmax along axis=0

    QNN requires an explicit axis, so we insert the reshape.
    """

    def __init__(self):
        super().__init__()
        self.op_map = {torch.ops.aten.argmax.default, torch.ops.aten.argmin.default}

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        modified = False

        for n in graph.nodes:
            if n.target in self.op_map:
                dim_arg = None if len(n.args) == 1 else n.args[1]

                if dim_arg is None:
                    inp = n.args[0]

                    # Insert reshape before argmax
                    with graph.inserting_before(n):
                        reshape_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.reshape.default,
                            (inp, [-1]),
                            {},
                        )
                        reshape_node.meta = dict(inp.meta)
                        if "val" in inp.meta:
                            reshape_node.meta["val"] = inp.meta["val"].reshape(-1)

                    # Rewrite argmax: take reshape_node as input, set dim=0
                    n.args = (reshape_node, 0, *n.args[2:])

                modified = True

        if modified:
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)

        return PassResult(graph_module, modified)
