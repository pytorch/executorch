# pyre-strict

import torch

from executorch.exir.pass_base import ExportPass, PassResult


class RemoveAssertAsyncPass(ExportPass):
    """
    Temporary pass to remove all the assert async ops until runtime decides to address it.
    """

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op == torch.ops.aten._assert_async.msg:
            return
        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        gm, modified = super().call(graph_module)
        gm.graph.eliminate_dead_code()
        return PassResult(gm, modified)
