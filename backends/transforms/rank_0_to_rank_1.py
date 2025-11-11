import torch
from executorch.exir.pass_base import ExportPass, PassResult


class Rank0ToRank1Pass(ExportPass):
    """
    Replace Rank-0 Tensor to Rank-1 Tensor for all the inputs.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" and node.meta["val"].shape == ():
                node.meta["val"] = node.meta["val"].reshape(1, 1)
        graph_module.recompile()
        return PassResult(graph_module, True)
