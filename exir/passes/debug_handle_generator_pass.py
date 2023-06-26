from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class DebugHandleGeneratorPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        """Lower a quantized reference model (with reference quantized operator patterns)
        to executorch backend, that has a canonical set of quantized operators
        """
        for index, node in enumerate(graph_module.graph.nodes):
            node.meta["debug_handle"] = index
        return PassResult(graph_module, True)
