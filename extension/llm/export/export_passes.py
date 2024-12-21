import torch
from torch._subclasses import FakeTensor

from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult

def _normalize_dims(tensor: FakeTensor, dim_0: int, dim_1: int):
    """
    Normalize the dimensions of a tensor.
    """
    assert tensor is not None, "Tensor is None"
    ndim = tensor.ndim
    if dim_0 < 0:
        dim_0 = ndim + dim_0
    if dim_1 < 0:
        dim_1 = ndim + dim_1
    assert dim_0 < ndim and dim_1 < ndim, f"Invalid dimensions: {dim_0}, {dim_1}"
    return dim_0, dim_1

class RemoveRedundantTransposes(ExportPass):
    """
    This pass removes redundant transpose nodes in the graph.
    It checks if the next node is also a transpose node and if the two transpose nodes undo each other.
    For example, if the graph has the following nodes:

    node1 = torch.ops.aten.transpose.int(x, 0, 1)
    node2 = torch.ops.aten.transpose.int(node1, 0, 1)

    Then node2's use can be replaced by x

    It will also check for permute nodes
    node1 = torch.ops.aten.permute(x, [0, 2, 1])
    node2 = torch.ops.aten.permute(node1, [0, 2, 1])

    Then also node2's use can be replaced by x

    NB: Does not work for inplace ops or functionalized _copy suffix ops
    """
    def call(self, graph_module: torch.fx.GraphModule):
        graph_changed = False
        for node in graph_module.graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.transpose.int:
                # Check if the next node is also a transpose node
                tranpose_users = list(node.users.keys())
                dim_0 = node.args[1]
                dim_1 = node.args[2]
                dim_0, dim_1 = _normalize_dims(node.args[0].meta["val"], dim_0, dim_1)

                for user in tranpose_users:
                    if user.op == 'call_function' and user.target == torch.ops.aten.transpose.int:
                        # Get the arguments of the current and next transpose nodes
                        user_dim_0 = user.args[1]
                        user_dim_1 = user.args[2]
                        user_dim_0, user_dim_1 = _normalize_dims(user.args[0].meta["val"], user_dim_0, user_dim_1)
                        
                        # Check if the two transpose nodes undo each other
                        if dim_0 == user_dim_0 and dim_1 == user_dim_1:
                            graph_changed = True
                            user.replace_all_uses_with(node.args[0])

        for node in graph_module.graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.permute.default:
                # Check if the next node is also a transpose node
                permute_users = list(node.users.keys())
                dim_list = node.args[1]

                for user in permute_users:
                    if user.op == 'call_function' and user.target == torch.ops.aten.permute.default:
                        # Get the arguments of the current and next transpose nodes
                        user_dim_list = user.args[1]

                        # Check if the two permutes undo each other
                        if dim_list == user_dim_list:
                            graph_changed = True
                            user.replace_all_uses_with(node.args[0])

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, graph_changed)
