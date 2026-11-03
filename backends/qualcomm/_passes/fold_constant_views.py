# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FoldConstantViews(ExportPass):
    """
    Replace views and splits of a parameter with constants before annotation.

    nn.MultiheadAttention splits in_proj_weight at runtime when query and value
    differ, so each projection's weight is the output of a split rather than a
    parameter. A per-channel weight encoding on such a tensor is never folded
    into a static weight, and lowering fails. Folding gives each projection its
    own parameter.
    """

    view_ops = {
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.chunk.default,
        torch.ops.aten.unbind.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
    }

    def __init__(self):
        super(FoldConstantViews, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (
                node.target not in self.view_ops
                or not isinstance(node.args[0], torch.fx.Node)
                or node.args[0].op != "get_attr"
                or any(isinstance(a, torch.fx.Node) for a in node.args[1:])
            ):
                continue
            source = node.args[0]
            *path, attr = source.target.split(".")
            tensor = getattr(graph_module.get_submodule(".".join(path)), attr)
            outputs = node.target(tensor, *node.args[1:], **node.kwargs)
            if isinstance(outputs, torch.Tensor):
                replaced = [(node, outputs)]
            else:
                replaced = [
                    (user, outputs[user.args[1]])
                    for user in node.users
                    if user.target == operator.getitem
                ]
            for old, value in replaced:
                name = f"{source.target.replace('.', '_')}_{old.name}"
                constant = value.detach().clone().contiguous()
                if isinstance(tensor, torch.nn.Parameter):
                    constant = torch.nn.Parameter(constant, requires_grad=False)
                setattr(graph_module, name, constant)
                with graph.inserting_before(old):
                    get_attr = graph.get_attr(name)
                get_attr.meta = old.meta.copy()
                old.replace_all_uses_with(get_attr)

        dead_code_elimination_pass(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
