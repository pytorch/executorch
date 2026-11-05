# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta


class ConvertFullLikeToFull(ExportPass):
    """
    Replace a full_like of a static shape with a full. full_like reads only
    the shape of its input, and when it is a delegate's only output, QNN
    drops the ops that only feed it together with the delegate input, which
    ExecuTorch still passes. A full reads nothing, so those ops are removed
    here instead. I64toI32 casts every int64 full to int32 but not a
    full_like, so an int64 full_like is kept.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.target != exir_ops.edge.aten.full_like.default:
                continue
            val = node.meta["val"]
            if val.dtype == torch.int64 or not all(
                isinstance(dim, int) for dim in val.shape
            ):
                continue
            with graph.inserting_before(node):
                full = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.full.default,
                    (list(val.shape), node.args[1]),
                    {"dtype": val.dtype},
                )
            full.meta = copy_meta(node.meta)
            node.replace_all_uses_with(full)
            graph.erase_node(node)
            modified = True
        if modified:
            dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, modified)
