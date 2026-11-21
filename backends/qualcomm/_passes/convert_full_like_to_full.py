# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

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
    here instead. A full_like the partitioner skips stays on CPU, so it is
    kept, and I64toI32 still finds it in the skip sets. A full is always
    contiguous, so a full_like of another dim order is kept too.
    """

    def __init__(
        self,
        skip_node_id_set: Optional[set] = None,
        skip_node_op_set: Optional[set] = None,
    ):
        super().__init__()
        self.skip_node_id_set = skip_node_id_set or set()
        self.skip_node_op_set = skip_node_op_set or set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if (
                node.target != exir_ops.edge.aten.full_like.default
                or node.name in self.skip_node_id_set
                or node.target.__name__ in self.skip_node_op_set
            ):
                continue
            val = node.meta["val"]
            if not all(
                isinstance(dim, int) for dim in val.shape
            ) or val.dim_order() != tuple(range(val.dim())):
                continue
            with graph.inserting_before(node):
                full = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.full.default,
                    (list(val.shape), node.args[1]),
                    {"dtype": val.dtype, "device": val.device},
                )
            full.meta = copy_meta(node.meta)
            node.replace_all_uses_with(full)
            graph.erase_node(node)
            modified = True
        if modified:
            dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, modified)
