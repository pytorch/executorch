# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassBase, PassResult

REDUCTION_OPS = {
    exir_ops.edge.aten.cumsum.default,
    exir_ops.edge.aten.prod.default,
    exir_ops.edge.aten.prod.dim_int,
}


class CastReductionInputPass(PassBase):
    """Cast the input of cumsum and prod to the dtype of their result.

    PyTorch reduces bool and integer tensors in int64, but OpenVINO keeps a
    narrower type for these two, though not for sum. A cumsum over a bool mask
    is computed in 8 bits and wraps after 255, and a cumsum or prod over int32
    returns int32.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in REDUCTION_OPS:
                continue
            source = node.args[0]
            dtype = node.meta["val"].dtype
            if source.meta["val"].dtype == dtype:
                continue
            with graph_module.graph.inserting_before(node):
                cast = graph_module.graph.call_function(
                    exir_ops.edge.aten._to_copy.default, (source,), {"dtype": dtype}
                )
            cast.meta["val"] = source.meta["val"].to(dtype)
            node.replace_input_with(source, cast)
            modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
