# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassBase, PassResult
from torch import fx


def _has_non_unit_leading_dim(shape: list[int]) -> bool:
    return 2 <= len(shape) <= 3 and shape[0] > 1 and all(d == 1 for d in shape[1:-1])


class CanonicalizeLinearInputRank(PassBase):
    """View a ``fused_quant.linear`` input of shape ``[N, K]`` or ``[N, 1, K]``
    (N > 1) as ``[1, N, K]`` and view the output back to its original shape, so
    the input is rank 3 with the unit dim leading.

    Only linears whose users are all views are rewritten, so no linear →
    elementwise fusion is lost.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        ):
            inp = get_arg(node, "inp", fx.Node)
            inp_val = inp.meta["val"]
            shape = list(inp_val.shape)
            if not _has_non_unit_leading_dim(shape):
                continue
            if not node.users or any(
                user.target != exir_ops.edge.aten.view_copy.default
                for user in node.users
            ):
                continue

            users = list(node.users)
            out_val = node.meta["val"]
            canonical_shape = [1, shape[0], shape[-1]]
            with graph.inserting_before(node):
                canonical = graph.call_function(
                    exir_ops.edge.aten.view_copy.default, args=(inp, canonical_shape)
                )
            canonical.meta = inp.meta.copy()
            canonical.meta["val"] = inp_val.view(canonical_shape)
            set_arg(node, "inp", canonical)

            with graph.inserting_after(node):
                restored = graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(node, list(out_val.shape)),
                )
            restored.meta = node.meta.copy()
            for user in users:
                user.replace_input_with(node, restored)
            node.meta["val"] = out_val.view(1, shape[0], out_val.shape[-1])
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
