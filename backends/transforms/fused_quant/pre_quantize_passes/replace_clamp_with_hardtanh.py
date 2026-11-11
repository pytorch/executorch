# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ReplaceClampWithHardtanh(PassBase):
    """Replace ``aten.clamp`` with an equivalent ``aten.hardtanh``.

    ``clamp(x, min, max)`` equals ``hardtanh(x, min, max)`` for a floating point
    ``x`` and scalar bounds with ``min <= max``. A clamp missing a bound, or
    with ``min > max``, is left untouched.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.clamp.default
        ):
            inp = get_arg(node, "input", fx.Node)
            min_val = get_arg(node, "min")
            max_val = get_arg(node, "max")
            if not inp.meta["val"].is_floating_point():
                continue
            if not isinstance(min_val, (int, float)) or not isinstance(
                max_val, (int, float)
            ):
                continue
            if min_val > max_val:
                continue

            with graph.inserting_before(node):
                hardtanh = graph.call_function(
                    torch.ops.aten.hardtanh.default,
                    args=(inp, float(min_val), float(max_val)),
                )
            hardtanh.meta = node.meta.copy()
            node.replace_all_uses_with(hardtanh)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
