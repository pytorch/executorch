# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import compute_meta_val
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult

_ADAPTIVE_AVG_POOL2D = (
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
)


class ReplaceAdaptiveAvgPoolWithAvgPool(PassBase):
    """Replace ``adaptive_avg_pool2d`` with an equivalent ``aten.avg_pool2d``.

    When each spatial input size is a multiple of the output size, every adaptive
    bin is the same ``in // out`` window with no overlap, so the op is exactly
    ``avg_pool2d(kernel=stride=in // out)``. Any other shape is left untouched.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for target in _ADAPTIVE_AVG_POOL2D:
            for node in graph.find_nodes(op="call_function", target=target):
                input_val = get_arg(node, "input", fx.Node).meta["val"]
                output_size = get_arg(node, "output_size", list[int])
                if input_val.ndim != 4 or len(output_size) != 2:
                    continue
                assert all(out > 0 for out in output_size), output_size
                spatial = input_val.shape[-2:]
                if any(i % o for i, o in zip(spatial, output_size)):
                    continue
                self._replace(node, [i // o for i, o in zip(spatial, output_size)])
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    @staticmethod
    def _replace(node: fx.Node, kernel_size: list[int]) -> None:
        graph = node.graph
        inp = get_arg(node, "input", fx.Node)

        with graph.inserting_before(node):
            avg_pool = graph.call_function(
                torch.ops.aten.avg_pool2d.default,
                args=(inp, kernel_size, kernel_size, [0, 0], False, True, None),
            )
        avg_pool.meta = node.meta.copy()
        avg_pool.meta["val"] = compute_meta_val(avg_pool)

        node.replace_all_uses_with(avg_pool)
        graph.erase_node(node)
