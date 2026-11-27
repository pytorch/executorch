# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from executorch.backends.vulkan.utils import kv_cache_position_source
from executorch.exir.pass_base import ExportPass, PassResult

_POSITION_ARG_IDX = 3


class ResolveKVCachePositionPass(ExportPass):
    """
    Point the position argument of each kvcache::update_and_attend at the graph
    input it is read from, so the runtime finds the step's positions in that
    input's staging buffer before it runs the graph.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or not hasattr(node.target, "name"):
                continue
            if node.target.name() != "kvcache::update_and_attend":
                continue
            position = node.args[_POSITION_ARG_IDX]
            source = kv_cache_position_source(position)
            if source is None:
                raise RuntimeError(
                    f"kvcache::update_and_attend: position {position} is not a "
                    "reshape or leading select of a graph input"
                )
            node.update_arg(_POSITION_ARG_IDX, source)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
