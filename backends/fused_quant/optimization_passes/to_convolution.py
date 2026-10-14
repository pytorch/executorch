# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import executorch.backends.fused_quant.ops  # noqa: F401
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ToConvolution(PassBase):
    """Canonicalize fused conv1d/2d/3d ops to fused_quant.convolution."""

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for target, spatial_dims in (
            (exir_ops.edge.fused_quant.conv1d.default, 1),
            (exir_ops.edge.fused_quant.conv2d.default, 2),
            (exir_ops.edge.fused_quant.conv3d.default, 3),
        ):
            for node in list(graph.find_nodes(op="call_function", target=target)):
                values = [get_arg(node, arg.name) for arg in target._schema.arguments]
                with graph.inserting_before(node):
                    convolution = graph.call_function(
                        exir_ops.edge.fused_quant.convolution.default,
                        args=(
                            *values[:-1],
                            False,
                            [0] * spatial_dims,
                            values[-1],
                        ),
                    )
                    convolution.meta = node.meta.copy()
                node.replace_all_uses_with(convolution)
                graph.erase_node(node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
