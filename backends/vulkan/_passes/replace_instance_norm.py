# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch
from executorch.backends.vulkan.op_registry import can_rewrite_batch_norm_as_group_norm
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.export import ExportedProgram


class ReplaceInstanceNormPass(ExportPass):
    """
    Replace ``aten._native_batch_norm_legit.no_stats`` with
    ``aten.native_group_norm`` using one group per channel.

    Without this, every ``nn.InstanceNorm2d`` is a graph break. Architectures that
    normalize in each block (the fast neural style transformer nets, for one) then
    copy their activations out to the CPU and back once per block, which costs far
    more than the normalization itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self._exported_program: Optional[ExportedProgram] = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        assert self._exported_program is not None
        constant_nodes = utils.get_constant_nodes(self._exported_program)
        modified = False

        for node in list(graph_module.graph.nodes):
            if not can_rewrite_batch_norm_as_group_norm(node, constant_nodes):
                continue

            input_node = node.args[0]
            assert isinstance(input_node, torch.fx.Node)
            input_val = input_node.meta["val"]
            batches, channels, height, width = (int(d) for d in input_val.shape)

            with graph_module.graph.inserting_before(node):
                group_norm_node = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.native_group_norm.default,
                    args=(
                        input_node,
                        node.args[1],  # weight
                        node.args[2],  # bias
                        batches,
                        channels,
                        height * width,
                        channels,  # one group per channel
                        node.args[5],  # eps
                    ),
                )

            out_val, _, _ = node.meta["val"]
            stats_val = input_val.new_empty((batches, channels))
            group_norm_node.meta = dict(node.meta)
            group_norm_node.meta["val"] = (out_val, stats_val, stats_val)

            node.replace_all_uses_with(group_norm_node)
            modified = True

        if modified:
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)

        return PassResult(graph_module, modified)
