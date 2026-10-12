# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch
from executorch.backends.vulkan.op_registry import (
    can_rewrite_batch_norm_as_group_norm,
    unwrap_identity_repeat,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.export import ExportedProgram


class ReplaceInstanceNormPass(ExportPass):
    """
    Replace ``aten._native_batch_norm_legit.no_stats`` with
    ``aten.native_group_norm`` using one group per channel.

    Without this, an instance norm is a graph break. Architectures that normalize
    in each block (the fast neural style transformer nets, for one) then copy their
    activations out to the CPU and back once per block, which costs far more than
    the normalization itself.

    Only the cases can_rewrite_batch_norm_as_group_norm() accepts are rewritten:
    a rank-4, static-shape, batch-1 input with weight and bias that are constants,
    as in ``nn.InstanceNorm2d(affine=True)``. Everything else stays on the CPU,
    including the ``nn.InstanceNorm2d`` default ``affine=False``, a batch above 1
    and computed or runtime affine args.
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
            # The predicate only accepts static shapes.
            batches, channels, height, width = input_val.shape

            with graph_module.graph.inserting_before(node):
                group_norm_node = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.native_group_norm.default,
                    args=(
                        input_node,
                        # F.instance_norm repeats the affine args once per batch.
                        # For a batch of 1 that repeat is an identity: use its
                        # source, so the constant is prepacked, and let DCE drop it.
                        unwrap_identity_repeat(node.args[1]),  # weight
                        unwrap_identity_repeat(node.args[2]),  # bias
                        batches,
                        channels,
                        height * width,
                        channels,  # one group per channel
                        node.args[5],  # eps
                    ),
                )

            out_val, _, _ = node.meta["val"]
            mean_val = input_val.new_empty((batches, channels))
            rstd_val = input_val.new_empty((batches, channels))
            group_norm_node.meta = dict(node.meta)
            group_norm_node.meta["val"] = (out_val, mean_val, rstd_val)

            node.replace_all_uses_with(group_norm_node)
            modified = True

        if modified:
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)

        # The partitioner only delegates the batch norms this pass can rewrite, and
        # the Vulkan runtime has no batch norm kernel without stats. A leftover one
        # would abort when the model loads, so fail the export instead.
        remaining = [
            node
            for node in graph_module.graph.nodes
            if node.target == exir_ops.edge.aten._native_batch_norm_legit.no_stats
        ]
        if remaining:
            raise RuntimeError(
                "Vulkan cannot run _native_batch_norm_legit.no_stats nodes that "
                f"ReplaceInstanceNormPass does not rewrite: {remaining}"
            )

        return PassResult(graph_module, modified)
