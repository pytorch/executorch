# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.backends.nxp.backend.edge_helper import input_rank

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class AddBatchSizeFor3DAvgPool2DAndMaxPool2D(PassBase):
    """Adds batch size dimension for aten.avgpool2d and aten.maxpool2d.default ops with 3D input,
    as the Neutron Converter is unable to convert these ops with 3D input.

                                                                   │
                                                            ┌──────▼──────┐
                                                            │   reshape   │
                                                            │ (add batch) │
                                                            └──────┬──────┘
             │                                                     │
      ┌──────▼──────┐            replace with               ┌──────▼──────┐
      │ avgpool2d/  │           ──────────────►             │ avgpool2d/  │
      │ maxpool2d   │                                       │ maxpool2d   │
      │ (3D input)  │                                       │ (4D input)  │
      └──────┬──────┘                                       └──────┬──────┘
             │                                                     │
                                                            ┌──────▼───────┐
                                                            │   reshape    │
                                                            │(remove batch)│
                                                            └──────┬───────┘
                                                                   │
    """

    def call(self, module: GraphModule) -> bool:
        def _is_3d_pool(node_: Node) -> bool:
            if node_.op != "call_function":
                return False

            if node_.target not in [
                torch.ops.aten.avg_pool2d.default,
                torch.ops.aten.max_pool2d.default,
            ]:
                return False

            # Check if input is 3D (C, H, W)
            rank = input_rank(node_, 0)
            return rank == 3

        made_changes = False

        for node in module.graph.nodes:
            if not _is_3d_pool(node):
                continue

            pool_node = node
            input_node = pool_node.args[0]

            # Get input shape (C, H, W)
            input_shape = input_node.meta["val"].shape

            # Get output shape (3D) before we modify metadata
            output_shape_3d = pool_node.meta["val"].shape

            # Insert reshape to add batch dimension (1, C, H, W)
            with module.graph.inserting_before(pool_node):
                reshape_add_batch = module.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(input_node, [1, *input_shape]),
                    kwargs={},
                )
                reshape_add_batch.meta["source_fn_stack"] = pool_node.meta.get(
                    "source_fn_stack", []
                )
                reshape_add_batch.meta["val"] = input_node.meta["val"].unsqueeze(0)

            # Update pool_node to use 4D input
            pool_node.args = (reshape_add_batch, *pool_node.args[1:])

            # Update pool_node output metadata to 4D
            pool_node.meta["val"] = pool_node.meta["val"].unsqueeze(0)

            # Insert reshape to remove batch dimension AFTER pool_node
            with module.graph.inserting_after(pool_node):
                reshape_remove_batch = module.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(pool_node, list(output_shape_3d)),
                    kwargs={},
                )
                reshape_remove_batch.meta["source_fn_stack"] = pool_node.meta.get(
                    "source_fn_stack", []
                )
                reshape_remove_batch.meta["val"] = pool_node.meta["val"].squeeze(0)

            # Replace all uses of pool_node with reshape_remove_batch (except reshape_remove_batch itself)
            pool_node.replace_all_uses_with(reshape_remove_batch)
            # Restore the connection: reshape_remove_batch should use pool_node as input
            reshape_remove_batch.update_arg(0, pool_node)

            made_changes = True

        return PassResult(module, made_changes)
