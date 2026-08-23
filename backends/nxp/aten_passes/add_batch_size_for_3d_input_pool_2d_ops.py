# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.edge_helper import input_rank

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class AddBatchSizeFor3DInputPool2DOps(PassBase):
    """Adds batch size dimension for aten.adaptive_avg_pool2d.default, aten.avg_pool2d.default
    and aten.max_pool2d.default ops with 3D input, as the Neutron Converter is unable to convert these ops with 3D input.

                                                                   │
                                                            ┌──────▼──────┐
                                                            │   reshape   │
                                                            │ (add batch) │
                                                            └──────┬──────┘
             │                                                     │
      ┌──────▼───────┐                                      ┌──────▼───────┐
      │   adaptive   │                                      │   adaptive   │
      │ _avg_pool2d/ │            replace with              │ _avg_pool2d/ │
      │ avg_pool2d/  │           ──────────────►            │ avg_pool2d/  │
      │ max_pool2d   │                                      │ max_pool2d   │
      │  3D input    │                                      │  4D input    │
      │  (C,H,W)     │                                      │  (1,C H,W)   │
      └──────┬───────┘                                      └──────┬───────┘
             │                                                     │
                                                            ┌──────▼───────┐
                                                            │   reshape    │
                                                            │(remove batch)│
                                                            └──────┬───────┘
                                                                   │
    """

    module: GraphModule

    def _create_reshape_node(self, pool_node: Node, *args) -> Node:
        reshape_node = self.module.graph.call_function(
            torch.ops.aten.reshape.default,
            args=args,
            kwargs={},
        )
        reshape_node.meta["source_fn_stack"] = pool_node.meta.get("source_fn_stack", [])
        input_node = args[0]
        if input_node == pool_node:
            # insert after pool_node
            reshape_node.meta["val"] = input_node.meta["val"].squeeze(0)
        else:
            # insert before pool_node
            reshape_node.meta["val"] = input_node.meta["val"].unsqueeze(0)
        return reshape_node

    def call(self, module: GraphModule) -> PassResult:
        self.module = module

        def _is_3d_pool(node_: Node) -> bool:
            if node_.op != "call_function":
                return False

            if node_.target not in [
                torch.ops.aten.adaptive_avg_pool2d.default,
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
                reshape_add_batch = self._create_reshape_node(
                    pool_node, input_node, [1, *input_shape]
                )

            # Update pool_node to use 4D input
            pool_node.args = (reshape_add_batch, *pool_node.args[1:])

            # Update pool_node output metadata to 4D
            pool_node.meta["val"] = pool_node.meta["val"].unsqueeze(0)

            # Insert reshape to remove batch dimension AFTER pool_node
            with module.graph.inserting_after(pool_node):
                reshape_remove_batch = self._create_reshape_node(
                    pool_node, pool_node, list(output_shape_3d)
                )

            # Replace all uses of pool_node with reshape_remove_batch (except reshape_remove_batch itself)
            pool_node.replace_all_uses_with(reshape_remove_batch)
            # Restore the connection: reshape_remove_batch should use pool_node as input
            reshape_remove_batch.update_arg(0, pool_node)

            made_changes = True

        return PassResult(module, made_changes)
