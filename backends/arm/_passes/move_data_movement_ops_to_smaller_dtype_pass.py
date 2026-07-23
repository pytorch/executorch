# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass_utils import refresh_permute_view_meta
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .arm_pass import ArmPass


class MoveDataMovementOpsToSmallerDtypePass(ArmPass):
    """Move layout operations next to rescales onto the smaller element type."""

    _RESCALE = exir_ops.backend.tosa.RESCALE.default
    _passes_required_after: Set[Type[ExportPass]] = set()
    _DATA_MOVEMENT_OPS = {
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        iteration_modified = True
        while iteration_modified:
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                if node.target not in self._DATA_MOVEMENT_OPS:
                    continue
                input_nodes = node.all_input_nodes
                if len(input_nodes) != 1:
                    continue

                producer = input_nodes[0]
                if producer.target == self._RESCALE and self._uses_wider_dtype(
                    node, producer, producer.all_input_nodes[0]
                ):
                    self._move_before_rescale(node, producer)
                    iteration_modified = True
                    break

                narrowing_rescale = next(
                    (
                        user
                        for user in node.users
                        if user.target == self._RESCALE
                        and self._uses_wider_dtype(node, user, user)
                    ),
                    None,
                )
                if narrowing_rescale is not None:
                    if len(node.users) == 1:
                        self._move_after_rescale(node, narrowing_rescale)
                    else:
                        self._split_after_rescale(node, narrowing_rescale)
                    iteration_modified = True
                    break
            modified |= iteration_modified

        if modified:
            graph_module.graph.lint()
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()
        return PassResult(graph_module, modified)

    def _uses_wider_dtype(
        self,
        data_movement: torch.fx.Node,
        rescale: torch.fx.Node,
        smaller_side: torch.fx.Node,
    ) -> bool:
        if not self._is_per_tensor_rescale(rescale):
            return False
        movement_val = data_movement.meta.get("val")
        smaller_val = smaller_side.meta.get("val")
        return (
            isinstance(movement_val, torch.Tensor)
            and isinstance(smaller_val, torch.Tensor)
            and movement_val.element_size() > smaller_val.element_size()
        )

    def _is_per_tensor_rescale(self, rescale: torch.fx.Node) -> bool:
        if len(rescale.args) < 3 or len(rescale.all_input_nodes) != 1:
            return False
        special_dtype_key = TosaSpecialDtype.meta_key()
        if rescale.all_input_nodes[0].meta.get(special_dtype_key) != rescale.meta.get(
            special_dtype_key
        ):
            return False
        scales = rescale.args[2]
        return not isinstance(scales, Sequence) or len(scales) == 1

    def _move_before_rescale(
        self, data_movement: torch.fx.Node, rescale: torch.fx.Node
    ) -> None:
        producer = rescale.all_input_nodes[0]
        movement_users = list(data_movement.users)
        data_movement.replace_input_with(rescale, producer)
        rescale.replace_input_with(producer, data_movement)
        for user in movement_users:
            user.replace_input_with(data_movement, rescale)
        data_movement.append(rescale)

        old_movement_val = cast(torch.Tensor, data_movement.meta["val"])
        producer_val = cast(torch.Tensor, producer.meta["val"])
        data_movement.meta = dict(data_movement.meta)
        data_movement.meta["val"] = old_movement_val.new_empty(
            old_movement_val.shape, dtype=producer_val.dtype
        )
        rescale.meta = dict(rescale.meta)
        rescale.meta["val"] = old_movement_val

    def _move_after_rescale(
        self, data_movement: torch.fx.Node, rescale: torch.fx.Node
    ) -> None:
        producer = data_movement.all_input_nodes[0]
        rescale_users = list(rescale.users)
        rescale.replace_input_with(data_movement, producer)
        data_movement.replace_input_with(producer, rescale)
        for user in rescale_users:
            user.replace_input_with(rescale, data_movement)
        rescale.append(data_movement)

        old_rescale_val = cast(torch.Tensor, rescale.meta["val"])
        producer_val = cast(torch.Tensor, producer.meta["val"])
        rescale.meta = dict(rescale.meta)
        rescale.meta["val"] = producer_val.new_empty(
            producer_val.shape, dtype=old_rescale_val.dtype
        )
        refresh_permute_view_meta(data_movement)

    def _split_after_rescale(
        self, data_movement: torch.fx.Node, rescale: torch.fx.Node
    ) -> None:
        producer = data_movement.all_input_nodes[0]
        rescale_users = list(rescale.users)
        rescale.replace_input_with(data_movement, producer)

        with rescale.graph.inserting_after(rescale):
            branch_movement = rescale.graph.call_function(
                cast(Any, data_movement.target),
                args=(rescale, *data_movement.args[1:]),
                kwargs=dict(data_movement.kwargs),
            )
        branch_movement.meta = dict(data_movement.meta)
        refresh_permute_view_meta(branch_movement)
        for user in rescale_users:
            user.replace_input_with(rescale, branch_movement)

        if not data_movement.users:
            data_movement.graph.erase_node(data_movement)
