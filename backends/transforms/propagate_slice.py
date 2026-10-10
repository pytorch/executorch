# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from math import prod
from typing import Callable, cast, Optional

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class PropagateSlice(RemoveOrReplacePassInterface):
    """Propagate slice_copy before element-wise ops when the cost model
    indicates it reduces total data movement.

    Supported ops (extensible via dispatch table):
        - quantize_per_tensor: unary element-wise
        - dequantize_per_tensor: unary element-wise
        - add.Tensor: binary with broadcast — slices non-broadcasting inputs
        - mul.Tensor: binary with broadcast — slices non-broadcasting inputs

    Callers can provide per-target filters for additional legality constraints.
    Handles any slice dim and any step size.
    """

    quant_unary_targets: list[EdgeOpOverload] = [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    ]

    def __init__(
        self,
        additional_unary_targets: Optional[list[EdgeOpOverload]] = None,
        additional_binary_targets: Optional[list[EdgeOpOverload]] = None,
        target_filters: Optional[
            dict[EdgeOpOverload, Callable[[torch.fx.Node], bool]]
        ] = None,
    ) -> None:
        super().__init__()
        unary_targets = [
            *self.quant_unary_targets,
            *(additional_unary_targets or []),
        ]
        binary_targets = [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            *(additional_binary_targets or []),
        ]
        self._dispatch: dict[
            EdgeOpOverload,
            tuple[
                Callable[[torch.fx.Node, torch.fx.Node], bool],
                Callable[[torch.fx.Node, torch.fx.Node], bool],
            ],
        ] = {}
        self._target_filters = target_filters or {}
        for t in unary_targets:
            self._dispatch[t] = (
                self._should_swap_elementwise,
                self._swap_elementwise_slice,
            )

        for t in binary_targets:
            self._dispatch[t] = (
                self._should_swap_binary_elementwise,
                self._swap_binary_elementwise_slice,
            )

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.slice_copy.Tensor]

    def _should_swap_elementwise(
        self, op_node: torch.fx.Node, slice_node: torch.fx.Node
    ) -> bool:
        full_size = prod(op_node.meta["val"].shape)
        sliced_size = prod(slice_node.meta["val"].shape)
        return sliced_size < full_size

    def _swap_elementwise_slice(
        self, op_node: torch.fx.Node, slice_node: torch.fx.Node
    ) -> bool:
        op_input = op_node.args[0]
        assert isinstance(op_input, torch.fx.Node)
        graph = slice_node.graph

        slice_dim = get_arg(slice_node, "dim", int)
        slice_start = get_arg(slice_node, "start")
        slice_end = get_arg(slice_node, "end")
        slice_step = get_arg(slice_node, "step", int)

        with graph.inserting_before(op_node):
            new_slice = graph.call_function(
                exir_ops.edge.aten.slice_copy.Tensor,
                args=(op_input, slice_dim, slice_start, slice_end, slice_step),
            )
            new_slice.meta["val"] = exir_ops.edge.aten.slice_copy.Tensor(
                op_input.meta["val"], slice_dim, slice_start, slice_end, slice_step
            )

            new_args = list(op_node.args)
            new_args[0] = new_slice
            target = cast(EdgeOpOverload, op_node.target)
            new_op = graph.call_function(
                target,
                args=tuple(new_args),
                kwargs=op_node.kwargs,
            )
            new_op.meta["val"] = target(
                new_slice.meta["val"],
                *[
                    a.meta["val"] if isinstance(a, torch.fx.Node) else a
                    for a in new_args[1:]
                ],
                **{
                    k: v.meta["val"] if isinstance(v, torch.fx.Node) else v
                    for k, v in op_node.kwargs.items()
                },
            )

        slice_node.replace_all_uses_with(new_op)
        graph.erase_node(slice_node)
        graph.erase_node(op_node)
        return True

    def _should_swap_binary_elementwise(
        self, op_node: torch.fx.Node, slice_node: torch.fx.Node
    ) -> bool:
        lhs, rhs = op_node.args[0], op_node.args[1]
        assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
        if lhs.meta["val"].shape == rhs.meta["val"].shape:
            return False
        full_size = prod(op_node.meta["val"].shape)
        sliced_size = prod(slice_node.meta["val"].shape)
        return sliced_size < full_size

    def _swap_binary_elementwise_slice(
        self, op_node: torch.fx.Node, slice_node: torch.fx.Node
    ) -> bool:
        lhs, rhs = op_node.args[0], op_node.args[1]
        assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
        graph = slice_node.graph

        slice_dim = get_arg(slice_node, "dim", int)
        slice_start = get_arg(slice_node, "start")
        slice_end = get_arg(slice_node, "end")
        slice_step = get_arg(slice_node, "step", int)

        output_shape = op_node.meta["val"].shape
        output_dim = slice_dim % len(output_shape)

        new_args = list(op_node.args)
        with graph.inserting_before(op_node):
            for i, inp in enumerate([lhs, rhs]):
                input_shape = inp.meta["val"].shape
                # Broadcasting aligns operand dimensions to the right of the output.
                input_dim = output_dim - (len(output_shape) - len(input_shape))
                if (
                    input_dim >= 0
                    and input_shape[input_dim] == output_shape[output_dim]
                ):
                    new_slice = graph.call_function(
                        exir_ops.edge.aten.slice_copy.Tensor,
                        args=(inp, input_dim, slice_start, slice_end, slice_step),
                    )
                    new_slice.meta["val"] = exir_ops.edge.aten.slice_copy.Tensor(
                        inp.meta["val"],
                        input_dim,
                        slice_start,
                        slice_end,
                        slice_step,
                    )
                    new_args[i] = new_slice

            target = cast(EdgeOpOverload, op_node.target)
            new_op = graph.call_function(
                target,
                args=tuple(new_args),
                kwargs=op_node.kwargs,
            )
            new_op.meta["val"] = target(
                *[
                    a.meta["val"] if isinstance(a, torch.fx.Node) else a
                    for a in new_args
                ],
                **{
                    k: v.meta["val"] if isinstance(v, torch.fx.Node) else v
                    for k, v in op_node.kwargs.items()
                },
            )

        slice_node.replace_all_uses_with(new_op)
        graph.erase_node(slice_node)
        graph.erase_node(op_node)
        return True

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        parent = get_arg(node, "input", torch.fx.Node)
        if len(parent.users) != 1:
            return False
        if not isinstance(parent.target, EdgeOpOverload):
            return False

        entry = self._dispatch.get(parent.target)
        if entry is None:
            return False
        target_filter = self._target_filters.get(parent.target)
        if target_filter is not None and not target_filter(parent):
            return False

        should_swap, do_swap = entry
        return should_swap(parent, node) and do_swap(parent, node)
