# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, Sequence, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.dim_maps import (
    _dim_equals,
    _is_permutation,
    _normalize_dim,
    _normalize_dims,
    ViewMap,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from torch.fx import GraphModule, Node
from torch.fx.node import Target
from torch.fx.passes.infra.pass_base import PassResult

_Dim = int | torch.SymInt


class CanonicalizeViewCopyPermutePass(ArmPass):
    """Canonicalize view/permute chains.

    The pass repeatedly fuses adjacent compatible ops and swaps adjacent
    view/permute pairs when the swap exposes more fusing.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _PERMUTE_TARGET}

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for chain in self._collect_chains(graph_module):
            updated_chain = chain

            while True:
                updated_chain, fused = self._fuse_sequential_ops(
                    graph_module, updated_chain
                )
                modified = modified or fused

                if len(updated_chain) > 2:
                    op1: Node = updated_chain[0]
                    op2: Node | None = updated_chain[1]
                    swapped_args = None
                    i = 2
                    while swapped_args is None and op2 is not None:
                        swapped_args = self._maybe_swap_args(op1, op2)
                        if swapped_args is None:
                            op1 = op2
                            op2 = updated_chain[i] if i < len(updated_chain) else None
                            i += 1

                    if swapped_args is not None:
                        input_node = op1.args[0]
                        assert isinstance(input_node, Node)
                        assert op2 is not None
                        op1_target = op1.target
                        op2_target = op2.target
                        self._set_node_op(op1, op2_target, input_node, swapped_args[0])
                        self._set_node_op(op2, op1_target, op1, swapped_args[1])
                        modified = True
                    else:
                        break

                else:
                    break

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _collect_chains(self, graph_module: GraphModule) -> list[list[Node]]:
        """Returns a list of linear chains of view/permutes in the graph."""
        chains: list[list[Node]] = []

        view_permute_nodes = [
            node for node in graph_module.graph.nodes if node.target in self._TARGETS
        ]

        while view_permute_nodes:
            node = view_permute_nodes.pop(0)
            chain = [node]
            current = node

            while len(current.users) == 1:
                user = next(iter(current.users))
                if user.target not in self._TARGETS:
                    break
                view_permute_nodes.remove(user)
                chain.append(user)
                current = user

            chains.append(chain)

        return chains

    def _fuse_sequential_ops(
        self, graph_module: GraphModule, chain: list[Node]
    ) -> tuple[list[Node], bool]:
        """Loop over chain and fuse adjacent ops and remove no-op views/permutes
        until no more fusions are possible.

        Returns the updated chain and whether any changes were made.

        """

        updated_chain = list(chain)
        any_changed = False

        while True:
            changed = False
            index = 0

            while index < len(updated_chain):
                node = updated_chain[index]
                input_node = cast(Node, node.args[0])

                if node.target == self._VIEW_TARGET and self._shapes_equal(
                    self._shape(input_node), self._shape(node)
                ):
                    # Identity view
                    self._remove_node(
                        graph_module, updated_chain, index, replacement=input_node
                    )
                    changed = True
                    any_changed = True
                    continue

                if node.target == self._PERMUTE_TARGET:

                    dims = self._permute_dims(node)

                    # Normalize dims
                    if any(
                        dim < 0 or dim >= len(self._shape(input_node)) for dim in dims
                    ):
                        dims = [
                            _normalize_dim(dim, len(self._shape(input_node)))
                            for dim in dims
                        ]
                        self._set_node_op(node, self._PERMUTE_TARGET, input_node, dims)
                        changed = True
                        any_changed = True

                    # Identity permute
                    if dims == list(range(len(dims))):
                        self._remove_node(
                            graph_module, updated_chain, index, replacement=input_node
                        )
                        changed = True
                        any_changed = True
                        continue

                    input_shape = self._shape(input_node)

                    # Permute w/o data movement e.g. [1, 2] -> [2, 1] decomposes to view
                    # Dynamic views are not supported by TOSA, so only check for static shapes
                    if not any(
                        isinstance(dim, torch.SymInt) for dim in input_shape
                    ) and self._is_singleton_permutation(input_shape, dims):
                        self._set_node_op(
                            node, self._VIEW_TARGET, input_node, self._shape(node)
                        )
                        changed = True
                        any_changed = True
                        continue

                if index + 1 < len(updated_chain):
                    next_node = updated_chain[index + 1]
                    if (
                        node.target == self._VIEW_TARGET
                        and next_node.target == self._VIEW_TARGET
                    ):
                        # Fuse conscutive views
                        self._set_node_op(
                            node, self._VIEW_TARGET, input_node, self._shape(next_node)
                        )
                        self._remove_node(
                            graph_module, updated_chain, index + 1, replacement=node
                        )
                        changed = True
                        any_changed = True
                        continue

                    if (
                        node.target == self._PERMUTE_TARGET
                        and next_node.target == self._PERMUTE_TARGET
                    ):
                        # Fuse consecutive permutes
                        dims = self._permute_dims(node)
                        next_dims = self._permute_dims(next_node)
                        self._set_node_op(
                            node,
                            self._PERMUTE_TARGET,
                            input_node,
                            [dims[dim] for dim in next_dims],
                        )
                        self._remove_node(
                            graph_module, updated_chain, index + 1, replacement=node
                        )
                        changed = True
                        any_changed = True
                        continue

                index += 1

            if not changed:
                return updated_chain, any_changed

    def _maybe_swap_args(
        self, op1: Node, op2: Node
    ) -> tuple[Sequence[_Dim], Sequence[_Dim]] | None:
        """Returns updated arguments for a valid op swap, or None if the ops
        cannot be swapped.
        """
        input_node = op1.args[0]
        assert isinstance(input_node, Node)
        input_val = input_node.meta["val"]

        if op1.target == self._PERMUTE_TARGET and op2.target == self._VIEW_TARGET:
            return self._permute_view_swap(input_val, op1, op2)

        if op1.target == self._VIEW_TARGET and op2.target == self._PERMUTE_TARGET:
            return self._view_permute_swap(op1, op2)

        return None

    def _permute_view_swap(
        self, input_val: torch.Tensor, permute_node: Node, view_node: Node
    ) -> tuple[list[_Dim], list[int]] | None:
        """Return updated args for swapping permute(P).view(S) ->
        view(S').permute(P')

        P describes each permuted output axis in terms of an input axis. Use
        inverse(P) to recover where each original input axis landed in the
        permuted tensor, then map that axis order through the view to construct
        S' and the final restoring permutation P'.

        """
        x_shape = cast(list[_Dim], list(input_val.shape))
        permute_dims = _normalize_dims(self._permute_dims(permute_node), len(x_shape))

        view_map = self._view_map(view_node)
        if view_map is None:
            return None
        if not view_map.is_valid_map:
            return None

        permuted_axis = self._inverse_permutation(permute_dims)
        target_axis_order = view_map.map_permutation(permuted_axis)
        if target_axis_order is None:
            return None

        view_shape_before_permute = [
            view_map.target_shape[target_axis] for target_axis in target_axis_order
        ]

        return (
            view_shape_before_permute,
            self._inverse_permutation(target_axis_order),
        )

    def _view_permute_swap(
        self, view_node: Node, permute_node: Node
    ) -> tuple[list[int], list[_Dim]] | None:
        """Return updated args for swapping view(S).permute(P) ->
        permute(P').view(S')

        where S' is the shape of the original permute output, and P' is computed
        using the view_map helper.

        """
        view_map = self._view_map(view_node)
        if view_map is None:
            return None
        if not view_map.is_valid_map:
            return None

        permute_dims = self._permute_dims(permute_node)
        mapped_dims = view_map.map_permutation_inverse(permute_dims)
        if mapped_dims is None:
            return None

        return mapped_dims, self._shape(permute_node)

    @staticmethod
    def _inverse_permutation(permutation: Sequence[int]) -> list[int]:
        inverse = [0] * len(permutation)
        for index, dim in enumerate(permutation):
            inverse[dim] = index
        return inverse

    @classmethod
    def _is_singleton_permutation(
        cls, shape: Sequence[_Dim], permutation: Sequence[int]
    ) -> bool:
        rank = len(shape)
        normalized_perm = [_normalize_dim(dim, rank) for dim in permutation]
        if not _is_permutation(normalized_perm, rank):
            return False

        non_singleton_axes = [
            axis for axis, dim in enumerate(shape) if not _dim_equals(dim, 1)
        ]
        permuted_non_singleton_axes = [
            axis for axis in normalized_perm if not _dim_equals(shape[axis], 1)
        ]
        return permuted_non_singleton_axes == non_singleton_axes

    @staticmethod
    def _view_map(view_node: Node) -> ViewMap | None:
        try:
            return ViewMap(view_node)
        except AssertionError:
            return None

    @staticmethod
    def _shapes_equal(lhs: Sequence[_Dim], rhs: Sequence[_Dim]) -> bool:
        return len(lhs) == len(rhs) and all(
            _dim_equals(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs, rhs)
        )

    def _remove_node(
        self,
        graph_module: GraphModule,
        chain: list[Node],
        index: int,
        replacement: Node,
    ) -> None:
        node = chain[index]
        assert node is not replacement

        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        del chain[index]

    def _set_node_op(
        self,
        node: Node,
        target: Target,
        input_node: Node,
        arg: Sequence[_Dim],
    ) -> None:
        node.target = target
        node.args = (input_node, list(arg))
        self._refresh_meta(node)

    def _permute_dims(self, node: Node) -> list[int]:
        assert node.target == self._PERMUTE_TARGET, "Expected permute node"
        return list(cast(Sequence[int], node.args[1]))

    @classmethod
    def _refresh_meta(cls, node: Node) -> None:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        input_val = input_node.meta.get("val")
        if input_val is None or node.target not in cls._TARGETS:
            return

        # Compute new meta shapes to preserve SymInts.
        if isinstance(input_val, torch.Tensor):
            if node.target == cls._VIEW_TARGET:
                node.meta["val"] = input_val.new_empty(
                    tuple(cast(Sequence[_Dim], node.args[1]))
                )
                return

            if node.target == cls._PERMUTE_TARGET:
                dims = _normalize_dims(
                    cast(Sequence[int], node.args[1]), len(input_val.shape)
                )
                node.meta["val"] = input_val.new_empty(
                    tuple(input_val.shape[dim] for dim in dims)
                )
                return

        node.meta["val"] = node.target(input_val, *node.args[1:])  # type: ignore[operator]

    @staticmethod
    def _shape(node: Node) -> list[_Dim]:
        return cast(list[_Dim], list(node.meta["val"].shape))
