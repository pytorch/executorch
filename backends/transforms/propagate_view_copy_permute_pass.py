# Copyright 2026 Arm Limited and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast, Set, Type

import torch
from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (
    CanonicalizeViewCopyPermutePass,
)
from executorch.backends.transforms.dim_maps import _Dim, PermuteMap, ViewMap
from executorch.backends.transforms.fuse_duplicate_users_pass import (
    FuseDuplicateUsersPass,
)
from executorch.backends.transforms.fuse_identical_input_transforms_pass import (
    FuseIdenticalInputTransformsPass,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


@dataclass(frozen=True)
class _ForkBranchSplit:
    next_node: torch.fx.Node
    source_shape: tuple[_Dim, ...]
    arg_update: tuple[Any, Any] | None


class PropagateViewCopyPermutePass(ExportPass, ABC):
    """Abstract implementation of a permute/view_copy propagation pass.

    To be used for upwards/downwards propagation by implementing the abstract
    methods for the direction of propagation. Backends may supply a closed set
    of equivalent permute targets and override the policy hooks when their
    layout contract permits more movement. Every hook preserves the existing
    behavior by default.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    # Moving an aliasing aten.view.default requires alias-aware reasoning that
    # this pass does not provide. Restrict propagation to copy semantics.
    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _PERMUTE_TARGET}
    _TRANSPARENT_TARGETS = {
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }

    _REDUCTION_TARGETS = {
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
    }
    _ARG_UPDATE_TARGETS = {
        *_REDUCTION_TARGETS,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def __init__(
        self,
        compile_spec: Any | None = None,
        exported_program: ExportedProgram | None = None,
        permute_targets: Iterable[Any] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(compile_spec, ExportedProgram) and exported_program is None:
            exported_program = compile_spec
            compile_spec = None
        self.exported_program = exported_program
        self.compile_spec = compile_spec
        # Which targets count as a permute. A backend carrying its own layout
        # dialect passes them here.
        self._permute_targets = frozenset(permute_targets or (self._PERMUTE_TARGET,))
        self._targets = {self._VIEW_TARGET} | self._permute_targets

    @staticmethod
    def _dim_arg(arg: Any) -> int | Sequence[int] | None:
        if isinstance(arg, int):
            return arg
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            return cast(Sequence[int], arg)
        return None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        result = self.fuse_horizontal(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        result = self.fuse_vertical(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        if result.modified:
            graph_module = self._retrace(graph_module)

        while True:
            iteration_modified = False
            # A rewrite invalidates metadata along its path and downstream.
            # Defer that region while still batching independent branches.
            stale_nodes: set[torch.fx.Node] = set()
            for node in list(graph_module.graph.nodes):
                if node in stale_nodes:
                    continue
                if node.target in self._targets:
                    if len(node.users) == 0:
                        continue
                    if self._propagate(node, stale_nodes):
                        iteration_modified = True

            if iteration_modified:
                modified = True
                graph_module = self._retrace(graph_module)
                continue

            result = self.fuse_horizontal(graph_module)
            graph_module = result.graph_module
            iteration_modified = result.modified
            result = self.fuse_vertical(graph_module)
            graph_module = result.graph_module
            iteration_modified |= result.modified

            modified |= iteration_modified
            if not iteration_modified:
                break

        if modified:
            graph_module = self._retrace(graph_module)
            graph_module.recompile()

        return PassResult(graph_module, modified)

    @staticmethod
    def _mark_downstream_nodes_stale(
        node: torch.fx.Node, stale_nodes: set[torch.fx.Node]
    ) -> None:
        pending = [node]
        while pending:
            current = pending.pop()
            if current in stale_nodes:
                continue
            stale_nodes.add(current)
            pending.extend(current.users)

    def _mark_stale_region(
        self,
        nodes: Iterable[torch.fx.Node],
        stale_nodes: set[torch.fx.Node],
    ) -> None:
        for node in nodes:
            self._mark_downstream_nodes_stale(node, stale_nodes)

    def _retrace(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        return super().call(graph_module).graph_module

    def _validated_next_nodes(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        stale_nodes: set[torch.fx.Node],
    ) -> list[torch.fx.Node] | None:
        next_nodes = list(self._get_next_nodes(frontier))
        if not next_nodes:
            assert frontier.op in (
                "placeholder",
                "output",
            ), f"{self.__class__.__name__} reached an endpoint node which is not a placeholder or output: {frontier}"
            return None

        if any(next_node in stale_nodes for next_node in next_nodes):
            return None

        if not self._can_cross_next_nodes(frontier, next_nodes):
            return None

        if self.blocks_moving(node, frontier, next_nodes):
            return None

        return next_nodes

    def _advance_through_next_node(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_node: torch.fx.Node,
    ) -> bool:
        if self._can_move_through_elementwise(node, frontier, next_node):
            return True

        if not self.is_swappable(next_node):
            return False

        swapped_args = self._maybe_swap_args(node, next_node)
        if swapped_args is None:
            return False

        node.args = swapped_args[0]
        next_node.args = swapped_args[1]
        return True

    def _propagate(self, node: torch.fx.Node, stale_nodes: set[torch.fx.Node]) -> bool:
        """Propagate one node without consulting metadata invalidated this
        scan.
        """

        frontier = node
        previous_frontier = None
        propagation_path = [node]
        moved = False
        while True:
            next_nodes = self._validated_next_nodes(node, frontier, stale_nodes)
            if next_nodes is None:
                break

            if len(next_nodes) > 1:
                if not self._maybe_split_fork(
                    node, frontier, previous_frontier, next_nodes
                ):
                    break
                self._mark_stale_region((*propagation_path, *next_nodes), stale_nodes)
                return True

            next_node = next_nodes[0]
            if self._advance_through_next_node(node, frontier, next_node):
                previous_frontier = frontier
                frontier = next_node
                propagation_path.append(next_node)
                moved = True
                continue

            if self._maybe_distribute_upwards_permute_over_elementwise(
                node, frontier, next_node
            ):
                return True

            # Concats are a special case since they branch the graph.
            # Perform the swap directly in this case and return.
            # Otherwise break and move the node before the concat
            if self._maybe_split_upwards_cat_fanout(node, next_node):
                self._mark_stale_region((*propagation_path, next_node), stale_nodes)
                return True

            # Unhandled case, stop propagation
            break

        if not moved:
            return False

        assert previous_frontier is not None
        self._move_node(node, frontier, previous_frontier)
        self._mark_stale_region(propagation_path, stale_nodes)
        return True

    def duplicate_user_fusion_exclusions(self) -> frozenset:
        """Targets whose duplicate users must not be collapsed onto one node.

        Fusing them is unsound wherever a later stage assumes each consumer
        keeps its own producer.
        """
        return frozenset()

    def duplicate_user_fusion_key(self, node: torch.fx.Node) -> Any:
        """Return backend metadata that must match before users are fused."""
        return None

    def make_fusion_pass(self) -> ExportPass | None:
        """The region-cancellation engine to run before canonicalization.

        Region cancellation reaches shapes single-node propagation cannot -- a
        diamond whose operands are both inside the region, for instance -- so
        the two are complementary. Return None to skip it.

        """
        return None

    def fuse_vertical(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Fuse consecutive permute/view nodes."""
        modified = False

        fusion_pass = self.make_fusion_pass()
        if fusion_pass is not None:
            result = fusion_pass.call(graph_module)
            graph_module = result.graph_module
            modified |= result.modified

        result = CanonicalizeViewCopyPermutePass(self._permute_targets).call(
            graph_module
        )
        graph_module = result.graph_module
        modified |= result.modified
        return PassResult(graph_module, modified)

    @abstractmethod
    def fuse_horizontal(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Fuse parallel permute/view nodes going into/ out a single node."""
        pass

    @abstractmethod
    def _get_next_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        """Return the next nodes in the direction of propagation."""
        pass

    @abstractmethod
    def _get_prev_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        """Return the previous nodes in the direction of propagation."""
        pass

    def _can_cross_next_nodes(
        self, frontier: torch.fx.Node, next_nodes: Sequence[torch.fx.Node]
    ) -> bool:
        return True

    @abstractmethod
    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        pass

    @abstractmethod
    def _maybe_swap_view_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        pass

    def _maybe_split_upwards_cat_fanout(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> bool:
        """Swap cat([x1,x2]).permute(p) -> cat([x1.permute(p'), x2.permute(p')])
        if permutes before the concat are noops.
        """
        return False

    def _maybe_distribute_upwards_permute_over_elementwise(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_node: torch.fx.Node,
    ) -> bool:
        """Optionally distribute an upward-moving permute over multiple inputs.

        The shared driver leaves this disabled because distribution can increase
        the number of layout copies. Backends may opt in when their layout
        strategy requires crossing a reconvergent elementwise node.
        """
        return False

    def _maybe_split_fork(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        previous_frontier: torch.fx.Node | None,
        next_nodes: Sequence[torch.fx.Node],
    ) -> bool:
        """Optionally split a transform over a fork in the propagation
        direction.
        """
        return False

    def _maybe_swap_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        """If the node can be swapped with its next_node, return the new args
        for the next_node and new shape, otherwise return None.
        """
        if node.target in self._permute_targets:
            return self._maybe_swap_permute_args(node, next_node)
        elif node.target == self._VIEW_TARGET:
            return self._maybe_swap_view_args(node, next_node)
        else:
            raise ValueError(
                f"Unexpected node target {node.target} in {self.__class__.__name__}"
            )

    def _move_node(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        previous_frontier: torch.fx.Node,
    ) -> None:
        """Update the graph to move the node into its new position."""
        raise NotImplementedError()

    def _shape_seen_by_next_node(self, moving_node: torch.fx.Node) -> Any:
        """Shape the crossed node operates on once ``moving_node`` has moved."""
        raise NotImplementedError()

    def is_transparent(self, node: torch.fx.Node) -> bool:
        """Ops a data-movement node may cross without changing meaning.

        Rank-changing views may not cross these, so a backend adding to the set
        is asserting layout-invariance, not merely elementwise-ness.

        """
        return node.target in self._TRANSPARENT_TARGETS

    def is_multi_input_elementwise(self, node: torch.fx.Node) -> bool:
        """Elementwise ops whose extra inputs are not layout-carrying."""
        return False

    def blocks_crossing(self, node: torch.fx.Node) -> bool:
        """Users past which a frontier node must not be moved."""
        return False

    def blocks_moving(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_nodes: Sequence[torch.fx.Node],
    ) -> bool:
        """Whether this copy must stop here, whatever the nodes ahead allow.

        Asked once per step, before the step is planned, for reasons about the
        value being moved rather than about the node being crossed -- a storage
        format the copy would address wrongly, say.

        """
        return False

    def tolerates_shape_after_move(self, next_node: torch.fx.Node, shape: Any) -> bool:
        """Whether ``next_node`` can operate on the shape the move leaves it.

        Crossing a node changes the shape it reads, and a backend can have
        operators that are correct only for some of them.

        """
        return True

    def is_elementwise(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if self.is_transparent(node):
            return True

        op = getattr(node.target, "_op", None)
        if op is not None and hasattr(op, "tags"):
            return torch.Tag.pointwise in op.tags
        return False

    def is_swappable(self, next_node: torch.fx.Node) -> bool:
        if next_node.target not in self._ARG_UPDATE_TARGETS:
            return False
        if next_node.target in self._REDUCTION_TARGETS:
            keep_dim = (
                next_node.args[2]
                if len(next_node.args) > 2
                else next_node.kwargs.get("keepdim")
            )
            if keep_dim is not True:
                # A reduction that drops the dimension changes rank, so the
                # permutation cannot simply be remapped across it.
                return False
        return True

    def _can_move_through_elementwise(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_node: torch.fx.Node,
    ) -> bool:
        """Return whether ``moving_node`` can cross an elementwise operation.

        Only simple single-input, single-output operations are handled here.
        Multi-input elementwise operations are handled by horizontal fusion.

        """
        if (
            not self.is_elementwise(next_node)
            or (
                len(next_node.all_input_nodes) != 1
                and not self.is_multi_input_elementwise(next_node)
            )
            or not isinstance(next_node.meta.get("val"), torch.Tensor)
        ):
            return False

        if moving_node.target == self._VIEW_TARGET:
            view_map = ViewMap(moving_node)
            if not view_map.is_valid_map:
                return False
            if (
                self.is_transparent(next_node)
                and view_map.source_rank != view_map.target_rank
            ):
                return False

        if not self.tolerates_shape_after_move(
            next_node, self._shape_seen_by_next_node(moving_node)
        ):
            return False

        if moving_node.target not in self._permute_targets:
            return True

        dims = self._dim_arg(moving_node.args[1])
        source_rank = len(dims) if isinstance(dims, Sequence) else None
        next_val = next_node.meta.get("val")
        return (
            source_rank is not None
            and isinstance(next_val, torch.Tensor)
            and len(next_val.shape) == source_rank
        )


class PropagateViewCopyPermuteUpPass(PropagateViewCopyPermutePass):
    """Implements PropagateViewCopyPermutePass for upwards propagation:

    - Next propagation nodes are the input of the current node
    - Previous propagation nodes are the users of the current node
    - Swaps are (op -> permute/view) to (permute/view -> op)
    - Node is moved before the frontier next_node
    - Horizontal fuses are performed on users
    """

    def _shape_seen_by_next_node(self, moving_node: torch.fx.Node) -> Any:
        """Moving up leaves the crossed node reading ``moving_node``'s output."""
        val = moving_node.meta.get("val")
        return getattr(val, "shape", None)

    def fuse_horizontal(self, graph_module):
        modified = False
        result = FuseDuplicateUsersPass(
            self.duplicate_user_fusion_exclusions(),
            allowed_targets={self._VIEW_TARGET, *self._permute_targets},
            semantic_key=self.duplicate_user_fusion_key,
        ).call(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        return PassResult(graph_module, modified)

    def _get_next_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.all_input_nodes)

    def _get_prev_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.users.keys())

    def _can_cross_next_nodes(
        self, frontier: torch.fx.Node, next_nodes: Sequence[torch.fx.Node]
    ) -> bool:
        if any(self.blocks_crossing(user) for user in frontier.users):
            return False
        return all(
            all(prev_node is frontier for prev_node in self._get_prev_nodes(next_node))
            for next_node in next_nodes
        )

    def _can_move_through_elementwise(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_node: torch.fx.Node,
    ) -> bool:
        if super()._can_move_through_elementwise(moving_node, frontier, next_node):
            return True
        if moving_node.target not in self._permute_targets or not self.is_elementwise(
            next_node
        ):
            return False

        frontier_val = frontier.meta.get("val")
        if not isinstance(frontier_val, torch.Tensor):
            return False
        rank = len(frontier_val.shape)
        layout_dependent_inputs = [
            input_node
            for input_node in next_node.all_input_nodes
            if not FuseIdenticalInputTransformsPass.is_layout_invariant(
                input_node, rank
            )
        ]
        return len(layout_dependent_inputs) == 1

    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        permute_map = PermuteMap(node)
        args = self._dim_arg(next_node.args[1])
        if args is None:
            return None
        mapped_args = permute_map.map_dims(args)
        new_args: int | list[int] = (
            mapped_args[0] if isinstance(args, int) else mapped_args
        )
        return (node.args, (*next_node.args[:1], new_args, *next_node.args[2:]))

    def _maybe_swap_view_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        view_map = ViewMap(node)
        if not view_map.is_valid_map or len(next_node.all_input_nodes) != 1:
            return None

        input_val = next_node.all_input_nodes[0].meta["val"]
        input_shape = list(input_val.shape)
        new_shape = view_map.remap_target_shape(input_shape)

        if next_node.target in self._REDUCTION_TARGETS:
            return self._maybe_swap_reduction_view_args(node, next_node, view_map)
        if next_node.target == exir_ops.edge.aten.slice_copy.Tensor:
            return self._maybe_swap_slice_view_args(
                node, next_node, view_map, input_shape, new_shape
            )
        return None

    def _maybe_swap_reduction_view_args(
        self,
        node: torch.fx.Node,
        next_node: torch.fx.Node,
        view_map: ViewMap,
    ) -> Any | None:
        if len(next_node.args) <= 2 or next_node.args[2] is not True:
            return None
        reduction_dims = cast(int | Sequence[int], next_node.args[1])
        input_val = next_node.all_input_nodes[0].meta["val"]
        swap = view_map.map_reduction_after_view(input_val.shape, reduction_dims)
        if swap is None:
            return None
        new_shape, new_dims = swap
        new_next_node_args = (*next_node.args[:1], new_dims, *next_node.args[2:])
        return ((*node.args[:1], new_shape), new_next_node_args)

    def _maybe_swap_slice_view_args(
        self,
        node: torch.fx.Node,
        next_node: torch.fx.Node,
        view_map: ViewMap,
        input_shape: list[_Dim],
        new_shape: list[_Dim] | None,
    ) -> Any | None:
        if len(next_node.args) < 4:
            return None

        step = next_node.args[4] if len(next_node.args) > 4 else 1
        unit_slice_swap = view_map.remap_unit_slice(
            input_shape,
            cast(int, next_node.args[1]),
            cast(_Dim, next_node.args[2]),
            cast(_Dim, next_node.args[3]),
            cast(_Dim, step),
        )
        if unit_slice_swap is not None:
            new_shape, new_dim, new_start, new_end = unit_slice_swap
            if not self._valid_slice_interval(new_shape, new_dim, new_start, new_end):
                return None
            new_next_node_args = (
                *next_node.args[:1],
                new_dim,
                new_start,
                new_end,
                *next_node.args[4:],
            )
            return ((*node.args[:1], new_shape), new_next_node_args)

        if new_shape is None:
            return None

        slice_dim = cast(int, next_node.args[1])
        mapped_dim = self._map_slice_dim(view_map, slice_dim)
        if mapped_dim is None:
            return None
        if not self._valid_slice_interval(
            new_shape,
            mapped_dim,
            cast(_Dim, next_node.args[2]),
            cast(_Dim, next_node.args[3]),
        ):
            return None
        new_next_node_args = (*next_node.args[:1], mapped_dim, *next_node.args[2:])
        return ((*node.args[:1], new_shape), new_next_node_args)

    @staticmethod
    def _map_slice_dim(view_map: ViewMap, slice_dim: int) -> int | None:
        new_dims = view_map.map_source_dims_to_target_axes(slice_dim)
        if new_dims is None or len(new_dims) != 1:
            return None

        new_dim = new_dims[0]
        normalized_slice_dim = slice_dim % view_map.source_rank
        source_to_target_axes = view_map.source_to_target_axes()
        target_source_axes = view_map.source_axes_for_target_axis(
            new_dim, source_to_target_axes
        )
        if any(
            source_axis != normalized_slice_dim for source_axis in target_source_axes
        ):
            return None
        return new_dim

    @staticmethod
    def _valid_slice_interval(
        shape: Sequence[_Dim], dim: int, start: _Dim, end: _Dim
    ) -> bool:
        try:
            dim = dim % len(shape)
            return 0 <= start < end <= shape[dim]
        except (RuntimeError, TypeError):
            return False

    def _move_node(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        previous_frontier: torch.fx.Node,
    ) -> None:
        original_input = node.all_input_nodes[0]
        if frontier.op == "placeholder":
            # Nodes cannot be moved before placeholders
            producer = frontier
            frontier_user = previous_frontier
        else:
            producer = frontier.all_input_nodes[0]
            frontier_user = frontier

        node.replace_input_with(original_input, producer)
        frontier_user.replace_input_with(producer, node)

        for user in list(node.users):
            if user is not frontier_user:
                user.replace_input_with(node, original_input)

        frontier_user.prepend(node)

    def _maybe_split_upwards_cat_fanout(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> bool:
        """Swap cat([x1,x2]).permute(p) -> cat([x1.permute(p'), x2.permute(p')])
        if permutes before the concat are noops.
        """
        if node.target not in self._permute_targets:
            return False
        if next_node.target != exir_ops.edge.aten.cat.default:
            return False

        cat_users = list(next_node.users)
        if len(cat_users) == 0:
            return False
        if not all(n.target in self._permute_targets for n in cat_users):
            return False

        permute_args = [self._dim_arg(n.args[1]) for n in cat_users]
        if not isinstance(permute_args[0], Sequence) or not all(
            p == permute_args[0] for p in permute_args
        ):
            return False

        cat_dim = (
            next_node.args[1]
            if len(next_node.args) >= 2
            else next_node.kwargs.get("dim", 0)
        )
        if not isinstance(cat_dim, int):
            return False
        new_cat_dim = PermuteMap(node).map_dims(cat_dim)[0]

        cat_inputs = list(next_node.all_input_nodes)
        cat_input_shapes = [input_node.meta["val"].shape for input_node in cat_inputs]

        # Ensure all input permutes are noops
        if not all(
            CanonicalizeViewCopyPermutePass._is_singleton_permutation(
                shape, permute_args[0]
            )
            for shape in cat_input_shapes
        ):
            return False

        # Add permutes to all cat inputs, update cat arg, and remove old output permute
        new_inputs = []
        for input_node in cat_inputs:
            input_val = input_node.meta["val"]
            output_shape = [input_val.shape[dim] for dim in permute_args[0]]
            with next_node.graph.inserting_before(next_node):
                permute = next_node.graph.call_function(
                    cast(Any, node.target),
                    args=(input_node, permute_args[0]),
                )
            permute.meta = dict(input_node.meta)
            permute.meta["val"] = input_val.new_empty(tuple(output_shape))
            new_inputs.append(permute)

        next_node.args = (new_inputs, new_cat_dim, *next_node.args[2:])
        next_node.meta = dict(node.meta)
        for cat_user in cat_users:
            cat_user.replace_all_uses_with(next_node)
        for cat_user in cat_users:
            if len(cat_user.users) == 0:
                next_node.graph.erase_node(cat_user)
        return True


class PropagateViewCopyPermuteDownPass(PropagateViewCopyPermutePass):
    """Implements PropagateViewCopyPermutePass for downward propagation:

    - Next propagation nodes are the users of the current node
    - Previous propagation nodes are the inputs of the current node
    - Swaps are (permute/view -> op) to (op -> permute/view)
    - Node is moved after the frontier next_node
    - Horizontal fuses are performed on inputs
    """

    def _shape_seen_by_next_node(self, moving_node: torch.fx.Node) -> Any:
        """Moving down leaves the crossed node reading ``moving_node``'s input."""
        val = cast(torch.fx.Node, moving_node.args[0]).meta.get("val")
        return getattr(val, "shape", None)

    def fuse_horizontal(self, graph_module):
        modified = False
        result = FuseIdenticalInputTransformsPass(
            permute_targets=self._permute_targets
        ).call(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        return PassResult(graph_module, modified)

    def _get_next_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.users.keys())

    def _get_prev_nodes(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.all_input_nodes)

    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        permute_map = PermuteMap(node)
        args = self._dim_arg(next_node.args[1])
        if args is None:
            return None
        mapped_args = permute_map.map_dims_inverse(args)
        new_args: int | list[int] = (
            mapped_args[0] if isinstance(args, int) else mapped_args
        )
        return (node.args, (*next_node.args[:1], new_args, *next_node.args[2:]))

    def _maybe_swap_view_args(self, node, next_node):
        view_map = ViewMap(node)
        if not view_map.is_valid_map:
            return None

        if next_node.target in self._REDUCTION_TARGETS:
            if len(next_node.args) <= 2 or next_node.args[2] is not True:
                return None
            swap = view_map.map_reduction_before_view(next_node.args[1])
            if swap is None:
                return None
            new_dims, output_shape = swap
        elif next_node.target == exir_ops.edge.aten.slice_copy.Tensor:
            new_dims = view_map.map_dim_inverse(next_node.args[1])
            if new_dims is None:
                return None
            if len(new_dims) != 1:
                return None
            new_dims = new_dims[0]
            output_shape = list(next_node.meta["val"].shape)
        else:
            return None

        new_next_node_args = (*next_node.args[:1], new_dims, *next_node.args[2:])
        return ((*node.args[:1], output_shape), new_next_node_args)

    def _maybe_split_fork(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        previous_frontier: torch.fx.Node | None,
        next_nodes: Sequence[torch.fx.Node],
    ) -> bool:
        if frontier is not node and previous_frontier is None:
            return False
        plan = self._plan_fork_split(node, frontier, next_nodes)
        if plan is None:
            return False
        producer, branch_splits = plan
        self._apply_fork_split(node, frontier, producer, branch_splits)
        return True

    def _plan_fork_split(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_nodes: Sequence[torch.fx.Node],
    ) -> tuple[torch.fx.Node, tuple[_ForkBranchSplit, ...]] | None:
        """Validate every fork branch and return an all-or-nothing rewrite plan.

        Fork propagation is only implemented for permutes. The permutation must be
        valid, every branch must support the same propagation step, and every branch
        output must retain the permutation rank. Planning all branches before editing
        is essential: discovering one unsupported branch after mutating earlier ones
        would leave a partially rewritten graph.

        Each planned branch records both its pre-permute shape and any argument update
        needed when crossing reductions or slices.

        """
        if node.target not in self._permute_targets or len(node.all_input_nodes) != 1:
            return None
        producer = node.all_input_nodes[0]
        if not isinstance(producer.meta.get("val"), torch.Tensor):
            return None
        permute_dims = self._dim_arg(node.args[1])
        if not isinstance(permute_dims, Sequence):
            return None
        rank = len(permute_dims)
        normalized_dims = [dim if dim >= 0 else dim + rank for dim in permute_dims]
        if sorted(normalized_dims) != list(range(rank)):
            return None

        branch_splits = []
        for next_node in next_nodes:
            if self._can_move_through_elementwise(
                node, frontier, next_node
            ) or self._can_split_through_elementwise(node, frontier, next_node):
                arg_update = None
            elif self.is_swappable(next_node):
                arg_update = self._maybe_swap_args(node, next_node)
                if arg_update is None:
                    return None
            else:
                return None

            next_val = next_node.meta.get("val")
            if not isinstance(next_val, torch.Tensor) or len(next_val.shape) != rank:
                return None
            source_shape: list[_Dim] = [1] * rank
            for output_axis, source_axis in enumerate(normalized_dims):
                source_shape[source_axis] = next_val.shape[output_axis]
            branch_splits.append(
                _ForkBranchSplit(next_node, tuple(source_shape), arg_update)
            )
        return producer, tuple(branch_splits)

    def _can_split_through_elementwise(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_node: torch.fx.Node,
    ) -> bool:
        if not self.is_elementwise(next_node):
            return False
        frontier_val = frontier.meta.get("val")
        if not isinstance(frontier_val, torch.Tensor):
            return False
        rank = len(frontier_val.shape)
        return all(
            input_node is frontier
            or FuseIdenticalInputTransformsPass.is_layout_invariant(input_node, rank)
            for input_node in next_node.all_input_nodes
        )

    def _apply_fork_split(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        producer: torch.fx.Node,
        branch_splits: Sequence[_ForkBranchSplit],
    ) -> None:
        """Apply a validated fork plan and place one transform after each
        branch.

        If propagation already crossed operators before reaching the fork, the
        original path is first detached from the permute. Each branch is then
        rewired to the pre-permute layout, its metadata is updated to that
        layout, and a copy of the original permute is inserted after it.
        Argument-changing operations use the transform arguments captured during
        planning.

        """
        if frontier is not node:
            original_user = next(iter(node.users))
            original_user.replace_input_with(node, producer)

        for branch_split in branch_splits:
            next_node = branch_split.next_node
            old_next_meta = copy.copy(next_node.meta)
            if frontier is node:
                next_node.replace_input_with(node, producer)
            if branch_split.arg_update is not None:
                branch_input = (
                    producer if frontier is node else branch_split.arg_update[1][0]
                )
                next_node.args = (branch_input, *branch_split.arg_update[1][1:])
            next_node.meta = copy.copy(next_node.meta)
            next_node.meta["val"] = old_next_meta["val"].new_empty(
                branch_split.source_shape
            )
            transform_suffix = (
                node.args[1:]
                if branch_split.arg_update is None
                else branch_split.arg_update[0][1:]
            )
            with next_node.graph.inserting_after(next_node):
                branch_transform = next_node.graph.call_function(
                    cast(Any, node.target),
                    args=(next_node, *transform_suffix),
                    kwargs=dict(node.kwargs),
                )
            branch_transform.meta = old_next_meta
            for user in list(next_node.users):
                if user is not branch_transform:
                    user.replace_input_with(next_node, branch_transform)

        if not node.users:
            node.graph.erase_node(node)

    def _move_node(
        self,
        node: torch.fx.Node,
        frontier: torch.fx.Node,
        previous_frontier: torch.fx.Node,
    ) -> None:
        original_user = next(iter(node.users))
        producer = node.all_input_nodes[0]
        if frontier.op == "output":
            # Nodes cannot be moved after output
            frontier_input = previous_frontier
        else:
            frontier_input = frontier
        frontier_users = list(frontier_input.users)

        original_user.replace_input_with(node, producer)
        node.replace_input_with(producer, frontier_input)

        for user in frontier_users:
            if user is not node:
                user.replace_input_with(frontier_input, node)

        if frontier.op == "output":
            frontier.prepend(node)
        else:
            frontier.append(node)
