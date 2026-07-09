# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass_utils import refresh_permute_view_meta
from executorch.backends.arm._passes.dim_maps import PermuteMap, ViewMap
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .arm_pass import ArmPass
from .canonicalize_view_copy_permute_pass import CanonicalizeViewCopyPermutePass
from .fuse_duplicate_users_pass import FuseDuplicateUsersPass
from .fuse_identical_input_transforms_pass import FuseIdenticalInputTransformsPass
from .remove_permutes_around_elementwise_tosa_ops import (
    RemovePermutesAroundElementwiseTosaOps,
)

_Dim = int | torch.SymInt


class PropagateViewCopyPermutePass(ArmPass, ABC):
    """Abstract implementation of a permute/view_copy propagation pass.

    To be used for upwards/downwards propagation by implementing the abstract
    methods for the direction of propagation.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _VIEW_DEFAULT_TARGET = exir_ops.edge.aten.view.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _VIEW_DEFAULT_TARGET, _PERMUTE_TARGET}
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
    ) -> None:
        super().__init__()
        if isinstance(compile_spec, ExportedProgram) and exported_program is None:
            exported_program = compile_spec
            compile_spec = None
        self.exported_program = exported_program
        self.compile_spec = compile_spec

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

        # Do not run for Ethos-U85 since this exposes a numerical issue
        # There is no target meta-data at this stage so use INT+cf as proxy
        # To be removed after MLBEDSW-11805
        while not self._is_u85_like_tosa_int_cf():
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                if node.target in self._TARGETS:
                    if len(node.users) == 0:
                        continue
                    iteration_modified |= self._propagate(node)

            if iteration_modified:
                graph_module = self._retrace(graph_module)
                result = self.fuse_horizontal(graph_module)
                graph_module = result.graph_module
                iteration_modified |= result.modified
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

    def _is_u85_like_tosa_int_cf(self) -> bool:
        if self.compile_spec is not None:
            tosa_spec = self.compile_spec.tosa_spec
        else:
            try:
                tosa_spec = get_context_spec()
            except RuntimeError:
                return False

        return (
            tosa_spec.support_integer()
            and not tosa_spec.support_float()
            and tosa_spec.support_extension("cf")
        )

    def _retrace(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        return super().call(graph_module).graph_module

    def _propagate(self, node: torch.fx.Node) -> bool:
        """Propagate a single permute/view node."""

        frontier = node
        previous_frontier = None
        moved = False
        while True:
            next_nodes = list(self._get_next_nodes(frontier))

            if len(next_nodes) == 0:
                assert node.op in (
                    "placeholder",
                    "output",
                ), f"{self.__class__.__name__} reached an endpoint node which is not a placeholder or output: {frontier}"
                break

            if not self._can_cross_next_nodes(frontier, next_nodes):
                break

            if len(next_nodes) > 1:
                if self._maybe_split_downwards_slice_fanout(node, next_nodes):
                    return True
                break

            next_node = next_nodes[0]
            if self.is_elementwise(next_node) and self._is_unary_elementwise(next_node):
                previous_frontier = frontier
                frontier = next_node
                moved = True
                continue

            if self.is_swappable(next_node):
                swapped_args = self._maybe_swap_args(node, next_node)
                if swapped_args is None:
                    break
                node.args = swapped_args[0]
                next_node.args = swapped_args[1]
                previous_frontier = frontier
                frontier = next_node
                moved = True
                continue

            # Concats are a special case since they branch the graph.
            # Perform the swap directly in this case and return.
            # Otherwise break and move the node before the concat
            if self._maybe_split_upwards_cat_fanout(node, next_node):
                return True

            # Unhandled case, stop propagation
            break

        if not moved:
            return False

        assert previous_frontier is not None
        self._move_node(node, frontier, previous_frontier)
        refresh_permute_view_meta(node)
        return True

    def fuse_vertical(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Fuse consecutive permute/view nodes."""
        modified = False

        if self.exported_program is not None:
            result = RemovePermutesAroundElementwiseTosaOps(self.exported_program).call(
                graph_module
            )
            graph_module = result.graph_module
            modified |= result.modified

        result = CanonicalizeViewCopyPermutePass().call(graph_module)
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

    def _maybe_split_downwards_slice_fanout(
        self, node: torch.fx.Node, next_nodes: Sequence[torch.fx.Node]
    ) -> bool:
        """Swap x2 = x1.permute; y1 = x2.slice_copy[0]; y2 = x2.slice_copy[1] to
        y1 = x1.permute.slice_copy[0]; y2 = x1.permute.slice_copy[1] Only if
        permutes after slice are noops.
        """
        return False

    def _maybe_swap_args(
        self, node: torch.fx.Node, next_node: torch.fx.Node
    ) -> Any | None:
        """If the node can be swapped with its next_node, return the new args
        for the next_node and new shape, otherwise return None.
        """
        if node.target == self._PERMUTE_TARGET:
            return self._maybe_swap_permute_args(node, next_node)
        elif node.target in {self._VIEW_TARGET, self._VIEW_DEFAULT_TARGET}:
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

    def is_elementwise(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target == exir_ops.backend.tosa.RESCALE.default:
            return self._is_per_tensor_rescale(node)

        if node.target == exir_ops.backend.tosa.TABLE.default:
            return True

        if node.target in self._TRANSPARENT_TARGETS:
            return True

        op = getattr(node.target, "_op", None)
        if op is not None and hasattr(op, "tags"):
            return torch.Tag.pointwise in op.tags
        return False

    def _is_per_tensor_rescale(self, node: torch.fx.Node) -> bool:
        if len(node.args) < 3:
            return False
        input_nodes = node.all_input_nodes
        if len(input_nodes) != 1:
            return False
        special_dtype_key = TosaSpecialDtype.meta_key()
        if input_nodes[0].meta.get(special_dtype_key) != node.meta.get(
            special_dtype_key
        ):
            return False
        scales = node.args[2]
        return not isinstance(scales, Sequence) or len(scales) == 1

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
                raise RuntimeError(
                    f"{self.__class__.__name__} expects keep_dim=True for reduction ops to simplify propagation logic, got {keep_dim} for node {next_node.name}."
                )
        return True

    def _is_unary_elementwise(self, node: torch.fx.Node) -> bool:
        if node.target == exir_ops.backend.tosa.TABLE.default:
            return True
        return len(node.all_input_nodes) == 1

    @staticmethod
    def _is_contiguous_nonempty(dims: Sequence[int]) -> bool:
        sorted_dims = sorted(set(dims))
        return bool(sorted_dims) and sorted_dims == list(
            range(sorted_dims[0], sorted_dims[-1] + 1)
        )


class PropagateViewCopyPermuteUpPass(PropagateViewCopyPermutePass):
    """Implements PropagateViewCopyPermutePass for upwards propagation:

    - Next propagation nodes are the input of the current node
    - Previous propagation nodes are the users of the current node
    - Swaps are (op -> permute/view) to (permute/view -> op)
    - Node is moved before the frontier next_node
    - Horizontal fuses are performed on users
    """

    def fuse_horizontal(self, graph_module):
        modified = False
        result = FuseDuplicateUsersPass().call(graph_module)
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
        if any(
            user.target == exir_ops.backend.tosa.SCATTER.default
            for user in frontier.users
        ):
            return False
        return all(
            all(prev_node is frontier for prev_node in self._get_prev_nodes(next_node))
            for next_node in next_nodes
        )

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
            return self._maybe_swap_reduction_view_args(
                node, next_node, view_map, new_shape
            )
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
        new_shape: list[_Dim] | None,
    ) -> Any | None:
        if new_shape is None:
            return None
        if len(next_node.args) <= 2 or next_node.args[2] is not True:
            return None
        reduction_dims = cast(int | Sequence[int], next_node.args[1])
        new_dims = view_map.map_dim(reduction_dims)
        if new_dims is None or not self._is_contiguous_nonempty(new_dims):
            return None
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
        if new_shape is None:
            return self._maybe_swap_unit_slice_view_args(
                node, next_node, view_map, input_shape
            )

        slice_dim = cast(int, next_node.args[1])
        new_dim = self._map_slice_dim(view_map, slice_dim)
        if new_dim is None:
            return None
        new_next_node_args = (*next_node.args[:1], new_dim, *next_node.args[2:])
        return ((*node.args[:1], new_shape), new_next_node_args)

    def _maybe_swap_unit_slice_view_args(
        self,
        node: torch.fx.Node,
        next_node: torch.fx.Node,
        view_map: ViewMap,
        input_shape: list[_Dim],
    ) -> Any | None:
        if len(next_node.args) < 4:
            return None
        step = next_node.args[4] if len(next_node.args) > 4 else 1
        remapped_slice = view_map.remap_unit_slice(
            input_shape,
            cast(int, next_node.args[1]),
            cast(_Dim, next_node.args[2]),
            cast(_Dim, next_node.args[3]),
            cast(_Dim, step),
        )
        if remapped_slice is None:
            return None

        new_shape, new_dim, new_start, new_end = remapped_slice
        new_next_node_args = (
            *next_node.args[:1],
            new_dim,
            new_start,
            new_end,
            *next_node.args[4:],
        )
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
        if node.target != self._PERMUTE_TARGET:
            return False
        if next_node.target != exir_ops.edge.aten.cat.default:
            return False

        cat_users = list(next_node.users)
        if len(cat_users) == 0:
            return False
        if not all(n.target == self._PERMUTE_TARGET for n in cat_users):
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
                    self._PERMUTE_TARGET,
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

    def fuse_horizontal(self, graph_module):
        modified = False
        result = FuseIdenticalInputTransformsPass().call(graph_module)
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
            new_dims = view_map.map_dim_inverse(next_node.args[1])
            if new_dims is None:
                return None
        elif next_node.target == exir_ops.edge.aten.slice_copy.Tensor:
            new_dims = view_map.map_dim_inverse(next_node.args[1])
            if new_dims is None:
                return None
            if len(new_dims) != 1:
                return None
            new_dims = new_dims[0]
        else:
            return None

        output_val = next_node.meta["val"]
        new_next_node_args = (*next_node.args[:1], new_dims, *next_node.args[2:])
        return ((*node.args[:1], list(output_val.shape)), new_next_node_args)

    def _maybe_split_downwards_slice_fanout(
        self, node: torch.fx.Node, next_nodes: Sequence[torch.fx.Node]
    ) -> bool:
        """Duplicate a permute onto each slice branch.

        The duplicated permutes are left before the slices; later propagation
        iterations handle swapping each one through its slice.

        """
        if node.target != self._PERMUTE_TARGET:
            return False
        if not all(
            next_node.target == exir_ops.edge.aten.slice_copy.Tensor
            and next_node.all_input_nodes == [node]
            for next_node in next_nodes
        ):
            return False

        producer = node.all_input_nodes[0]
        for next_node in next_nodes:
            with next_node.graph.inserting_before(next_node):
                branch_permute = next_node.graph.call_function(
                    self._PERMUTE_TARGET,
                    args=(producer, node.args[1]),
                )
            branch_permute.meta = dict(node.meta)
            next_node.replace_input_with(node, branch_permute)

        if len(node.users) == 0:
            node.graph.erase_node(node)
        return True

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
