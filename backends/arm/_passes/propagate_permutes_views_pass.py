# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, Set, Type

import torch
from executorch.backends.arm._passes.view_map import PermuteMap, ViewMap

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .arm_pass import ArmPass
from .canonicalize_view_copy_permute_pass import CanonicalizeViewCopyPermutePass
from .fuse_duplicate_users_pass import FuseDuplicateUsersPass
from .fuse_identical_input_transforms_pass import FuseIdenticalInputTransformsPass


class PropagatePermuteViewsPass(ArmPass):
    """Move permute/view nodes towards graph inputs."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _VIEW_DEFAULT_TARGET = exir_ops.edge.aten.view.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _VIEW_DEFAULT_TARGET, _PERMUTE_TARGET}

    _REDUCTION_TARGETS = {
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
    }
    _ARG_UPDATE_TARGETS = {
        *_REDUCTION_TARGETS,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def __init__(self, exported_program: ExportedProgram | None = None) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _dim_arg(arg: Any) -> int | Sequence[int] | None:
        if isinstance(arg, int):
            return arg
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            return cast(Sequence[int], arg)
        return None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        result = self.fuse_vertical(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        result = self.fuse_horizontal(graph_module)
        graph_module = result.graph_module
        modified |= result.modified
        if result.modified:
            graph_module = self._retrace(graph_module)

        while True:
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                if node.target in self._TARGETS:
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
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)

    def _retrace(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
        return super().call(graph_module).graph_module

    def _propagate(self, node: torch.fx.Node) -> bool:
        """Propagate a single permute/view node."""

        frontier = node
        moved = False
        while True:
            neighbours = list(self._get_neighbours(frontier))

            if len(neighbours) == 0:
                assert node.op in (
                    "placeholder",
                    "output",
                ), f"{self.__class__.__name__} reached an endpoint node which is not a placeholder or output: {frontier}"
                break

            if len(neighbours) > 1:
                break

            neighbour = neighbours[0]
            if self.is_elementwise(neighbour) and self._is_unary_elementwise(neighbour):
                frontier = neighbour
                moved = True
                continue

            if self.is_swappable(neighbour):
                swapped_args = self._maybe_swap_args(node, neighbour)
                if swapped_args is None:
                    break
                node.args = swapped_args[0]
                neighbour.args = swapped_args[1]
                frontier = neighbour
                moved = True
                continue

            # Unhandled case, stop propagation
            break

        if not moved:
            return False

        self._move_node(node, frontier)
        return True

    def fuse_vertical(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Fuse consecutive permute/view nodes."""
        return CanonicalizeViewCopyPermutePass().call(graph_module)

    def fuse_horizontal(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Fuse parallel permute/view nodes going into/ out a single node."""
        raise NotImplementedError()

    def _run_horizontal_passes(
        self,
        graph_module: torch.fx.GraphModule,
        passes: Sequence[ExportPass],
    ) -> PassResult:
        modified = False
        for pass_ in passes:
            result = pass_.call(graph_module)
            graph_module = result.graph_module
            modified |= result.modified
        return PassResult(graph_module, modified)

    def _get_neighbours(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        """Return the next nodes in the direction of propagation."""
        raise NotImplementedError()

    def _maybe_swap_args(
        self, node: torch.fx.Node, neighbour: torch.fx.Node
    ) -> Any | None:
        """If the node can be swapped with its neighbour, return the new args
        for the neighbour and new shape, otherwise return None.
        """
        if node.target == self._PERMUTE_TARGET:
            return self._maybe_swap_permute_args(node, neighbour)
        elif node.target in {self._VIEW_TARGET, self._VIEW_DEFAULT_TARGET}:
            return self._maybe_swap_view_args(node, neighbour)
        else:
            raise ValueError(
                f"Unexpected node target {node.target} in {self.__class__.__name__}"
            )

    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, neighbour: torch.fx.Node
    ) -> Any | None:
        raise NotImplementedError()

    def _maybe_swap_view_args(
        self, node: torch.fx.Node, neighbour: torch.fx.Node
    ) -> Any | None:
        raise NotImplementedError()

    def _move_node(self, node: torch.fx.Node, frontier: torch.fx.Node) -> None:
        """Update the graph to move the node into its new position."""
        raise NotImplementedError()

    def is_elementwise(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False

        if node.target in {
            exir_ops.backend.tosa.RESCALE.default,
            exir_ops.backend.tosa.TABLE.default,
        }:
            return True

        op = getattr(node.target, "_op", None)
        if op is not None and hasattr(op, "tags"):
            return torch.Tag.pointwise in op.tags
        return False

    def is_swappable(self, neighbour: torch.fx.Node) -> bool:
        return neighbour.target in self._ARG_UPDATE_TARGETS

    def _is_unary_elementwise(self, node: torch.fx.Node) -> bool:
        return len(node.all_input_nodes) == 1


class PropagatePermutesDownConcatPass(ArmPass):
    """Sink matching input permutes through concat."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    _CAT_TARGETS = {exir_ops.edge.aten.cat.default, torch.ops.aten.cat.default}
    _PERMUTE_TARGETS = {
        exir_ops.edge.aten.permute_copy.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.permute.default,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        while True:
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                iteration_modified |= self._sink_matching_input_permutes(node)

            if not iteration_modified:
                break
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _sink_matching_input_permutes(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target not in self._CAT_TARGETS:
            return False

        cat_inputs = self._cat_inputs(node)
        if cat_inputs is None or len(cat_inputs) == 0:
            return False
        if any(
            input_node.target not in self._PERMUTE_TARGETS for input_node in cat_inputs
        ):
            return False

        permute_dims = self._permutation(cat_inputs[0])
        if permute_dims is None:
            return False
        if any(
            self._permutation(input_node) != permute_dims for input_node in cat_inputs
        ):
            return False
        if any(
            any(user is not node for user in input_node.users)
            for input_node in cat_inputs
        ):
            return False

        cat_dim = self._cat_dim(node)
        if cat_dim is None:
            return False
        new_cat_dim = PermuteMap(cat_inputs[0]).map_target_to_source(cat_dim)[0]

        producers = [
            cast(torch.fx.Node, input_node.args[0]) for input_node in cat_inputs
        ]
        output_val = node.meta.get("val")
        if not isinstance(output_val, torch.Tensor):
            return False

        node.args = (producers, new_cat_dim, *node.args[2:])
        node.meta = copy.copy(node.meta)
        node.meta["val"] = self._cat_output_meta_val(producers, new_cat_dim, output_val)

        users = list(node.users)
        with node.graph.inserting_after(node):
            permute = node.graph.call_function(
                cast(Callable[..., Any], cat_inputs[0].target),
                args=(node, permute_dims),
                kwargs=dict(cat_inputs[0].kwargs),
            )
        permute.meta = copy.copy(cat_inputs[0].meta)
        permute.meta["val"] = output_val

        for user in users:
            user.replace_input_with(node, permute)

        for input_node in cat_inputs:
            if len(input_node.users) == 0:
                node.graph.erase_node(input_node)

        return True

    def _cat_inputs(self, node: torch.fx.Node) -> list[torch.fx.Node] | None:
        if len(node.args) == 0 or not isinstance(node.args[0], (list, tuple)):
            return None
        inputs = node.args[0]
        if not all(isinstance(input_node, torch.fx.Node) for input_node in inputs):
            return None
        return list(cast(Sequence[torch.fx.Node], inputs))

    def _cat_dim(self, node: torch.fx.Node) -> int | None:
        dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim", 0)
        return dim if isinstance(dim, int) else None

    def _permutation(self, node: torch.fx.Node) -> list[int] | None:
        if node.target not in self._PERMUTE_TARGETS or len(node.args) < 2:
            return None

        dims = list(cast(Sequence[int], node.args[1]))
        rank = len(dims)
        normalized_dims = [dim + rank if dim < 0 else dim for dim in dims]
        if sorted(normalized_dims) != list(range(rank)):
            return None
        return normalized_dims

    def _cat_output_meta_val(
        self,
        producers: list[torch.fx.Node],
        cat_dim: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        producer_vals = [producer.meta.get("val") for producer in producers]
        if not all(isinstance(val, torch.Tensor) for val in producer_vals):
            return fallback

        producer_shapes = [
            list(cast(torch.Tensor, producer_val).shape)
            for producer_val in producer_vals
        ]
        output_shape = producer_shapes[0]
        output_shape[cat_dim] = sum(shape[cat_dim] for shape in producer_shapes)
        return fallback.new_empty(tuple(output_shape))


class PropagatePermuteUpConcatPass(ArmPass):
    """Hoist output permutes across concat when input permutes are no-ops."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    _CAT_TARGETS = {exir_ops.edge.aten.cat.default, torch.ops.aten.cat.default}
    _PERMUTE_TARGETS = {
        exir_ops.edge.aten.permute_copy.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.permute.default,
    }
    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        while True:
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                iteration_modified |= self._hoist_noop_input_permutes(node)

            if not iteration_modified:
                break
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _hoist_noop_input_permutes(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function" or node.target not in self._PERMUTE_TARGETS:
            return False
        if len(node.all_input_nodes) != 1:
            return False

        cat_node = node.all_input_nodes[0]
        if cat_node.target not in self._CAT_TARGETS or any(
            user is not node for user in cat_node.users
        ):
            return False

        permutation = self._permutation(node)
        if permutation is None:
            return False

        cat_inputs = self._cat_inputs(cat_node)
        if cat_inputs is None or len(cat_inputs) == 0:
            return False

        cat_dim = self._cat_dim(cat_node)
        if cat_dim is None:
            return False
        new_cat_dim = PermuteMap(node).map_source_to_target(cat_dim)[0]

        if any(
            not self._is_data_movement_free_permutation(input_node, permutation)
            for input_node in cat_inputs
        ):
            return False

        new_inputs = [
            self._maybe_insert_view_for_permute(input_node, permutation, cat_node)
            for input_node in cat_inputs
        ]
        cat_node.args = (new_inputs, new_cat_dim, *cat_node.args[2:])
        cat_node.meta = copy.copy(node.meta)

        node.replace_all_uses_with(cat_node)
        node.graph.erase_node(node)
        return True

    def _cat_inputs(self, node: torch.fx.Node) -> list[torch.fx.Node] | None:
        if len(node.args) == 0 or not isinstance(node.args[0], (list, tuple)):
            return None
        inputs = node.args[0]
        if not all(isinstance(input_node, torch.fx.Node) for input_node in inputs):
            return None
        return list(cast(Sequence[torch.fx.Node], inputs))

    def _cat_dim(self, node: torch.fx.Node) -> int | None:
        dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim", 0)
        return dim if isinstance(dim, int) else None

    def _permutation(self, node: torch.fx.Node) -> list[int] | None:
        if node.target not in self._PERMUTE_TARGETS or len(node.args) < 2:
            return None

        dims = list(cast(Sequence[int], node.args[1]))
        rank = len(dims)
        normalized_dims = [dim + rank if dim < 0 else dim for dim in dims]
        if sorted(normalized_dims) != list(range(rank)):
            return None
        return normalized_dims

    def _is_data_movement_free_permutation(
        self, node: torch.fx.Node, permutation: Sequence[int]
    ) -> bool:
        input_val = node.meta.get("val")
        if not isinstance(input_val, torch.Tensor):
            return False
        return CanonicalizeViewCopyPermutePass._is_singleton_permutation(
            input_val.shape, permutation
        )

    def _maybe_insert_view_for_permute(
        self,
        node: torch.fx.Node,
        permutation: Sequence[int],
        insert_before: torch.fx.Node,
    ) -> torch.fx.Node:
        input_val = node.meta.get("val")
        if not isinstance(input_val, torch.Tensor):
            return node

        output_shape = [input_val.shape[dim] for dim in permutation]
        if ViewMap.shapes_equal(input_val.shape, output_shape):
            return node

        with node.graph.inserting_before(insert_before):
            view = node.graph.call_function(
                self._VIEW_TARGET,
                args=(node, output_shape),
            )
        view.meta = copy.copy(node.meta)
        view.meta["val"] = input_val.new_empty(tuple(output_shape))
        return view


class PropagatePermuteViewsUpPass(PropagatePermuteViewsPass):
    """Move permute/view nodes towards graph inputs."""

    def fuse_horizontal(self, graph_module):
        return self._run_horizontal_passes(
            graph_module,
            (
                FuseIdenticalInputTransformsPass(),
                FuseDuplicateUsersPass(),
                PropagatePermutesDownConcatPass(),
                PropagatePermuteUpConcatPass(),
            ),
        )

    def _get_neighbours(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.all_input_nodes)

    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, neighbour: torch.fx.Node
    ) -> Any | None:
        permute_map = PermuteMap(node)
        args = self._dim_arg(neighbour.args[1])
        if args is None:
            return None
        mapped_args = permute_map.map_source_to_target(args)
        new_args: int | list[int] = (
            mapped_args[0] if isinstance(args, int) else mapped_args
        )
        return (node.args, (*neighbour.args[:1], new_args, *neighbour.args[2:]))

    def _maybe_swap_view_args(self, node, neighbour):
        view_map = ViewMap(node)
        if not view_map.is_valid_map or len(neighbour.all_input_nodes) != 1:
            return None

        input_val = neighbour.all_input_nodes[0].meta["val"]
        new_shape = view_map.target_shape_for_source_shape(list(input_val.shape))
        if new_shape is None:
            return None

        if neighbour.target in self._REDUCTION_TARGETS:
            if len(neighbour.args) <= 2 or neighbour.args[2] is not True:
                return None
            new_dims = view_map.map_source_to_target(neighbour.args[1])
            if not view_map.validate_mapped_reduction(new_dims):
                return None
        elif neighbour.target == exir_ops.edge.aten.slice_copy.Tensor:
            new_dims = view_map.map_source_to_target(neighbour.args[1])
            if len(new_dims) != 1:
                return None
            new_dims = new_dims[0]
        else:
            return None

        new_neighbour_args = (*neighbour.args[:1], new_dims, *neighbour.args[2:])
        return ((*node.args[:1], new_shape), new_neighbour_args)

    def _move_node(self, node: torch.fx.Node, frontier: torch.fx.Node) -> None:
        original_input = node.all_input_nodes[0]
        if frontier.op == "placeholder":
            producer = frontier
            frontier_user = next(
                user for user in list(frontier.users) if user is not node
            )
        else:
            producer = frontier.all_input_nodes[0]
            frontier_user = frontier

        node.replace_input_with(original_input, producer)
        frontier_user.replace_input_with(producer, node)

        for user in list(node.users):
            if user is not frontier_user:
                user.replace_input_with(node, original_input)

        frontier_user.prepend(node)


class PropagatePermuteViewsDownPass(PropagatePermuteViewsPass):
    """Move permute/view nodes towards graph outputs."""

    def fuse_horizontal(self, graph_module):
        return self._run_horizontal_passes(
            graph_module,
            (
                FuseIdenticalInputTransformsPass(),
                FuseDuplicateUsersPass(),
                PropagatePermutesDownConcatPass(),
                PropagatePermuteUpConcatPass(),
            ),
        )

    def _get_neighbours(self, node: torch.fx.Node) -> Iterable[torch.fx.Node]:
        return list(node.users.keys())

    def _maybe_swap_permute_args(
        self, node: torch.fx.Node, neighbour: torch.fx.Node
    ) -> Any | None:
        permute_map = PermuteMap(node)
        args = self._dim_arg(neighbour.args[1])
        if args is None:
            return None
        mapped_args = permute_map.map_target_to_source(args)
        new_args: int | list[int] = (
            mapped_args[0] if isinstance(args, int) else mapped_args
        )
        return (node.args, (*neighbour.args[:1], new_args, *neighbour.args[2:]))

    def _maybe_swap_view_args(self, node, neighbour):
        view_map = ViewMap(node)
        if not view_map.is_valid_map:
            return None

        if neighbour.target in self._REDUCTION_TARGETS:
            if len(neighbour.args) <= 2 or neighbour.args[2] is not True:
                return None
            new_dims = view_map.map_target_to_source(neighbour.args[1])
            if not view_map.validate_mapped_source_reduction(new_dims):
                return None
        elif neighbour.target == exir_ops.edge.aten.slice_copy.Tensor:
            new_dims = view_map.map_target_to_source(neighbour.args[1])
            if len(new_dims) != 1:
                return None
            new_dims = new_dims[0]
        else:
            return None

        output_val = neighbour.meta["val"]
        new_neighbour_args = (*neighbour.args[:1], new_dims, *neighbour.args[2:])
        return ((*node.args[:1], list(output_val.shape)), new_neighbour_args)

    def _move_node(self, node: torch.fx.Node, frontier: torch.fx.Node) -> None:
        original_user = next(iter(node.users))
        producer = node.all_input_nodes[0]
        if frontier.op == "output":
            frontier_input = frontier.all_input_nodes[0]
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
