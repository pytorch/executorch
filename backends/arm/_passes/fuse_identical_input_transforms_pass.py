# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
from typing import Any, cast, Iterable, Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class FuseIdenticalInputTransformsPass(ArmPass):
    """Sink identical input transforms through pointwise ops.

    Example:

        add(permute(x), permute(y))

    becomes:

        permute(add(x, y))

    The pass is intentionally limited to pointwise/elementwise nodes.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _VIEW_DEFAULT_TARGET = exir_ops.edge.aten.view.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _VIEW_DEFAULT_TARGET, _PERMUTE_TARGET}

    _ELEMENTWISE_TARGETS = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.clamp.default,
    }

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        while True:
            iteration_modified = False
            for node in list(graph.nodes):
                while self._sink_identical_input_transforms(node):
                    iteration_modified = True

            if not iteration_modified:
                break
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _sink_identical_input_transforms(self, node: Node) -> bool:
        if not self._is_elementwise_node(node):
            return False

        input_transforms = self._identical_input_transforms(node)
        if input_transforms is None:
            return False

        producers_and_shapes = self._producer_nodes_and_shapes(input_transforms)
        if producers_and_shapes is None:
            return False
        producers, producer_shapes = producers_and_shapes

        transform = input_transforms[0]
        transform_args = self._transform_args_for_node(transform, node)
        if transform_args is None:
            return False

        node_val = node.meta.get("val")
        if not isinstance(node_val, torch.Tensor):
            return False

        for input_transform, producer in zip(input_transforms, producers):
            node.replace_input_with(input_transform, producer)
        for input_transform in input_transforms:
            if len(input_transform.users) == 0:
                node.graph.erase_node(input_transform)

        node.meta = copy.copy(node.meta)
        node.meta["val"] = node_val.new_empty(producer_shapes[0])

        with node.graph.inserting_after(node):
            new_node = node.graph.call_function(
                cast(Any, transform.target),
                args=transform_args,
                kwargs=dict(transform.kwargs),
            )
        new_node.meta = self._meta_from_input(new_node, transform)

        for user in list(node.users):
            if user is not new_node:
                user.replace_input_with(node, new_node)

        return True

    def _identical_input_transforms(self, node: Node) -> list[Node] | None:
        input_transforms = list(node.all_input_nodes)
        if len(input_transforms) < 2:
            return None
        if any(
            not self._is_transform_node(input_node) for input_node in input_transforms
        ):
            return None
        if len(set(input_transforms)) != len(input_transforms):
            return None

        transform = input_transforms[0]
        if any(
            not self._is_equivalent_transform(input_node, transform)
            for input_node in input_transforms
        ):
            return None
        if any(
            any(user is not node for user in input_transform.users)
            for input_transform in input_transforms
        ):
            return None
        return input_transforms

    def _producer_nodes_and_shapes(
        self, input_transforms: list[Node]
    ) -> tuple[list[Node], list[tuple[int, ...]]] | None:
        producers = []
        producer_shapes = []
        for input_transform in input_transforms:
            producer = self._node_input(input_transform)
            producer_val = producer.meta.get("val") if producer is not None else None
            if producer is None or not isinstance(producer_val, torch.Tensor):
                return None
            producers.append(producer)
            producer_shapes.append(tuple(producer_val.shape))

        if any(shape != producer_shapes[0] for shape in producer_shapes):
            return None
        return producers, producer_shapes

    def _is_elementwise_node(self, node: Node) -> bool:
        if node.op != "call_function":
            return False
        if node.target in self._ELEMENTWISE_TARGETS:
            return True

        op = getattr(node.target, "_op", None)
        return op is not None and hasattr(op, "tags") and torch.Tag.pointwise in op.tags

    def _is_transform_node(self, node: Node) -> bool:
        return node.op == "call_function" and node.target in self._TARGETS

    def _node_input(self, node: Node) -> Node | None:
        input_node = node.args[0] if len(node.args) > 0 else node.kwargs.get("input")
        return input_node if isinstance(input_node, Node) else None

    def _is_equivalent_transform(self, node: Node, transform: Node) -> bool:
        if not self._is_transform_node(node) or node.target != transform.target:
            return False
        if not self._has_matching_transform_ranks(node, transform):
            return False
        return self._transform_signature(node) == self._transform_signature(transform)

    def _has_matching_transform_ranks(self, node: Node, transform: Node) -> bool:
        node_input = self._node_input(node)
        transform_input = self._node_input(transform)
        node_input_val = node_input.meta.get("val") if node_input is not None else None
        transform_input_val = (
            transform_input.meta.get("val") if transform_input is not None else None
        )
        node_output_val = node.meta.get("val")
        transform_output_val = transform.meta.get("val")
        if not (
            isinstance(node_input_val, torch.Tensor)
            and isinstance(transform_input_val, torch.Tensor)
            and isinstance(node_output_val, torch.Tensor)
            and isinstance(transform_output_val, torch.Tensor)
        ):
            return False
        return len(node_input_val.shape) == len(transform_input_val.shape) and len(
            node_output_val.shape
        ) == len(transform_output_val.shape)

    def _transform_signature(self, node: Node) -> tuple[tuple[Any, ...], Any]:
        return (tuple(node.args[1:]), tuple(sorted(node.kwargs.items())))

    def _transform_args_for_node(
        self, transform: Node, node: Node
    ) -> tuple[Any, ...] | None:
        if transform.target == self._PERMUTE_TARGET:
            permutation = self._get_permutation(transform)
            node_val = node.meta.get("val")
            if (
                permutation is None
                or not isinstance(node_val, torch.Tensor)
                or len(node_val.shape) != len(permutation)
            ):
                return None
            return (node, *transform.args[1:])

        if transform.target in (self._VIEW_TARGET, self._VIEW_DEFAULT_TARGET):
            node_val = node.meta.get("val")
            transform_val = transform.meta.get("val")
            if not (
                isinstance(node_val, torch.Tensor)
                and isinstance(transform_val, torch.Tensor)
                and self._same_numel(node_val.shape, transform_val.shape)
            ):
                return None
            return (node, *transform.args[1:])

        return None

    def _meta_from_input(self, new_node: Node, original_node: Node) -> dict:
        node_meta = copy.copy(original_node.meta)
        node_val = node_meta.get("val")
        input_node = self._node_input(new_node)
        input_val = input_node.meta.get("val") if input_node is not None else None
        if isinstance(node_val, torch.Tensor) and isinstance(input_val, torch.Tensor):
            if new_node.target == self._PERMUTE_TARGET:
                permutation = self._get_permutation(new_node)
                if permutation is not None:
                    try:
                        node_meta["val"] = node_val.new_empty(
                            tuple(input_val.shape[dim] for dim in permutation)
                        )
                    except RuntimeError:
                        pass
            elif len(new_node.args) > 1:
                try:
                    node_meta["val"] = node_val.new_empty(
                        tuple(cast(Iterable[int], new_node.args[1]))
                    )
                except RuntimeError:
                    pass
        return node_meta

    def _get_permutation(self, permute_node: Node) -> list[int] | None:
        if permute_node.target != self._PERMUTE_TARGET:
            return None
        if len(permute_node.args) >= 2:
            raw_permute = list(cast(list[int], permute_node.args[1]))
        else:
            raw_dims = permute_node.kwargs.get("dims", permute_node.kwargs.get("dim"))
            if raw_dims is None:
                return None
            raw_permute = list(cast(list[int], raw_dims))

        rank = len(raw_permute)
        normalized_permute = [dim + rank if dim < 0 else dim for dim in raw_permute]
        if sorted(normalized_permute) != list(range(rank)):
            return None
        return normalized_permute

    def _same_numel(self, first: Iterable[Any], second: Iterable[Any]) -> bool:
        first_numel = 1
        for dim in first:
            first_numel *= dim
        second_numel = 1
        for dim in second:
            second_numel *= dim
        return first_numel == second_numel
