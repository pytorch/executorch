# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections.abc import Sequence
from typing import Any, Callable, cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import refresh_node_meta
from executorch.backends.arm._passes.dim_maps import (
    _dim_equals,
    PermuteMap,
    same_numel,
    ViewMap,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputSpec, TensorArgument
from torch.fx import GraphModule, Node


class NormalizeTransformInputPlaceholdersPass(ExportPass):
    """Normalize placeholder names for lifted transform inputs."""

    def __init__(self, exported_program: ExportedProgram | None = None) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(
            graph_module,
            self._normalize_transform_user_input_placeholders(graph_module),
        )

    def _normalize_transform_user_input_placeholders(
        self, graph_module: GraphModule
    ) -> bool:
        if self.exported_program is None:
            return False

        signature = self.exported_program.graph_signature
        placeholder_names = {
            node.name for node in graph_module.graph.nodes if node.op == "placeholder"
        }
        name_updates: dict[str, str] = {}
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if not self._placeholder_represents_transform(node):
                continue
            old_target = str(node.target)

            if node.target != node.name:
                node.target = node.name
                modified = True
                if old_target not in placeholder_names:
                    name_updates[old_target] = node.name

        if not modified:
            return False

        if name_updates:
            signature.input_specs = [
                self._renamed_input_spec(spec, name_updates)
                for spec in signature.input_specs
            ]
        graph_module.graph.lint()
        graph_module.recompile()
        return True

    def _placeholder_represents_transform(self, node: Node) -> bool:
        return any(
            self._matches_transform_placeholder_name(node.name, target)
            or self._matches_transform_placeholder_name(str(node.target), target)
            for target in FuseIdenticalInputTransformsPass._TARGETS
        )

    def _matches_transform_placeholder_name(
        self, name: str, target: torch.fx.node.Target
    ) -> bool:
        base_name = self._target_placeholder_name(target)
        if name == base_name:
            return True

        suffix = name.removeprefix(f"{base_name}_")
        return suffix != name and suffix.isdecimal()

    def _target_placeholder_name(self, target: torch.fx.node.Target) -> str:
        return target.__name__.replace(".", "_")  # type: ignore[union-attr]

    def _renamed_input_spec(
        self, spec: InputSpec, name_updates: dict[str, str]
    ) -> InputSpec:
        if isinstance(spec.arg, TensorArgument) and spec.arg.name in name_updates:
            return InputSpec(
                kind=spec.kind,
                arg=TensorArgument(name=name_updates[spec.arg.name]),
                target=spec.target,
                persistent=spec.persistent,
            )
        return spec


class FuseIdenticalInputTransformsPass(ArmOpTargetedPass):
    """Sink identical input transforms through pointwise ops with multiple
    inputs. Note that this is only valid for data movement transforms; most
    operators cannot be swapped in this way while preserving semantics.

    Example:
        add(permute(x), permute(y))
    becomes:
        permute(add(x, y))

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGET = exir_ops.edge.aten.view_copy.default
    _PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
    _TARGETS = {_VIEW_TARGET, _PERMUTE_TARGET}
    _CONCAT_OPS = {
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.concatenate.default,
    }

    _BINARY_ELEMENTWISE_OPS = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.remainder.Tensor,
    }
    _NARY_ELEMENTWISE_OPS = {
        exir_ops.edge.aten.where.self,
    }
    _ELEMENTWISE_OPS = _BINARY_ELEMENTWISE_OPS | _NARY_ELEMENTWISE_OPS

    target_ops = _ELEMENTWISE_OPS | _CONCAT_OPS

    def __init__(self, exported_program: ExportedProgram | None = None) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = (
            NormalizeTransformInputPlaceholdersPass(self.exported_program)
            .call(graph_module)
            .modified
        )

        while True:
            iteration_modified = False
            for node in list(graph_module.graph.nodes):
                if node.op != "call_function":
                    continue
                while self._sink_identical_input_transforms(node):
                    iteration_modified = True

            if not iteration_modified:
                break
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)

    def _sink_identical_input_transforms(self, node: Node) -> bool:
        if node.target not in self.target_ops:
            return False

        input_nodes = list(node.all_input_nodes)
        if len(input_nodes) < 2:
            return False

        node_val = node.meta.get("val", None)
        if node_val is None:
            return False

        transforms = [n for n in input_nodes if n.target in self._TARGETS]
        if not transforms:
            return False
        transform = transforms[0]
        if not self._inputs_share_transform_or_are_layout_invariant(
            node, transform, input_nodes
        ):
            return False

        updated_args = self._updated_node_args(node, transform, node_val, input_nodes)
        if updated_args is None:
            return False
        node_args, node_kwargs, transform_args, node_output_shape = updated_args

        # Remove input transforms
        producers = [
            n.all_input_nodes[0] if n.target in self._TARGETS else n
            for n in input_nodes
        ]

        node.args = node_args
        node.kwargs = node_kwargs
        for input_transform, producer in zip(input_nodes, producers):
            node.replace_input_with(input_transform, producer)
        for input_transform in dict.fromkeys(input_nodes):
            if (
                input_transform.target in self._TARGETS
                and len(input_transform.users) == 0
            ):
                node.graph.erase_node(input_transform)

        node.meta = copy.copy(node.meta)
        node.meta["val"] = node_val.new_empty(node_output_shape)

        # Insert new transform after binary elementwise op
        with node.graph.inserting_after(node):
            new_node = node.graph.call_function(
                cast(Callable[..., Any], transform.target),
                args=transform_args,
                kwargs=dict(transform.kwargs),
            )
        new_node.meta = self._new_transform_meta(node, transform)
        refresh_node_meta(new_node)

        for user in list(node.users):
            if user is not new_node:
                user.replace_input_with(node, new_node)

        return True

    def _new_transform_meta(self, node: Node, transform: Node) -> dict[str, Any]:
        meta = copy.copy(transform.meta)
        if "delegation_tag" in node.meta:
            meta["delegation_tag"] = node.meta["delegation_tag"]
        else:
            meta.pop("delegation_tag", None)
        return meta

    def _updated_node_args(
        self, node: Node, transform: Node, node_val: Any, input_nodes: list[Node]
    ) -> (
        tuple[tuple[Any, ...], dict[str, Any], tuple[Any, ...], tuple[Any, ...]] | None
    ):
        if node.target in self._BINARY_ELEMENTWISE_OPS:
            return self._update_node_args_binary(node, transform, node_val, input_nodes)

        if node.target in self._CONCAT_OPS:
            return self._update_node_args_concat(node, transform, node_val, input_nodes)

        if node.target in self._NARY_ELEMENTWISE_OPS:
            return self._update_node_args_binary(node, transform, node_val, input_nodes)

        return None

    def _inputs_share_transform_or_are_layout_invariant(
        self, node: Node, transform: Node, input_nodes: list[Node]
    ) -> bool:
        transforms = [n for n in input_nodes if n.target in self._TARGETS]
        if not self._transforms_are_identical(transforms):
            return False
        if not self._transforms_only_used_by_node(node, transforms):
            return False
        if len(transforms) == len(input_nodes):
            return True
        if node.target not in self._ELEMENTWISE_OPS:
            return False

        transform_val = transform.meta.get("val")
        if not isinstance(transform_val, torch.Tensor):
            return False
        rank = len(transform_val.shape)
        return all(
            input_node in transforms or self.is_layout_invariant(input_node, rank)
            for input_node in input_nodes
        )

    @staticmethod
    def is_layout_invariant(node: Node, rank: int) -> bool:
        value = node.meta.get("val")
        return (
            isinstance(value, torch.Tensor)
            and len(value.shape) == rank
            and all(_dim_equals(dim, 1) for dim in value.shape)
        )

    def _transforms_are_identical(self, input_transforms: list[Node]) -> bool:
        target = input_transforms[0].target
        if target not in self._TARGETS:
            return False
        if not all(
            input_transform.target == target for input_transform in input_transforms
        ):
            return False

        transform_arg = input_transforms[0].args[1]
        return all(
            input_transform.args[1] == transform_arg
            for input_transform in input_transforms
        )

    def _transforms_only_used_by_node(
        self, node: Node, input_transforms: list[Node]
    ) -> bool:
        return all(
            all(user is node for user in input_transform.users)
            for input_transform in input_transforms
        )

    def _update_node_args_binary(self, node, transform, node_val, input_transforms):
        producer_shapes = [
            tuple(
                (
                    input_node.all_input_nodes[0]
                    if input_node.target in self._TARGETS
                    else input_node
                )
                .meta["val"]
                .shape
            )
            for input_node in input_transforms
        ]

        try:
            node_output_shape = tuple(torch.broadcast_shapes(*producer_shapes))
        except RuntimeError:
            return None

        transform_args = (node, *transform.args[1:])
        if transform.target == self._VIEW_TARGET:
            transform_args = (node, list(node_val.shape))
            # Reshaping before an elementwise op can change which dimensions
            # broadcast. Sinking is safe only when the broadcast in the source
            # layout already has exactly one of the producer shapes.
            if node_output_shape not in producer_shapes:
                return None

        if transform.target == self._PERMUTE_TARGET:
            dims = cast(Sequence[int], transform.args[1])
            rank = len(node_output_shape)
            normalized_dims = tuple(dim if dim >= 0 else dim + rank for dim in dims)
            if tuple(node_val.shape) != tuple(
                node_output_shape[dim] for dim in normalized_dims
            ):
                return None

        return node.args, dict(node.kwargs), transform_args, node_output_shape

    def _update_node_args_concat(self, node, transform, node_val, input_transforms):
        concat_dim = self._concat_dim(node)
        if concat_dim is None:
            return None

        new_concat_dim = self._mapped_concat_dim(transform, concat_dim)
        if new_concat_dim is None:
            return None

        node_args = (node.args[0], new_concat_dim, *node.args[2:])
        node_kwargs = dict(node.kwargs)
        node_kwargs.pop("dim", None)

        producer_shapes = [
            tuple(input_transform.all_input_nodes[0].meta["val"].shape)
            for input_transform in input_transforms
        ]
        if not self._concat_producer_shapes_match(producer_shapes, new_concat_dim):
            return None

        node_output_shape = list(producer_shapes[0])
        node_output_shape[new_concat_dim] = sum(
            shape[new_concat_dim] for shape in producer_shapes
        )

        transform_args = (node, *transform.args[1:])
        if transform.target == self._VIEW_TARGET:
            if not same_numel(node_output_shape, node_val.shape):
                return None
            transform_args = (node, list(node_val.shape))

        if transform.target == self._PERMUTE_TARGET:
            dims = cast(Sequence[int], transform.args[1])
            rank = len(node_output_shape)
            normalized_dims = tuple(dim if dim >= 0 else dim + rank for dim in dims)
            if tuple(node_val.shape) != tuple(
                node_output_shape[dim] for dim in normalized_dims
            ):
                return None

        return node_args, node_kwargs, transform_args, tuple(node_output_shape)

    def _concat_producer_shapes_match(
        self, producer_shapes: list[tuple[Any, ...]], concat_dim: int
    ) -> bool:
        reference_shape = producer_shapes[0]
        rank = len(reference_shape)
        normalized_concat_dim = concat_dim if concat_dim >= 0 else concat_dim + rank
        if not 0 <= normalized_concat_dim < rank:
            return False

        return all(
            len(shape) == rank
            and all(
                dim == normalized_concat_dim or shape[dim] == reference_shape[dim]
                for dim in range(rank)
            )
            for shape in producer_shapes
        )

    def _concat_dim(self, node: Node) -> int | None:
        dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim", 0)
        return dim if isinstance(dim, int) else None

    def _mapped_concat_dim(self, transform: Node, concat_dim: int) -> int | None:
        if transform.target == self._PERMUTE_TARGET:
            return PermuteMap(transform).map_dims_inverse(concat_dim)[0]

        view_map = ViewMap(transform)
        if not view_map.is_valid_map:
            return None
        mapped_dims = view_map.map_dim_inverse(concat_dim)
        if mapped_dims is None or len(mapped_dims) != 1:
            return None
        return mapped_dims[0]
