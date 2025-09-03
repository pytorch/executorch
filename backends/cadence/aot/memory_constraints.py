# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, cast, DefaultDict, Iterable, Optional, Sequence, TypeAlias

import torch
import torch.fx
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    create_cadence_pass_filter,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.utils import get_shape, is_node_in_flattened_output
from executorch.exir import memory
from executorch.exir.pass_manager import PassManager
from executorch.exir.tensor import num_bytes_from_shape_and_dtype, TensorSpec
from torch.fx.passes.infra.pass_base import PassBase, PassResult


@dataclass(frozen=True)
class RelativePlacementConstraint:
    """Information of source node and offset used for views."""

    source: torch.fx.Node
    offset: int = 0


@dataclass(frozen=True)
class AbsolutePlacementConstraint:
    """Information on placement constraint memory id and offset."""

    pinned_memory_id: int

    # If offset is None, then the tensor can be placed anywhere in the memory id.
    offset: Optional[int] = None


class MemConstraints:
    """
    This class contains all the tensor placement constraints that we create
    during memory planning.

    We have two types of placement constraints:
    1. Relative placement constraints: These are constraints that specify the
       relative placement of a tensor with respect to another tensor. For
       example, when slice dim is 0, slice output can be placed relative to
       their inputs and the op can be replaced with a nop.
    2. Absolute placement constraints: These are constraints that specify the
       absolute placement of a tensor either in a specific memory id, or both
       a specific memory id and offset. For example, for operators that require
       a specific memory id + offset for we can use this constraint to specify
       location of inputs/outputs or even temporary buffers.
    """

    def __init__(
        self,
        opt_level: int = 1,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
    ) -> None:
        self.opt_level = opt_level
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output

        # Relative location constraints, indicating that tensor x is at offset
        # o from tensor y.
        self.unresolved_loc_constraints: DefaultDict[int, set[torch.fx.Node]] = (
            defaultdict(lambda: set())
        )

        # A set of tensor spec ids that must be skipped during memory allocation.
        # The exact mem_id and offset of the skipped tensors will be computed from
        # the constraints.
        self._relative_placement_constraint: dict[int, RelativePlacementConstraint] = {}

        # A map from `id(TensorSpec)` to a set of mem_ids that cannot be used for
        # allocating the tensor.
        self._mem_id_blocklist: dict[int, set[int]] = {}

        # A map from `id(TensorSpec)` to a AbsolutePlacementConstraint that specifies mem_id and optionally exact offset.
        self._absolute_placement_constraints: dict[int, AbsolutePlacementConstraint] = (
            {}
        )

    def get_relative_placement_source(
        self, node: torch.fx.Node
    ) -> Optional[RelativePlacementConstraint]:
        spec = node.meta.get("spec")
        spec_id = id(spec)
        if spec_id not in self._relative_placement_constraint:
            return None
        return self._relative_placement_constraint[spec_id]

    def set_relative_placement_constraint(
        self,
        dependent: torch.fx.Node,
        placement_constraint: RelativePlacementConstraint,
    ) -> None:
        dependent_spec = dependent.meta.get("spec")
        spec_id = id(dependent_spec)
        self._relative_placement_constraint[spec_id] = placement_constraint
        if self.is_memory_planned(placement_constraint.source):
            # Only add dependent nodes if source node needs memory planning.
            self.unresolved_loc_constraints[
                id(placement_constraint.source.meta.get("spec"))
            ].add(dependent)

    def add_mem_id_to_blocklist(self, spec: TensorSpec, mem_id: int) -> None:
        spec_id = id(spec)
        if spec_id not in self._mem_id_blocklist:
            self._mem_id_blocklist[spec_id] = set()
        self._mem_id_blocklist[spec_id].add(mem_id)

    def is_mem_id_in_blocklist(self, spec: TensorSpec, mem_id: int) -> bool:
        spec_id = id(spec)
        if spec_id not in self._mem_id_blocklist:
            return False
        return mem_id in self._mem_id_blocklist[spec_id]

    def is_alias_of(self, node: torch.fx.Node, other_node: torch.fx.Node) -> bool:
        """Returns true if `spec` is an alias of `other_spec`.
        Two specs are considered aliases if they have the same number of
        elements and dtype.

        TODO(hardiksharma): aliases should not allow in-place modification.
        For the following case, either the view can be an alias, or the relu can
        be in-place. But not both.

        node --> view
             --> relu (or some other op that can be in-place)
        """
        if node_source_info := self.get_relative_placement_source(node):
            node_spec = node.meta.get("spec")
            node_source_spec = node_source_info.source.meta.get("spec")
            return (
                node_source_info.offset == 0
                and math.prod(node_source_spec.shape) == math.prod(node_spec.shape)  # type: ignore[union-attr]
                and node_source_spec.dtype == node_spec.dtype  # type: ignore[union-attr]
                and self.is_alias_of(node_source_info.source, other_node)
            )

        if self.get_relative_placement_source(other_node) is not None:
            return self.is_alias_of(other_node, node)

        return node == other_node

    # Return true if the unresolved_loc_constraints is empty
    def relative_loc_constraints_exist(self) -> bool:
        return len(self.unresolved_loc_constraints) != 0

    # Return true if the spec is marked as skipped
    def skipped_spec(self, spec: TensorSpec) -> bool:
        return id(spec) in self._relative_placement_constraint

    def is_memory_planned(
        self,
        node: torch.fx.Node,
    ) -> bool:
        """Return true if the node is either (1) a parameter, or (2) a placeholder."""
        if (source_info := self.get_relative_placement_source(node)) is not None:
            # If node has relative placement constraints, then check the source.
            return self.is_memory_planned(source_info.source)
        # Check if any node is a param.
        if node.op == "get_attr":
            return False
        if node.op == "placeholder" and node.meta.get("spec").const:  # type: ignore[union-attr]
            # Parameters / constants are not memory planned.
            return False
        if node.op == "placeholder" and not (self.alloc_graph_input):
            # For placeholder input tensors, we need alloc_graph_input = True.
            return False
        for user in node.users:
            if user.op == "output" and not (self.alloc_graph_output):
                # For placeholder output tensors, we need alloc_graph_output = True.
                return False
        return True

    def contains_unoptimizable_placeholder_or_param(
        self,
        nodes: Iterable[torch.fx.Node],
    ) -> bool:
        """Return true if any node in the incoming nodes list is not memory planned."""
        return any(not self.is_memory_planned(node) for node in nodes)

    # Return true if the node is (1) among the output of the graph, and
    # (2) not allocated memory by the mem planning algorithm.
    def is_unallocated_output(self, graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
        # If we have allocated memory for all the outputs, return False
        if self.alloc_graph_output:
            return False
        # Get the output node, and check if the incoming node is in its args
        return is_node_in_flattened_output(graph, node)

    # A recursive call that, given a spec, checks if it is the key in
    # unresolved_loc_constraints. If so, then it derives the mem_offset and
    # mem_id for all the specs that are its values.
    def resolve_relative_loc_constraints(self, spec: TensorSpec) -> None:
        spec_id = id(spec)
        if spec_id not in self.unresolved_loc_constraints:
            return

        assert isinstance(spec, TensorSpec)
        for dependent_node in self.unresolved_loc_constraints[spec_id]:
            source_info = self.get_relative_placement_source(dependent_node)
            assert source_info is not None
            dependent_spec = cast(TensorSpec, dependent_node.meta.get("spec"))
            dependent_spec.mem_id = spec.mem_id  # type: ignore[assignment]
            dependent_spec.mem_offset = spec.mem_offset + source_info.offset  # type: ignore[operator,assignment]
            # Recursively resolve any relative constraints on this arg_spec
            self.resolve_relative_loc_constraints(dependent_spec)

        # We have recursively resolved the constraints for spec, so remove
        # the key.
        self.unresolved_loc_constraints.pop(spec_id)

    def update_children_nodes(self, node: torch.fx.Node, update_lifetime: bool) -> None:
        """Update the source node for child nodes of `node`.
        Converts source -> node -> child to source -> child.
        """
        children_nodes = self.unresolved_loc_constraints[id(node.meta.get("spec"))]
        self.unresolved_loc_constraints.pop(id(node.meta.get("spec")))

        source_info = self.get_relative_placement_source(node)
        assert source_info is not None

        for child_node in children_nodes:
            child_info = self._relative_placement_constraint.pop(
                id(child_node.meta.get("spec"))
            )
            self.add_relative_placement_constraint(
                source_info.source,
                child_node,
                offset=source_info.offset + child_info.offset,
                update_lifetime=update_lifetime,
            )

    def add_relative_placement_constraint(
        self,
        source: torch.fx.Node,
        dependent: torch.fx.Node,
        offset: int = 0,
        update_lifetime: bool = True,
    ) -> None:
        """Add a location constraint between source and dependent nodes.

        The imperative is that both the nodes are tensors. The constraint generated
        will imply that shadow tensor is at a distance `offset` from
        representative tensor.
        """
        logging.debug(f"Adding constraint {dependent} = {source} + {offset=}")

        # Assert that both source and dependent node are tensors.
        if (info := self.get_relative_placement_source(source)) is not None:
            source = info.source
            offset += info.offset

        if (info := self.get_relative_placement_source(dependent)) is not None:
            # Dependent node can only be an alias (same size, offset = 0).
            assert self.is_alias_of(
                info.source, dependent
            ), f"Multiple constraints for allocation of {dependent}. Previous constraint: {info} new constraint: {source=} {offset=}"
            dependent = info.source

        # Add the dependent spec to skip list. Its memory offset will be computed
        # after the output tensor is allocated space.
        source_info = RelativePlacementConstraint(source=source, offset=offset)
        self.set_relative_placement_constraint(dependent, source_info)

        # If update_lifetime is True, take a union of the lifetime of representaitve
        # and dependent tensors; this will become the new lifetime of source tensor.
        dependent_spec = dependent.meta.get("spec")
        if update_lifetime:
            source_spec = source.meta.get("spec")
            source.meta.get("spec").lifetime = [  # type: ignore[union-attr]
                min(source_spec.lifetime[0], dependent_spec.lifetime[0]),  # type: ignore[union-attr]
                max(source_spec.lifetime[1], dependent_spec.lifetime[1]),  # type: ignore[union-attr]
            ]

        self.update_children_nodes(dependent, update_lifetime)

        abs_constraint = self.get_absolute_placement_constraint(dependent_spec)  # type: ignore[arg-type]
        if abs_constraint is None:
            return

        # Dependent node has an absolute placement constraint.
        # If the offset is not 0, then we cannot add a relative placement constraint.
        if not self.is_alias_of(dependent, source):
            raise RuntimeError(
                f"Cannot add relative placement constraint for {dependent} with non-zero offset {offset} when it has an absolute placement constraint {abs_constraint}"
            )

        # Add the absolute placement constraint to the source node.
        self._absolute_placement_constraints.pop(id(dependent_spec))
        self.add_absolute_placement_constraint(
            source, abs_constraint.pinned_memory_id, abs_constraint.offset
        )

    def add_absolute_placement_constraint(
        self, node: torch.fx.Node, pinned_memory_id: int, offset: Optional[int] = None
    ) -> None:
        """Add a memory pinning constraint for `node` to `mem_id`."""
        logging.debug(
            f"Adding memory pinning constraint {node=} = {pinned_memory_id=} at {offset=}"
        )
        source_node: torch.fx.Node = node
        if (info := self.get_relative_placement_source(node)) is not None:
            assert self.is_alias_of(info.source, node)
            logging.debug(
                f"Setting {node} to {info.source} + {offset=}. Pinned to {pinned_memory_id=}"
            )
            source_node = info.source
        self._absolute_placement_constraints[id(source_node.meta.get("spec"))] = (
            AbsolutePlacementConstraint(
                pinned_memory_id=pinned_memory_id, offset=offset
            )
        )

    def get_absolute_placement_constraint(
        self, spec: TensorSpec
    ) -> Optional[AbsolutePlacementConstraint]:
        """Return true if `node` has an absolute placement constraint."""
        return self._absolute_placement_constraints.get(id(spec), None)


def get_relative_offsets_of_cat_tensors(
    cat_tensors: Sequence[torch.fx.Node],
) -> list[int]:
    """
    Return the relative offsets of all tensors being concatenated, starting at
    0. We do not use allocated_memory from TensorSpec to compute the relative
    offsets, since allocated_memory is adjusted to be a multiple of 16. Instead,
    we compute the actual size of the tensor in bytes.
    """
    relative_offsets = [0]
    for arg in cat_tensors:
        arg_spec = arg.meta.get("spec")
        assert isinstance(arg_spec, TensorSpec)
        # Update the relative_offset by the total memory of arg tensor.
        # Do not use allocated_memory, since it is adjusted to be a
        # multiple of 16.
        relative_offset = relative_offsets[-1] + num_bytes_from_shape_and_dtype(
            torch.Size(arg_spec.shape), arg_spec.scalar_type
        )
        relative_offsets.append(relative_offset)

    return relative_offsets


def get_relative_offset_of_slice(slice_node: torch.fx.Node) -> int:
    """
    Return the relative offset of the output of a slice node from the input. We
    do not use allocated_memory from TensorSpec to compute the relative offsets,
    since allocated_memory is adjusted to be a multiple of 16. Instead, we
    compute the actual size of the tensor in bytes.
    """
    # get the slice input shape
    slice_input = slice_node.args[0]
    assert isinstance(slice_input, torch.fx.Node)
    input_spec = slice_input.meta.get("spec")
    tensor_shape = list(input_spec.shape)  # type: ignore[union-attr]
    assert tensor_shape
    # get the slice dimension
    dim = 0 if len(slice_node.args) == 1 else cast(int, slice_node.args[1])

    # Compute the start. Also account for negative start values, which
    # can be converted to non-negative values by adding the tensor size
    # along the slice dim.
    start = (
        0
        if len(slice_node.args) < 3
        else (
            start_param
            if (start_param := cast(int, slice_node.args[2])) >= 0
            else start_param + tensor_shape[dim]
        )
    )

    # Compute the offset of the output tensor in input tensor.
    # Remove the leading non-unit dimension from tensor_shape, and then
    # compute the tensor size in bytes.
    tensor_shape[dim] = 1

    nbytes = num_bytes_from_shape_and_dtype(
        torch.Size(tensor_shape), input_spec.scalar_type  # type: ignore[union-attr]
    )
    offset = start * nbytes
    return offset


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class GenerateCatNopConstraints(PassBase):
    """
    For cat op where the concatenation is along the outermost dimension, generate
    contiguity constraints for all the input tensors, and mark the cat nop.
    """

    def __init__(self, constraint: MemConstraints) -> None:
        self.constraint = constraint

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:  # type: ignore[return]
        self.compute_cat_contiguity_constraints(graph_module)

    def is_slice_view(self, node: torch.fx.Node) -> bool:
        """Return if `node` has constraints and is not an alias of another node."""
        if (
            source_info := self.constraint.get_relative_placement_source(node)
        ) is not None:
            return not self.constraint.is_alias_of(source_info.source, node)
        return False

    # Return true if the cat node performs concatenation along outermost dimension
    def is_cat_along_outermost_dim(
        self, graph_module: torch.fx.GraphModule, cat_node: torch.fx.Node
    ) -> bool:
        assert len(cat_node.args) > 0
        cat_tensors = cat_node.args[0]
        if not isinstance(cat_tensors, Sequence) or not all(
            isinstance(t, torch.fx.Node) for t in cat_tensors
        ):
            raise ValueError("cat_tensors must be a sequence of torch.fx.Node objects.")

        if len(cat_node.args) > 1:
            cat_dim = cat_node.args[1]
        else:
            cat_dim = cat_node.kwargs.get("dim", 0)
        if not isinstance(cat_dim, int):
            raise ValueError("cat_dim must be an integer.")

        # If the cat op has default dim, then the concat dim is 0
        if len(cat_tensors) == 1 or cat_dim == 0:
            return True

        # Make sure all dimes before cat_dim are 1.
        for tensor in cat_tensors:
            if not isinstance(tensor, torch.fx.Node):
                continue
            shape = get_shape(graph_module, tensor)
            if shape is None or not all(dim == 1 for dim in shape[0:cat_dim]):
                return False
        return True

    # If A = cat(B, C), and the concatenation is along the outermost dimension, then
    # we can optimize away this cat operation if (1) B and C are placed contiguously,
    # and (2) the absolute memory location of tensor A is the same as B. This function
    # returns True if node is such a cat op.
    def is_removable_cat_op(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> bool:
        # Only track aten.cat.out ops
        if node.op != "call_function" or node.target != torch.ops.aten.cat.out:
            return False

        # If the cat node is part of the graph output, and unallocated by mem
        # planning, we cannot eliminate it.
        if self.constraint.is_unallocated_output(graph_module.graph, node):
            return False

        # A cat op is redundant only if the cat is along the outermost dimension.
        if not self.is_cat_along_outermost_dim(graph_module, node):
            return False

        cat_tensors = cast(Sequence[torch.fx.Node], node.args[0])

        # If any arg of the node is an unoptimizable placeholder or param, bail.
        # This is because we cannot control/change their memory location.
        # However, we could consider memory prefetching to the target memory in
        # the future.
        if self.constraint.contains_unoptimizable_placeholder_or_param(cat_tensors):
            return False

        # If any of the tensors to be concatenated is slice_nop or cat_nop, bail
        if any(self.is_slice_view(arg) for arg in cat_tensors):
            return False

        # Many ops in HiFi require the input to be aligned to 8-byte boundary.
        # If the cat is not the graph's output, then ensure that the relative
        # offset of any concatenated non-placeholder tensor is a multiple of
        # 8 bytes,
        if not is_node_in_flattened_output(graph_module.graph, node):
            expected_alignment = 8
            relative_offsets = get_relative_offsets_of_cat_tensors(cat_tensors)
            for idx, arg in enumerate(cat_tensors):
                if not (arg.op == "placeholder") and (
                    relative_offsets[idx] & (expected_alignment - 1) != 0
                ):
                    return False

        return True

    # Currently the contiguity constraints are generated by cat operator.
    def compute_cat_contiguity_constraints(
        self, graph_module: torch.fx.GraphModule
    ) -> None:
        for node in graph_module.graph.nodes:
            # Only compute relative constraints if the cat node can be replaced with
            # its nop version
            if not self.is_removable_cat_op(graph_module, node):
                continue

            # Finally replace the target of this cat op with the nop target
            node.target = torch.ops.aten._cat_nop.out

            # Get the list of tensors to be concatenated. It should be non-empty.
            cat_tensors = node.args[0]

            # Assert that the output of the cat is a tensor. We will now generate
            # relative location constraints for each concatenated tensor wrt. the
            # output tensor.
            node_spec = node.meta.get("spec")
            assert isinstance(node_spec, TensorSpec)

            # Get the relative offsets for each tensor to be concatenated.
            relative_offsets = get_relative_offsets_of_cat_tensors(cat_tensors)
            for arg, offset in zip(cat_tensors, relative_offsets):
                self.constraint.add_relative_placement_constraint(
                    node, arg, offset=offset
                )

            # Update the lifetimes of the args to that of the output tensor, so
            # that they don't get overwritten
            lifetime = node_spec.lifetime
            for arg in cat_tensors:
                arg_spec = arg.meta.get("spec")
                arg_spec.lifetime = lifetime

        graph_module.recompile()


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class GenerateMemoryViewConstraints(PassBase):
    """
    For memory.view ops, where input and output use the same underlying memory,
    generate input output colocation constraints.
    """

    def __init__(self, constraint: MemConstraints) -> None:
        self.constraint = constraint

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:  # type: ignore[return]
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target != memory.view:
                continue
            self.constraint.add_relative_placement_constraint(node.args[0], node)


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class GenerateSliceAndSelectNopConstraints(PassBase):
    """
    For slice ops, where the slice is along the outermost dimension, generate
    an offset-based location constraint. Also optimize select ops, since select
    op is nothing but slice op along the selected dimension at the given index.
    """

    def __init__(self, constraint: MemConstraints) -> None:
        self.constraint = constraint

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:  # type: ignore[return]
        self.compute_slice_and_select_loc_constraints(graph_module)

    # Return True if the slice or select op can be replaced by a nop after
    # generating some relative location constraints.
    def removable_slice_or_select_op(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> bool:
        # Only track aten.slice_copy and aten.select_copy ops
        if node.op != "call_function" or node.target not in {
            torch.ops.aten.slice_copy.Tensor_out,
            torch.ops.aten.select_copy.int_out,
        }:
            return False

        # If the slice/select node is part of the graph output, and unallocated
        # by mem planning, we cannot eliminate it.
        if self.constraint.is_unallocated_output(graph_module.graph, node):
            return False

        # If the sliced tensor is an unoptimizable placeholder or param, bail.
        input_node = node.args[0]
        assert isinstance(input_node, torch.fx.Node)
        if not self.constraint.is_memory_planned(input_node):
            return False

        # A slice/select op is redundant only if (a) either the slicing/select
        # is along the outermost dimension, or (b) all dimensions previous to
        # slicing/select dimension are 0 or 1.
        node_spec = node.meta.get("spec")
        tensor_shape = list(node_spec.shape)  # type: ignore[union-attr]
        dim = 0 if len(node.args) == 1 else node.args[1]
        if dim and not set(tensor_shape[0:dim]).issubset({0, 1}):  # type: ignore[misc]
            return False

        # The slice step should be 1 for contiguity.
        step = 1 if len(node.args) < 5 else node.args[4]
        if step != 1:
            return False

        # Many ops in HiFi require the input to be aligned to 8-byte boundary.
        # If the slice op is not the graph's output, the ensure that the relative
        # offset of the output tensor of slice is a multiple of 8 bytes,
        if not is_node_in_flattened_output(graph_module.graph, node):
            expected_alignment = 8
            output_offset = get_relative_offset_of_slice(node)
            if output_offset & (expected_alignment - 1) != 0:
                return False

        return True

    # For slice ops where the input is not a placeholder, the slicing is along
    # the outermost dimension with step=1, generate a colocation constraint between
    # the input and output tensor.
    def compute_slice_and_select_loc_constraints(
        self, graph_module: torch.fx.GraphModule
    ) -> None:
        for node in graph_module.graph.nodes:
            # Only compute relative constraints if the slice node can be
            # replaced with its nop version
            if not self.removable_slice_or_select_op(graph_module, node):
                continue

            # Replace the target of this slice op with the nop target
            node.target = {
                torch.ops.aten.slice_copy.Tensor_out: torch.ops.aten._slice_copy_nop.Tensor_out,
                torch.ops.aten.select_copy.int_out: torch.ops.aten._select_copy_nop.int_out,
            }[node.target]

            # Compute the offset of the output tensor in input tensor.
            offset = get_relative_offset_of_slice(node)

            # And now generate location constraint between input and output
            # tensors of slice node
            arg = node.args[0]
            self.constraint.add_relative_placement_constraint(
                arg,
                node,
                offset=offset,
                update_lifetime=True,
            )

        graph_module.recompile()


ConstraintsGenPass: TypeAlias = Callable[
    [MemConstraints],
    Callable[[torch.fx.GraphModule], Optional[PassResult]],
]


# The class to generate all the constraints that will be passed on to the memory
# planning algorithm.
class GenerateMemConstraints:
    def __init__(
        self,
        mem_constraints: MemConstraints,
        additional_constraint_gen_passes: Sequence[ConstraintsGenPass] | None = None,
    ) -> None:
        self.mem_constraints: MemConstraints = mem_constraints
        self.additional_constraint_gen_passes: Sequence[ConstraintsGenPass] = (
            additional_constraint_gen_passes or []
        )

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        constraint_gen_passes: Sequence[ConstraintsGenPass] = cast(
            list[ConstraintsGenPass],
            [
                GenerateMemoryViewConstraints,
                GenerateSliceAndSelectNopConstraints,
                GenerateCatNopConstraints,
            ],
        ) + list(self.additional_constraint_gen_passes)
        # Create a filter using the opt level in mem_constraints, and filter
        # the relevant passes.
        pass_filter = create_cadence_pass_filter(self.mem_constraints.opt_level)
        filtered_passes = [
            mcg_pass(self.mem_constraints)
            for mcg_pass in cast(
                list[ConstraintsGenPass],
                # pyre-ignore[6]: Incompatible parameter type.
                list(filter(pass_filter, constraint_gen_passes)),  # type: ignore[arg-type]
            )
        ]
        # Now run the pass manager on the filtered passes
        return PassManager(passes=filtered_passes)(graph_module)
