# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import logging
import math
import typing
from collections import defaultdict
from dataclasses import dataclass
from typing import cast, DefaultDict, Iterable, Optional, Sequence

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
class SourceInfo:
    """Information of source node and offset used for views."""

    source: torch.fx.Node
    offset: int = 0


class MemConstraints:
    """
    This class contains all the tensor placement constraints that we create
    during memory planning.
    Any tensor whose placement is derived off another tensor via a constraint
    is not included in memory planning, and is marked as skipped.
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
        self._source_node: dict[int, SourceInfo] = {}

        # A map from `id(TensorSpec)` to a set of mem_ids that cannot be used for
        # allocating the tensor.
        self._mem_id_blocklist: dict[int, set[int]] = {}

    def get_source_info(self, node: torch.fx.Node) -> Optional[SourceInfo]:
        spec = node.meta.get("spec")
        spec_id = id(spec)
        if spec_id not in self._source_node:
            return None
        return self._source_node[spec_id]

    def set_source_info(
        self, dependent: torch.fx.Node, source_info: SourceInfo
    ) -> None:
        dependent_spec = dependent.meta.get("spec")
        spec_id = id(dependent_spec)
        self._source_node[spec_id] = source_info
        if self.is_memory_planned(source_info.source):
            # Only add dependent nodes if source node needs memory planning.
            self.unresolved_loc_constraints[
                id(source_info.source.meta.get("spec"))
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
        if node_source_info := self.get_source_info(node):
            node_spec = node.meta.get("spec")
            node_source_spec = node_source_info.source.meta.get("spec")
            return (
                node_source_info.offset == 0
                and math.prod(node_source_spec.shape) == math.prod(node_spec.shape)
                and node_source_spec.dtype == node_spec.dtype
                and self.is_alias_of(node_source_info.source, other_node)
            )

        if self.get_source_info(other_node) is not None:
            return self.is_alias_of(other_node, node)

        return node == other_node

    # Return true if the unresolved_loc_constraints is empty
    def relative_loc_constraints_exist(self) -> bool:
        return len(self.unresolved_loc_constraints) != 0

    # Return true if the spec is marked as skipped
    def skipped_spec(self, spec: TensorSpec) -> bool:
        return id(spec) in self._source_node

    def is_memory_planned(
        self,
        node: torch.fx.Node,
    ) -> bool:
        """Return true if the node is either (1) a parameter, or (2) a placeholder."""
        if (source_info := self.get_source_info(node)) is not None:
            # If node has relative placement constraints, then check the source.
            return self.is_memory_planned(source_info.source)
        # Check if any node is a param.
        if node.op == "get_attr":
            return False
        if node.op == "placeholder" and node.meta.get("spec").const:
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
            source_info = self.get_source_info(dependent_node)
            assert source_info is not None
            dependent_spec = cast(TensorSpec, dependent_node.meta.get("spec"))
            dependent_spec.mem_id = spec.mem_id
            dependent_spec.mem_offset = spec.mem_offset + source_info.offset
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

        source_info = self.get_source_info(node)
        assert source_info is not None

        for child_node in children_nodes:
            child_info = self._source_node.pop(id(child_node.meta.get("spec")))
            self.generate_location_constraint(
                source_info.source,
                child_node,
                offset=source_info.offset + child_info.offset,
                update_lifetime=update_lifetime,
            )

    def generate_location_constraint(
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
        if (info := self.get_source_info(source)) is not None:
            return self.generate_location_constraint(
                info.source, dependent, offset + info.offset, update_lifetime
            )

        if (info := self.get_source_info(dependent)) is not None:
            # Dependent node can only be an alias (same size, offset = 0).
            assert self.is_alias_of(
                info.source, dependent
            ), f"Multiple constraints for allocation of {dependent}. Previous constraint: {info} new constraint: {source=} {offset=}"
            return self.generate_location_constraint(
                source, info.source, offset, update_lifetime=update_lifetime
            )

        # Add the dependent spec to skip list. Its memory offset will be computed
        # after the output tensor is allocated space.
        source_info = SourceInfo(source=source, offset=offset)
        self.set_source_info(dependent, source_info)

        # If update_lifetime is True, take a union of the lifetime of representaitve
        # and dependent tensors; this will become the new lifetime of source tensor.
        if update_lifetime:
            dependent_spec = dependent.meta.get("spec")
            source_spec = source.meta.get("spec")
            source.meta.get("spec").lifetime = [
                min(source_spec.lifetime[0], dependent_spec.lifetime[0]),
                max(source_spec.lifetime[1], dependent_spec.lifetime[1]),
            ]

        self.update_children_nodes(dependent, update_lifetime)


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
    tensor_shape = list(input_spec.shape)
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
        torch.Size(tensor_shape), input_spec.scalar_type
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

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
        self.compute_cat_contiguity_constraints(graph_module)

    def is_slice_view(self, node: torch.fx.Node) -> bool:
        """Return if `node` has constraints and is not an alias of another node."""
        if (source_info := self.constraint.get_source_info(node)) is not None:
            return not self.constraint.is_alias_of(source_info.source, node)
        return False

    # Return true if the cat node performs concatenation along outermost dimension
    def is_cat_along_outermost_dim(
        self, graph_module: torch.fx.GraphModule, cat_node: torch.fx.Node
    ) -> bool:
        # If the cat op has default dim, then the concat dim is 0
        if len(cat_node.args) == 1 or cat_node.args[1] == 0:
            return True
        # Get the concatenation dimension and concatenated tensors
        (cat_tensors, cat_dim) = cast(
            tuple[Sequence[torch.fx.Node], int], cat_node.args
        )
        for tensor in cat_tensors:
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
    def compute_cat_contiguity_constraints(self, graph_module: torch.fx.GraphModule):
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
                self.constraint.generate_location_constraint(node, arg, offset=offset)

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

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target != memory.view:
                continue
            self.constraint.generate_location_constraint(node.args[0], node)


@register_cadence_pass(CadencePassAttribute(opt_level=2))
class GenerateSliceAndSelectNopConstraints(PassBase):
    """
    For slice ops, where the slice is along the outermost dimension, generate
    an offset-based location constraint. Also optimize select ops, since select
    op is nothing but slice op along the selected dimension at the given index.
    """

    def __init__(self, constraint: MemConstraints) -> None:
        self.constraint = constraint

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
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
        tensor_shape = list(node_spec.shape)
        dim = 0 if len(node.args) == 1 else node.args[1]
        if dim and not set(tensor_shape[0:dim]).issubset({0, 1}):
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
    ):
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
            self.constraint.generate_location_constraint(
                arg,
                node,
                offset=offset,
                update_lifetime=True,
            )

        graph_module.recompile()


# The class to generate all the constraints that will be passed on to the memory
# planning algorithm.
class GenerateMemConstraints:
    def __init__(
        self,
        mem_constraints: MemConstraints,
        additional_constraint_gen_passes: list | None = None,
    ) -> None:
        self.mem_constraints = mem_constraints
        self.additional_constraint_gen_passes = additional_constraint_gen_passes or []

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        constraint_gen_passes: list = [
            GenerateMemoryViewConstraints,
            GenerateSliceAndSelectNopConstraints,
            GenerateCatNopConstraints,
        ] + self.additional_constraint_gen_passes
        # Create a filter using the opt level in mem_constraints, and filter
        # the relevant passes.
        pass_filter = create_cadence_pass_filter(self.mem_constraints.opt_level)
        filtered_passes = [
            mcg_pass(self.mem_constraints)
            for mcg_pass in cast(
                list[
                    typing.Callable[
                        [MemConstraints],
                        typing.Callable[[torch.fx.GraphModule], Optional[PassResult]],
                    ]
                ],
                list(filter(pass_filter, constraint_gen_passes)),
            )
        ]
        # Now run the pass manager on the filtered passes
        return PassManager(passes=filtered_passes)(graph_module)
