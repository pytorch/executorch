# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import itertools
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from executorch.exir import memory
from executorch.exir.control_flow import while_loop as exir_while
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.error import internal_assert, InternalError
from executorch.exir.operator.convert import is_inplace_variant, is_out_variant
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec

from torch import fx
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
)
from torch.fx import Node
from torch.utils._pytree import tree_flatten

REGISTERED_ALGOS: Dict[str, Callable[..., List[int]]] = {}


class Verifier:
    """
    Verify if the outcome of a memory planning algorithm makes sense.
    E.g., make sure tensors having overlapping lifetime does not have overlapping
    storage/buffer.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        alloc_graph_input: bool,
        alloc_graph_output: bool,
        alloc_mutable_buffers: bool,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> None:
        self.graph_module = graph_module
        self.graph_signature = graph_signature
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.alloc_mutable_buffers = alloc_mutable_buffers

    @classmethod
    def mem_obj_id_match(
        cls, lhs_spec: TensorSpec, rhs_spec: TensorSpec, accept_both_none: bool = True
    ) -> bool:
        """
        Given two `TensorSpec`, return if their `mem_obj_id` are the same. Note that if
        both are None, this function will return True if `accept_both_none` is True and
        False otherwise.
        """
        if lhs_spec.mem_id != rhs_spec.mem_id:
            return False

        # both are None
        if lhs_spec.mem_obj_id is None and rhs_spec.mem_obj_id is None:
            return accept_both_none

        return lhs_spec.mem_obj_id == rhs_spec.mem_obj_id

    @classmethod
    def has_overlap(cls, lhs_ivl: List[int], rhs_ivl: List[int]) -> bool:
        r"""
        The passed in intervals are inclusive in both sides. Return if they have
        overlapping.
        """
        # empty interval
        if lhs_ivl[0] > lhs_ivl[1] or rhs_ivl[0] > rhs_ivl[1]:
            return False

        return (lhs_ivl[0] >= rhs_ivl[0] and lhs_ivl[0] <= rhs_ivl[1]) or (
            rhs_ivl[0] >= lhs_ivl[0] and rhs_ivl[0] <= lhs_ivl[1]
        )

    @classmethod
    def lifetime_overlap(cls, lhs_spec: TensorSpec, rhs_spec: TensorSpec) -> bool:
        lhs_lifetime = lhs_spec.lifetime
        rhs_lifetime = rhs_spec.lifetime
        internal_assert(
            lhs_lifetime[0] is not None and lhs_lifetime[1] is not None,
            f"{lhs_spec} should have valid start and end",
        )
        internal_assert(
            rhs_lifetime[0] is not None and rhs_lifetime[1] is not None,
            f"{rhs_spec} should have valid start and end",
        )
        return cls.has_overlap(lhs_lifetime, rhs_lifetime)

    @classmethod
    def storage_overlap(cls, lhs_spec: TensorSpec, rhs_spec: TensorSpec) -> bool:
        intervals = []
        if lhs_spec.mem_id != rhs_spec.mem_id:
            return False
        for spec in [lhs_spec, rhs_spec]:
            internal_assert(
                spec.allocated_memory >= 0,
                f"{spec} should have non-zero allocated memory",
            )
            internal_assert(
                isinstance(spec.mem_offset, int) and spec.mem_offset >= 0,
                f"{spec} should have specified memory offset",
            )
            intervals.append(
                [spec.mem_offset, spec.mem_offset + spec.allocated_memory - 1]
            )
        has_overlap = cls.has_overlap(*intervals)

        return has_overlap

    @classmethod
    def _debug_message_from_specs(
        cls, lhs_spec: TensorSpec, rhs_spec: TensorSpec
    ) -> str:
        message = (
            f"lhs life time: {lhs_spec.lifetime}, rhs lifetime: {rhs_spec.lifetime} "
        )
        message += f"lhs: mem_id {lhs_spec.mem_id} storage: {lhs_spec.mem_offset}, {lhs_spec.allocated_memory} "
        message += f"rhs: mem_id {rhs_spec.mem_id} storage: {rhs_spec.mem_offset}, {rhs_spec.allocated_memory}"
        return message

    def verify_storage_reuse(
        self, allow_lifetime_and_storage_overlap: bool = False
    ) -> int:
        """
        'allow_lifetime_and_storage_overlap' allows tensors to overlap in both
        lifetime and storage. If is it False, and two tensors have both overlapping
        lifetime and storage, throw an exception.
        Returns:
            Number of pairs of tenors that have overlapping storage.
        """
        num_reuse_pairs = 0

        # unique tensors specs
        all_specs = list(
            collect_specs_from_nodes(
                self.graph_module.graph.nodes,
                self.graph_signature,
                ignore_const=True,
                ignore_graph_input=not self.alloc_graph_input,
                ignore_graph_output=not self.alloc_graph_output,
                ignore_mutable_buffers=not self.alloc_mutable_buffers,
                do_assertion=False,
                ignore_out_var_node=False,
                dedup=True,
            )
        )

        for lhs_spec_idx, lhs_spec in enumerate(all_specs):
            for rhs_spec in all_specs[lhs_spec_idx + 1 :]:
                # Check that both specs are consistent about whether mem_obj_id is defined
                if (lhs_spec.mem_obj_id is None) != (rhs_spec.mem_obj_id is None):
                    raise InternalError(
                        "Specs do not agree on whether mem_obj_id is defined."
                    )

                has_storage_overlap = Verifier.storage_overlap(lhs_spec, rhs_spec)
                if not has_storage_overlap:
                    continue

                if not allow_lifetime_and_storage_overlap and self.lifetime_overlap(
                    lhs_spec, rhs_spec
                ):
                    raise InternalError(
                        f"Unexpected storage overlap: {Verifier._debug_message_from_specs(lhs_spec, rhs_spec)}"
                    )

                # Check that each mem_obj_id is consistent with whether the tensors have
                # storage overlap
                if not Verifier.mem_obj_id_match(lhs_spec, rhs_spec):
                    raise InternalError(
                        f"Unexpected mem_obj_id mismatch: lhs {lhs_spec}, rhs {rhs_spec}"
                    )

                num_reuse_pairs += 1

        return num_reuse_pairs

    def verify_graph_input_output(self) -> None:
        r"""
        alloc_graph_input / alloc_graph_output indicas if memory for graph
        input/output is allocated by the compiler. If not, the runtime will
        set them using buffers provided by users.
        """
        graph_module = self.graph_module
        # There is one tricky case here. If the graph input and graph output
        # tensors have overlap, but alloc_graph_input != alloc_graph_output,
        # then the overlapped tensor will cause assertion failure below.
        # The current behavior is if either alloc_graph_input or alloc_graph_output
        # is false, those overlapped tensor will not have memory allocated.
        #
        # Ignore the check in this case for now.
        overlap = get_graph_input_tensors(
            graph_module.graph.nodes, self.graph_signature
        ) & get_graph_output_tensors(graph_module.graph.nodes)
        if overlap and (self.alloc_graph_input != self.alloc_graph_output):
            logging.debug(
                "Having overlapping graph input/output tensors while the allocation decision for graph input/output mismatch."
            )
            return

        graph_input_allocated = None
        graph_output_allocated = None

        has_dynamic_unbound_input = False
        has_dynamic_unbound_output = False

        check_list = {"placeholder", "output"} & {
            node.op for node in graph_module.graph.nodes
        }
        assert "output" in check_list, f"graph module has no output: {graph_module}"

        for nd in graph_module.graph.nodes:
            if nd.op in check_list:
                if not (specs := get_node_tensor_specs(nd)):
                    continue
                if _is_mutable_buffer(nd, self.graph_signature):
                    continue
                assert len(specs) > 0, "Expect tensor specs"
                specs = list(filter(lambda spec: not spec.const, specs))
                if len(specs) == 0:
                    continue
                allocated = any(
                    spec is None or spec.mem_offset is not None for spec in specs
                )
                has_dynamic_unbound_tensor = any(
                    spec is None
                    or spec.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND
                    for spec in specs
                )
                assert (
                    all(spec is None or spec.mem_offset is not None for spec in specs)
                    == allocated
                ), "Either all or non of the tensors should be allocated memory"
                if nd.op == "placeholder":
                    graph_input_allocated = allocated
                    has_dynamic_unbound_input |= has_dynamic_unbound_tensor
                else:
                    graph_output_allocated = allocated
                    has_dynamic_unbound_output |= has_dynamic_unbound_tensor

        # only check if inputs are allocated if there are user inputs:
        user_inputs_exist = _do_user_inputs_exist(graph_signature=self.graph_signature)

        if "placeholder" in check_list and user_inputs_exist:
            assert graph_input_allocated is not None, "graph_input_allocated not set"
            if not has_dynamic_unbound_input:
                assert (
                    graph_input_allocated == self.alloc_graph_input
                ), f"Misallocate graph input: {graph_input_allocated} v.s. {self.alloc_graph_input}"

        assert graph_output_allocated is not None, "graph_output_allocated not set"
        if not has_dynamic_unbound_output:
            assert (
                graph_output_allocated == self.alloc_graph_output
            ), f"Misallocate graph output {graph_output_allocated} v.s. {self.alloc_graph_output}"


def _is_out_var_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and is_out_variant(node.target._schema.name, node.target._schema.overload_name)
    )


def _is_inplace_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and is_inplace_variant(
            node.target._schema.name, node.target._schema.overload_name
        )
    )


def update_tensor_lifetime(
    node: torch.fx.Node,
    spec: TensorSpec,
    node_idx: int,
    max_node_idx: int,
    gs: Optional[ExportGraphSignature] = None,
) -> None:
    r"""
    Update the lifetime of the tensor to cover node_idx. A tensor's lifetime
    are represented by the index of the first and last node referring
    that tensor in its inputs/outputs.

    Arguments:
        spec: the TensorSpec for the tensor
        node_idx: extend the tensor's lifetime to cover node_idx
    """
    start, end = spec.lifetime
    if node.op == "placeholder":
        start = 0
    else:
        start = node_idx if start is None or start > node_idx else start

    if node.op == "placeholder" and _is_mutable_buffer(node, gs):
        # mutable buffers are never freed
        end = max_node_idx
    else:
        end = node_idx if end is None or end < node_idx else end
    spec.lifetime = [start, end]


# pyre-ignore
def filter_nodes(inputs: Iterable[Any]) -> Iterable[Node]:
    """
    This method need return Node object embedded inside List/Dict as well.
    """
    return [nd for nd in tree_flatten(list(inputs))[0] if isinstance(nd, Node)]


def _is_mutable_buffer(
    node: Node, graph_signature: Optional[ExportGraphSignature] = None
) -> bool:
    """
    Check if the node is mutable buffer according to the provided graph signature.
    """
    # graph signature is None for memory planning passes not called from EdgeProgramManager, these paths are deprecated so mutable buffers are not supported on them.
    if graph_signature is None:
        return False
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                # if the buffer is mutated then record that
                if fqn in graph_signature.buffers_to_mutate.values():
                    return True
    return False


def _do_user_inputs_exist(graph_signature: Optional[ExportGraphSignature]) -> bool:
    if graph_signature is None:
        return False

    user_inputs = list(
        filter(
            lambda input: input.kind == InputKind.USER_INPUT,
            graph_signature.input_specs,
        )
    )

    # Return false if:
    # - there are no inputs.
    # - if user inputs are all prims (as this currently
    #   causes the memory planning verifier to blow up).
    # Otherwise, return true.
    return any(
        not isinstance(input.arg, ConstantArgument)
        or not isinstance(input.arg.value, (int, float, bool, str))
        for input in user_inputs
    )


def get_graph_input_tensors(
    nodes: Iterable[Node], graph_signature: Optional[ExportGraphSignature] = None
) -> Set[TensorSpec]:
    graph_input_tensors = set()
    for node in nodes:
        if node.op == "placeholder" and not _is_mutable_buffer(node, graph_signature):
            for spec in get_node_tensor_specs(node):
                graph_input_tensors.add(spec)

    return graph_input_tensors


def get_graph_output_tensors(nodes: Iterable[Node]) -> Set[TensorSpec]:
    graph_output_tensors = set()
    for node in nodes:
        if node.op == "output":
            for spec in get_node_tensor_specs(node):
                graph_output_tensors.add(spec)

    return graph_output_tensors


def collect_specs_from_nodes(  # noqa: C901
    nodes: Iterable[Node],
    graph_signature: Optional[ExportGraphSignature] = None,
    ignore_graph_input: bool = False,
    ignore_graph_output: bool = False,
    ignore_mutable_buffers: bool = False,
    ignore_const: bool = True,
    ignore_out_var_node: bool = True,
    dedup: bool = True,
    do_assertion: bool = True,
    ignore_dynamic_unbound_tensor: bool = True,
) -> Iterable[TensorSpec]:
    r"""
    Collect specs from the passed in nodes. Do filtering as controlled by
    arguments.
    Arguments:
        ignore_graph_input: ignore graph input tensors from placeholder nodes
        ignore_const: whether to ignore the const
        ignore_out_var_node: whether to ignore out variant node
        dedup: whether do dedup
        do_assertion: whether to assert the filtered nodes belong to a resticted set like alloc, getitem
    """
    unique_spec = set()
    graph_input_tensors: Set[TensorSpec] = (
        get_graph_input_tensors(nodes, graph_signature) if ignore_graph_input else set()
    )
    graph_output_tensors: Set[TensorSpec] = (
        get_graph_output_tensors(nodes) if ignore_graph_output else set()
    )

    for node in nodes:
        # ignore the specs from unrelevant Fx ops for now.
        if node.op in ["get_attr"]:
            continue

        # don't reallocate memory for out-variant op's output tensors,
        # since they are just input tenors.
        if ignore_out_var_node and _is_out_var_node(node):
            continue

        if not (specs := get_node_tensor_specs(node)):
            continue

        if _is_inplace_node(node):
            continue

        if _is_mutable_buffer(node, graph_signature) and ignore_mutable_buffers:
            continue

        if do_assertion:
            internal_assert(
                node.op in ("placeholder", "output")
                or node.target
                in [
                    memory.alloc,
                    memory.view,
                    operator.getitem,
                    torch.ops.higher_order.cond,
                    exir_while,
                    torch.ops.higher_order.map_impl,
                    executorch_call_delegate,
                ],
                f"Unexpected op {node.op}, target {node.target}",
            )
        for spec in specs:
            if spec is None:
                continue
            # Dynamic unbound tensors' memory will be allocated by the runtime.
            # Memory planning should ignore them.
            if (
                ignore_dynamic_unbound_tensor
                and spec.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND
            ):
                continue

            # Note: graph input may be the output of other ops (e.g. the return op)
            # If ignore_graph_input is true, we should ignore those Tensor so
            # we skip planning memory for graph input.
            if ignore_graph_input and spec in graph_input_tensors:
                continue
            if ignore_graph_output and spec in graph_output_tensors:
                continue
            if (
                ignore_const
                and spec.const
                and not node.meta.get("weight_has_gradient", False)
            ):
                continue
            if dedup:
                if spec in unique_spec:
                    continue
                else:
                    unique_spec.add(spec)
            yield spec


def update_all_tensors_lifetime(
    graph_module: torch.fx.GraphModule,
    graph_signature: Optional[ExportGraphSignature] = None,
) -> Set[TensorSpec]:
    r"""
    Set the lifetime for all the tensors encountered in the Fx graph.
    """
    specs = set()
    max_node_idx = len(graph_module.graph.nodes) - 1
    for node_idx, node in enumerate(graph_module.graph.nodes):
        for spec in collect_specs_from_nodes(
            filter_nodes(itertools.chain([node], node.args, node.kwargs.values())),
            graph_signature,
            ignore_graph_input=False,
            ignore_const=False,
            ignore_out_var_node=False,
            dedup=False,
            do_assertion=False,
            ignore_dynamic_unbound_tensor=False,
        ):
            update_tensor_lifetime(node, spec, node_idx, max_node_idx, graph_signature)
            specs.add(spec)
    return specs


@dataclass
class AllocationSpec:
    """
    AllocationSpec is used to represent the allocation of a tensor.
    """

    # The offset of the tensor in the shared object/pool.
    offset: int
    # TensorSpec
    spec: TensorSpec


@dataclass
class SharedObject:
    r"""
    We define the concept of shared object, which represents a segment
    in the memory buffer that can be shared by multiple tensors. In order to
    check if a shared object is available for a tensor, we maintain the
    last_used_index attribute. The shared object will be available for nodes
    with index greater than last_used_index.
    """

    # index of the shared object in the list of shared objects, used as a unique id
    idx: int
    # offset in the memory buffer
    offset: int
    # size of this shared object in bytes
    size: int
    # When the object is first created
    first_used_index: int
    # the object will be available for index (last_used_index + 1)
    last_used_index: int
    # list of allocations belong to this shared object
    allocations: List[AllocationSpec] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SharedObject(idx={self.idx}, offset={self.offset}, size={self.size}, lifetime=[{self.first_used_index, self.last_used_index}])"


@dataclass
class SpecAllocResult:
    """These are the values that a memory plannig algorithm assigns to a spec.
    These are not directly written back into the spec object, but are used to
    track the allocation decisions and assigned back to the spec object in the
    end, based on which algorithm is picked as the best performing one.
    """

    mem_id: int
    mem_obj_id: int
    mem_offset: int


@dataclass
class MemoryAlgoResult:
    """This is the result returned by a memory planning algorithm that is
    invoked by memory_planning_algorithm_suite. It contains the allocation
    decisions of that algorithm for all the specs, and the size of the buffer
    that was used for different memory hierarchies.
    """

    spec_dict: Dict[TensorSpec, SpecAllocResult]
    bufsizes: List[int]


def materialize_buffer(
    shared_objects: List[SharedObject], input_total_size: int = 0
) -> int:
    r"""
    Assign concrete location in the buffer for each SharedObject.offset.

    Assuming all the passed in shared objects belong to the same memory buffer.
    """
    total_size = input_total_size
    for sobj in shared_objects:
        sobj.offset = total_size
        total_size += sobj.size
    return total_size


def _does_not_overlap(sobj: SharedObject, spec: TensorSpec) -> bool:
    r"""
    Check if a shared object and a tensor do not overlap.
    """
    for alloc in sobj.allocations:
        if not (
            spec.lifetime[1] < alloc.spec.lifetime[0]
            or spec.lifetime[0] > alloc.spec.lifetime[1]
        ):
            return False
    return True


def _find_max_overlapping_allocations_offset(
    sobj: SharedObject, spec: TensorSpec
) -> int:
    max_offset = 0
    for alloc in sobj.allocations:
        if (
            spec.lifetime[1] < alloc.spec.lifetime[0]
            or spec.lifetime[0] > alloc.spec.lifetime[1]
        ):
            continue
        max_offset = max(alloc.offset + alloc.spec.allocated_memory, max_offset)
    return max_offset


def pick_shared_obj(
    shared_objects: List[SharedObject],
    spec: TensorSpec,
    allow_overlapping_allocations: bool = True,
) -> SharedObject:
    r"""
    Pick the available shared object to which to assign this spec,
    or create a new one
    Algorithm details
    Previous: Look at every spec in chronological order. Find if previously allocated object
    allows it to fit in. If not, allocate a new object.
    New:
    - Sort all the specs by allocation size
    - Process the specs in order
    - If the spec's size in smaller than previously allocated buckets:
        - Conditions under which previously allocated bucket can be used:
          - Lifetime of the spec does not overlap with lifetime of the bucket.
              - In this case allocate spec to that bucket and expand its lifetime.
              - Spec is allocated at offset = 0 in this bucket.
              - Add this spec to allocated object's list of specs.
          - Lifetime of the spec overlaps with lifetime of the bucket,
            partially or fully (e.g. spec's lifetime subset of bucket's lifetime)
              - If none of the specs in the bucket overlaps with spec's lifetime.
                - Allocate spec to the bucket at offset = 0.
                - Add this spec to the bucket's list of specs.
                - Expand bucket's lifetime accounting for added spec's lifetime.
              - If one or more specs in the bucket overlaps with spec's lifetime.
                - Collect offsets (at which the given overlapping spec is allocated in the bucket).
                  of all the overlapping specs, and find the max offset.
                - Allocate spec to the bucket at offset = max_offset + max_offset_spec_size.
                - Add this spec to the bucket's list of specs.
                - Expand bucket's lifetime accounting for added spec's lifetime.
        - If none of these conditions are met, allocate a new bucket.
            - Add spec to this bucket.
            - Update bucket's lifetime to that of the spec.
    - If the spec's size is larger than previously allocated buckets, allocate a new bucket.
        - Size and lifetime of this bucket is that of the spec

    Proof of correctness:
    - If allocating a new bucket, it is correct.
    - If allocating spec to an existing bucket, whose lifetime does not overlap with any
      of the previously allocated specs' lifetime, then the allocation is correct.
    Proof of correctness by induction when adding spec to an existing bucket:
    - If all previous allocations in the given bucket are correct:
        - Then the new one being added must be correct because when the requested allocation
          overlaps with one or more previous allocations, we find the largest offset among
          all the overlapping allocations, and allocate the new spec at that offset. Hence,
          the allocation at such an offset, will not overlap with any previous allocations.
    Base case: A newly added allocation within a bucket with single allocation is correct:
    because a) it must fit and b) its lifetime must not overlap with object's lifetime.
    This holds true because of the following invariants:
    - Once a bucket is created, it is never resized.
    - All the allocations within a bucket follow this:
      - Span, defined by allocation's offset + size, of two allocations can only overlap,
        if their timelines do not overlap.
    """
    picked = None
    for sobj in shared_objects:
        if _does_not_overlap(sobj, spec):
            assert sobj.size >= spec.allocated_memory, "Allocation specs are not sorted"
            picked = sobj
            sobj.first_used_index = min(sobj.first_used_index, spec.lifetime[0])
            sobj.last_used_index = max(sobj.last_used_index, spec.lifetime[1])
            allocation_spec = AllocationSpec(0, spec)
            picked.allocations.append(allocation_spec)
            break

    if picked is None and allow_overlapping_allocations:
        for sobj in shared_objects:
            max_offset = _find_max_overlapping_allocations_offset(sobj, spec)
            if max_offset > 0:
                if max_offset + spec.allocated_memory <= sobj.size:
                    picked = sobj
                    sobj.first_used_index = min(sobj.first_used_index, spec.lifetime[0])
                    sobj.last_used_index = max(sobj.last_used_index, spec.lifetime[1])
                    allocation_spec = AllocationSpec(max_offset, spec)
                    picked.allocations.append(allocation_spec)
                    break

    if picked is None:
        picked = SharedObject(
            len(shared_objects),
            -1,
            spec.allocated_memory,
            spec.lifetime[0],
            spec.lifetime[1],
        )
        allocation_spec = AllocationSpec(0, spec)
        picked.allocations.append(allocation_spec)
        picked.first_used_index = spec.lifetime[0]
        picked.last_used_index = spec.lifetime[1]
        shared_objects.append(picked)

    return picked


def get_node_tensor_specs(
    node: torch.fx.Node,
) -> Union[List[TensorSpec], Tuple[TensorSpec]]:
    r"""
    Return the list of the tensor specs for the node or empty list if the node
    has no tensor specs.
    """
    # get tensor specs
    if node.target == memory.view:
        base = node.args[0]
        assert isinstance(base, torch.fx.Node)
        specs = base.meta.get("spec")
    else:
        specs = node.meta.get("spec")

    if isinstance(specs, TensorSpec):
        specs = [specs]
    if not isinstance(specs, (list, tuple)):
        return []
    else:
        return [
            spec
            for spec in specs
            if not isinstance(spec, (int, float, bool, str, type(None)))
        ]


# Little bit hacky to check if the graph contains
# XNNPACK delegate
# Why?


def _contains_xnnpack_delegate(graph_module: torch.fx.GraphModule) -> bool:
    for node in graph_module.graph.nodes:
        if node.target == executorch_call_delegate:
            lowered_module = getattr(
                graph_module.graph.owning_module, node.args[0].target
            )
            if "xnnpack" in lowered_module.backend_id.lower():
                return True
    return False


def greedy(
    alignment: int,
    specs: Set[TensorSpec],
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    extra_padding: int = 0,
    *,
    allow_overlapping_allocations: bool = True,
) -> MemoryAlgoResult:
    r"""Greedy algorithm to allocate memory for tensors in the graph.

    Args:
        alignment: Memory alignment requirement
        specs: Set of TensorSpec objects with updated lifetimes
        graph_module: Graph module
        graph_signature: Graph signature
        extra_padding: Additional padding to add to each memory buffer (in bytes)
        allow_overlapping_allocations: If set to true, allows for allocations that overlap
            in their lifetime but are at different offsets in the storage. By default true.
            This flag is added to allow for Vulkan to use MemoryPlanningPass with overlapping
            allocations disabled

    Returns:
        MemoryAlgoResult containing the allocation decisions
    """
    greedy_result = MemoryAlgoResult({}, [])
    spec2obj = {}
    shared_objects = defaultdict(list)

    # For each tensor, pick the available shared object with closest size to
    # the tensor. If there are no available shared object left, create a new
    # one.
    import bisect

    sorted_specs = []
    for spec in specs:
        bisect.insort(sorted_specs, spec, key=lambda x: x.allocated_memory)

    sorted_specs.reverse()

    for spec in sorted_specs:
        # Create an entry for this TensorSpec in the result object that we'll be
        # returning from this algorithm.
        spec_alloc_result = greedy_result.spec_dict.get(spec, SpecAllocResult(0, 0, 0))
        if spec.mem_id is None:
            spec_alloc_result.mem_id = 1
        else:
            spec_alloc_result.mem_id = spec.mem_id
        greedy_result.spec_dict[spec] = spec_alloc_result
        spec.realign(alignment)
        spec2obj[spec] = pick_shared_obj(
            shared_objects[spec_alloc_result.mem_id],
            spec,
            allow_overlapping_allocations,
        )

    if len(shared_objects) == 0:
        # Cannot find any tensor in the graph that needs to be allocated.
        # Return [0, 0] to be consistent with default behavior of naive.
        total_sizes = [0, 0]
    else:
        total_sizes = [0] * (max(shared_objects.keys()) + 1)
        num_specs_processed = 0
        for mem_id in shared_objects:
            input_total_size = 0
            if bufsizes := getattr(graph_module, "input_mem_buffer_sizes", None):
                assert isinstance(bufsizes, list)
                if len(bufsizes) > mem_id:
                    input_total_size = bufsizes[mem_id]
            total_sizes[mem_id] = materialize_buffer(
                shared_objects[mem_id], input_total_size
            )
            total_sizes[mem_id] += extra_padding

            # Since we now know the number of shared objects we need and the size of
            # each shared object, we can assign offset in the memory buffer for each
            # shared object.
            for sobj in shared_objects[mem_id]:
                for alloc in sobj.allocations:
                    spec = alloc.spec
                    # Get the spec_alloc_result for this spec and update it with the
                    # mem_obj_id and mem_offset generated by this algorithm.
                    spec_alloc_result = greedy_result.spec_dict.get(spec, None)
                    assert spec_alloc_result is not None, f"Spec {spec} not found."
                    spec_alloc_result.mem_obj_id = sobj.idx
                    spec_alloc_result.mem_offset = sobj.offset + alloc.offset
                    num_specs_processed += 1
        assert (
            len(spec2obj) == num_specs_processed
        ), f"All specs should be processed but there were {len(spec2obj)} specs and processed {num_specs_processed} specs"

    logging.debug(f"greedy algorithm returns bufsizes: {total_sizes}")
    greedy_result.bufsizes = total_sizes
    return greedy_result


class MemoryPlanningAlgorithmSuite:
    def __init__(
        self,
        algo_list: Optional[List[Callable[..., MemoryAlgoResult]]] = None,
    ) -> None:
        if algo_list is None:
            algo_list = [greedy]
        self.algo_list: List[Callable[..., MemoryAlgoResult]] = algo_list

    def __call__(
        self,
        alignment: int,
        specs: Set[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int,
    ) -> List[int]:
        r"""
        Memory planning algorithm suite that runs a list of memory planning algorithms
        and returns the result of the algorithm that minimizes the total memory usage.

        Args:
            graph_module: The graph module to allocate memory for
            alignment: Memory alignment requirement
            graph_signature: Optional graph signature
            alloc_graph_input: Whether to allocate memory for graph input
            alloc_graph_output: Whether to allocate memory for graph output
            allow_overlapping_allocations: Whether to allow overlapping allocations
            algo_list: List of memory planning algorithms to run
            specs: Optional set of TensorSpec objects with updated lifetimes. If None, they will be
                calculated from the graph_module.

        Returns:
            List of buffer sizes for each memory hierarchy
        """

        mem_algo_results = {}
        for algo in self.algo_list:
            if isinstance(algo, functools.partial):
                name = algo.func.__name__
            else:
                name = getattr(algo, "__name__", None)

            mem_algo_results[name] = algo(
                alignment,
                specs,
                graph_module,
                graph_signature,
                extra_padding,
            )

        # All the algorithms should have the same number of buffers allocated.
        assert (
            len(
                {
                    len(mem_algo_result.bufsizes)
                    for mem_algo_result in mem_algo_results.values()
                }
            )
            == 1
        ), "Different memory planning algorithms should have the same number of buffers allocated."

        # Find the algorithm that minimizes the total memory usage.
        best_algo = min(
            mem_algo_results, key=lambda k: sum(mem_algo_results[k].bufsizes)
        )
        logging.debug(f"Best memory planning algo for this model is {best_algo}")
        bufsizes = mem_algo_results[best_algo].bufsizes

        # Update the mem_id and mem_offset for each spec in the graph module based on the
        # values provided by the best memory planning algorithm.
        for spec in mem_algo_results[best_algo].spec_dict:
            spec_alloc_result = mem_algo_results[best_algo].spec_dict[spec]
            spec.mem_id = spec_alloc_result.mem_id
            spec.mem_offset = spec_alloc_result.mem_offset
            spec.mem_obj_id = spec_alloc_result.mem_obj_id

        return bufsizes


def naive(
    alignment: int,
    specs: Set[TensorSpec],
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    extra_padding: int,
) -> MemoryAlgoResult:
    """Naive algorithm to allocate memory for tensors in the graph.

    This algorithm simply allocates memory for each tensor sequentially without reusing memory.

    Args:
        alignment: Memory alignment requirement
        specs: Set of TensorSpec objects with updated lifetimes
        graph_module: Graph module
        graph_signature: Graph signature
        extra_padding: Additional padding to add to each memory buffer (in bytes)

    Returns:
        MemoryAlgoResult containing the allocation decisions
    """
    naive_result = MemoryAlgoResult({}, [])

    # allocate 'allocated' bytes from buffer with id mem_id.
    # return the starting offset of the allocated buffer.
    def _allocate_buf(bufsizes: List[int], mem_id: int, allocated: int) -> int:
        if mem_id >= len(bufsizes):
            bufsizes.extend([0] * (mem_id - len(bufsizes) + 1))
        ret = bufsizes[mem_id]
        bufsizes[mem_id] += allocated
        return ret

    bufsizes = getattr(graph_module, "input_mem_buffer_sizes", None)
    if bufsizes is None:
        bufsizes = [0, 0]
    bufsizes = cast(List[int], bufsizes)

    for spec in specs:
        spec_alloc_result = naive_result.spec_dict.get(spec, SpecAllocResult(0, 0, 0))
        # assume a single memory layer which has mem_id 1
        if spec.mem_id is None:
            spec_alloc_result.mem_id = 1
        else:
            spec_alloc_result.mem_id = spec.mem_id
        naive_result.spec_dict[spec] = spec_alloc_result

        # allocate spec.allocated_memory bytes in the buffer
        # with the corresponding mem_id
        spec.realign(alignment)
        spec_alloc_result.mem_offset = _allocate_buf(
            bufsizes, spec_alloc_result.mem_id, spec.allocated_memory
        )

    logging.debug(f"naive algorithm returns bufsizes: {bufsizes}")
    naive_result.bufsizes = bufsizes
    return naive_result


def get_cond_nodes(graph_module: torch.fx.GraphModule) -> Iterable[Node]:
    for nd in graph_module.graph.nodes:
        if nd.target is torch.ops.higher_order.cond:
            yield nd


def get_while_nodes(graph_module: torch.fx.GraphModule) -> Iterable[Node]:
    for nd in graph_module.graph.nodes:
        if nd.target is exir_while:
            yield nd


def get_map_nodes(graph_module: torch.fx.GraphModule) -> Iterable[Node]:
    for nd in graph_module.graph.nodes:
        if nd.target is torch.ops.higher_order.map_impl:
            yield nd


def get_return_specs(graph_module: fx.GraphModule) -> Set[TensorSpec]:
    return_specs = set()
    nodes = graph_module.graph.nodes
    if len(nodes) > 0:
        last_node = next(iter(reversed(nodes)))
        for spec in tree_flatten(last_node.meta["spec"])[0]:
            return_specs.add(spec)
    return return_specs


def get_input_specs(graph_module: fx.GraphModule) -> Set[TensorSpec]:
    input_specs = set()
    nodes = graph_module.graph.nodes
    for node in nodes:
        if node.op == "placeholder":
            for spec in tree_flatten(node.meta["spec"])[0]:
                input_specs.add(spec)
    return input_specs


def insert_calls_to_free(
    graph_module: fx.GraphModule, allspecs: Set[TensorSpec]
) -> None:
    """
    Insert calls to free for dynamic unbound tensors that goes out of lifetime.

    Only handle the module itself. Submodule is handles in separate calls of
    this function.

    NOTE: this method will invalidate lifetime recorded in TensorSpec because
    of extra free node added to the graph.
    """
    # Note: we should never free a output tensor
    return_specs = get_return_specs(graph_module)
    # Note: we should never free a input tensor since buffer for input tensor
    # may be passed in from user.
    input_specs = get_input_specs(graph_module)
    idx_to_dead_specs = defaultdict(list)
    for spec in allspecs:
        if (
            spec.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND
            and spec not in return_specs
            and spec not in input_specs
        ):
            idx_to_dead_specs[spec.lifetime[1]].append(spec)

    num_nodes = len(graph_module.graph.nodes)
    # iterate in reverse order so inserted node does not disturbe node
    # numbering.
    for node, node_idx in zip(
        reversed(graph_module.graph.nodes), range(num_nodes - 1, -1, -1)
    ):
        dead_specs = idx_to_dead_specs.get(node_idx, [])
        if not dead_specs:
            continue
        with graph_module.graph.inserting_after(node):
            for spec in dead_specs:
                graph_module.graph.call_function(memory.free, (spec,))
    graph_module.recompile()


def _merge_bufsizes(bufsizes: list[int], new_bufsizes: list[int]) -> list[int]:
    """Combine two buffer size lists."""
    if len(bufsizes) < len(new_bufsizes):
        bufsizes.extend([0] * (len(new_bufsizes) - len(bufsizes)))
    for i in range(len(new_bufsizes)):
        bufsizes[i] = max(bufsizes[i], new_bufsizes[i])
    return bufsizes


def _handle_submodule(
    algo: Callable[..., list[int]],
    parent_graph_module: torch.fx.GraphModule,
    alignment: int,
    submodule_node: torch.fx.Node,
    graph_signature: Optional[ExportGraphSignature] = None,
    alloc_graph_input: bool = False,
) -> list[int]:
    """Apply algo to nodes in a submodule of the graph module."""
    assert submodule_node.op == "get_attr"
    submodule = getattr(parent_graph_module, submodule_node.target)

    logging.debug(f"Planning memory for submodule {submodule_node.name}...")
    bufsizes = apply_algo(
        algo,
        submodule,
        alignment,
        graph_signature,
        alloc_graph_input=alloc_graph_input,
        alloc_graph_output=True,
    )
    submodule.meta.update({"non_const_buffer_sizes": bufsizes})
    logging.debug(f"Buffer sizes for submodule {submodule_node.name}: {bufsizes}")
    return bufsizes


def _apply_algo_to_submodules(
    algo: Callable[..., list[int]],
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: Optional[ExportGraphSignature] = None,
) -> list[int]:
    """Apply algo to map/cond/while nodes in the graph module.

    This method will popuate graph_module.meta["non_const_buffer_sizes"] for
    all submodules and return a bufsizes list that is the maximum size of all
    buffers.
    """

    # Bufsizes for submodules.
    bufsizes: list[int] = []

    def _handle(
        submodule_node: torch.fx.Node,
        alloc_graph_input: bool = False,
    ) -> None:
        current_bufsizes = _handle_submodule(
            algo,
            graph_module,
            alignment,
            submodule_node,
            graph_signature,
            alloc_graph_input=alloc_graph_input,
        )
        nonlocal bufsizes
        _merge_bufsizes(bufsizes, current_bufsizes)

    for cond_node in get_cond_nodes(graph_module):
        _handle(cast(torch.fx.Node, cond_node.args[1]))
        _handle(cast(torch.fx.Node, cond_node.args[2]))

    for while_node in get_while_nodes(graph_module):
        _handle(cast(torch.fx.Node, while_node.args[0]))
        _handle(cast(torch.fx.Node, while_node.args[1]))

    for map_node in get_map_nodes(graph_module):
        _handle(cast(torch.fx.Node, map_node.args[0]), alloc_graph_input=True)

    # TODO: We can handle delegates the same way as map/cond/while.
    # Maybe populate the graph_module.meta["non_const_buffer_sizes"] for delegates.

    return bufsizes


def apply_algo(
    algo: Callable[..., list[int]],
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: Optional[ExportGraphSignature] = None,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
    alloc_mutable_buffers: bool = True,
) -> list[int]:
    """
    Recursively apply algo to graph_module and its submodules for control flow.

    Algo implementation should handle one of two meta entries for submodules:
    1. input_mem_buffer_sizes: List of int offset bytes. Memory allocated by
       `algo` should start at the offset specified by this list;
    OR
    2. non_const_buffer_sizes: List of bufsizes for planned memory in submodule.
       `algo` should reserve the space specified by this list for the lifetime
       of the submodule node (e.g. cond, while, map).

    TODO: Missing optimizations:
    1. To handle maps, we set `alloc_graph_input=True`, which allocates
    appropriate space for mapped arg but ends up allocating extra space for
    `operand` arg. The memory for operands is unused.
    """
    # Extract the nodes and their lifespans from the graph_module
    # Difficult to just filter the list of specs returned by this due to
    # how we flag trainable weights.
    _ = update_all_tensors_lifetime(graph_module, graph_signature)

    # Filter specs based on alloc_graph_input and alloc_graph_output
    specs = collect_specs_from_nodes(
        graph_module.graph.nodes,
        graph_signature,
        do_assertion=False,
        ignore_graph_input=not alloc_graph_input,
        ignore_graph_output=not alloc_graph_output,
        ignore_mutable_buffers=not alloc_mutable_buffers,
    )

    # Get temporary specs for submodules to set aside space during execution
    # of submodules.
    submodule_bufsizes = _apply_algo_to_submodules(
        algo, graph_module, alignment, graph_signature
    )

    # Update `input_mem_buffer_sizes` in graph_module. This will allow existing
    # algos to work using `input_mem_buffer_sizes` or use
    # `non_const_buffer_sizes` directly.
    # pyre-ignore[16]: `torch.fx.GraphModule` has no attribute `input_mem_buffer_sizes`.
    graph_module.input_mem_buffer_sizes = submodule_bufsizes

    # Get extra padding for XNNPACK if needed
    extra_padding = 0
    if _contains_xnnpack_delegate(graph_module):
        extra_padding = 64

    # Pass the filtered specs to the algorithm
    bufsizes: list[int] = algo(
        alignment,
        specs,
        graph_module,
        graph_signature,
        extra_padding,
    )

    # pyre-ignore[6]: Incompatible parameter type [6]
    # In call `insert_calls_to_free`, for 2nd positional argument, expected `Set[TensorSpec]` but got `Iterable[TensorSpec]`
    insert_calls_to_free(graph_module, specs)

    graph_module.meta.update({"non_const_buffer_sizes": bufsizes})
    return bufsizes
