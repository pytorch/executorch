# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
import operator
import typing
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from executorch.exir import memory
from executorch.exir.control_flow import while_loop as exir_while
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.error import internal_assert, InternalError
from executorch.exir.operator.convert import is_inplace_variant, is_out_variant
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec

from torch import fx
from torch.export.exported_program import ExportGraphSignature
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
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> None:
        self.graph_module = graph_module
        self.graph_signature = graph_signature
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output

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
                        f"Unexpected storage overlap: lhs {lhs_spec}, rhs {rhs_spec}"
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

        if "placeholder" in check_list:
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


def update_tensor_lifetime(spec: TensorSpec, node_idx: int) -> None:
    r"""
    Update the lifetime of the tensor to cover node_idx. A tensor's lifetime
    are represented by the index of the first and last node referring
    that tensor in its inputs/outputs.

    Arguments:
        spec: the TensorSpec for the tensor
        node_idx: extend the tensor's lifetime to cover node_idx
    """
    start, end = spec.lifetime
    start = node_idx if start is None or start > node_idx else start
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
            update_tensor_lifetime(spec, node_idx)
            specs.add(spec)
    return specs


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
    # the object will be available for index (last_used_index + 1)
    last_used_index: int


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


def _size_abs_dif(sobj: SharedObject, spec: TensorSpec) -> int:
    r"""
    Calculate the absolute different between the size of a shared object and
    a tensor.
    """
    return abs(sobj.size - spec.allocated_memory)


def pick_shared_obj(
    shared_objects: List[SharedObject], spec: TensorSpec
) -> SharedObject:
    r"""
    Pick the available shared object with closest size to the tensor.
    If there are no available shared object left, create a new one.
    """
    # TODO: do better than linear scan
    picked = None
    for sobj in shared_objects:
        if spec.lifetime[0] > sobj.last_used_index:
            if picked is None or _size_abs_dif(sobj, spec) < _size_abs_dif(
                picked, spec
            ):
                picked = sobj
                sobj.last_used_index = spec.lifetime[1]
                sobj.size = max(sobj.size, spec.allocated_memory)
    if picked is None:
        picked = SharedObject(
            len(shared_objects), -1, spec.allocated_memory, spec.lifetime[1]
        )
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


def greedy(
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: Optional[ExportGraphSignature] = None,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
) -> List[int]:
    spec2obj = {}
    shared_objects = defaultdict(list)
    # Don't do assertion in collect_specs_from_nodes if we have already encountered
    # and ignored some to_out_variant errors.
    do_assertion = not getattr(graph_module, "encounter_to_out_var_failure", False)
    # For each tensor, pick the available shared object with closest size to
    # the tensor. If there are no available shared object left, create a new
    # one.
    for spec in collect_specs_from_nodes(
        graph_module.graph.nodes,
        graph_signature,
        do_assertion=do_assertion,
        ignore_graph_input=not alloc_graph_input,
        ignore_graph_output=not alloc_graph_output,
    ):
        if spec.mem_id is None:
            spec.mem_id = 1
        spec.realign(alignment)
        spec2obj[spec] = pick_shared_obj(shared_objects[spec.mem_id], spec)

    if len(shared_objects) == 0:
        # Cannot find any tensor in the graph that needs to be allocated.
        # Return [0, 0] to be consistent with default behavior of naive.
        total_sizes = [0, 0]
    else:
        total_sizes = [0] * (max(shared_objects.keys()) + 1)
        for mem_id in shared_objects:
            input_total_size = 0
            if bufsizes := getattr(graph_module, "input_mem_buffer_sizes", None):
                if len(bufsizes) > mem_id:
                    input_total_size = bufsizes[mem_id]
            total_sizes[mem_id] = materialize_buffer(
                shared_objects[mem_id], input_total_size
            )

        # Since we now know the number of shared objects we need and the size of
        # each shared object, we can assign offset in the memory buffer for each
        # shared object.
        for spec, sobj in spec2obj.items():
            spec.mem_obj_id = sobj.idx
            spec.mem_offset = sobj.offset

    logging.debug(f"greedy algorithm returns bufsizes: {total_sizes}")
    return total_sizes


def naive(
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: Optional[ExportGraphSignature] = None,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
) -> List[int]:

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

    bufsizes = typing.cast(List[int], bufsizes)
    for spec in collect_specs_from_nodes(
        graph_module.graph.nodes,
        graph_signature,
        ignore_graph_input=not alloc_graph_input,
        ignore_graph_output=not alloc_graph_output,
    ):
        # assume a single memory layer which has mem_id 1
        if spec.mem_id is None:
            spec.mem_id = 1
        # allocate spec.allocated_memory bytes in the buffer
        # with the corresponding mem_id
        spec.realign(alignment)
        spec.mem_offset = _allocate_buf(bufsizes, spec.mem_id, spec.allocated_memory)

    logging.debug(f"naive algorithm returns bufsizes: {bufsizes}")
    return bufsizes


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


def apply_algo(
    algo: Callable[
        [torch.fx.GraphModule, int, Optional[ExportGraphSignature], bool, bool],
        List[int],
    ],
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: Optional[ExportGraphSignature] = None,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
) -> List[int]:
    """
    Recursively apply algo to graph_module and its submodules for control flow.

    Quite naively right now since it does not take the following optimizations
    into considerating:
    1. for conditional structure, true branch and false true does not overlap
       in lifetime and can share tensor storage
    2. tensors inside a submodule (e.g. true branch) has opportunities to share
       storage with tensors in the outer module.
    TODO: make these optimizations once we have some baseline working.
    """
    specs = update_all_tensors_lifetime(graph_module, graph_signature)
    bufsizes: List[int] = algo(
        graph_module, alignment, graph_signature, alloc_graph_input, alloc_graph_output
    )
    insert_calls_to_free(graph_module, specs)

    def handle_submodule(
        submodule_nd: torch.fx.Node, alloc_graph_input: bool = False
    ) -> None:
        nonlocal bufsizes
        assert submodule_nd.op == "get_attr"
        submodule = getattr(graph_module, submodule_nd.target)
        # memory planning for submodule need to be aware of the amount of
        # buffer already allocated.
        submodule.input_mem_buffer_sizes = bufsizes
        bufsizes = apply_algo(
            algo,
            submodule,
            alignment,
            graph_signature,
            alloc_graph_input=alloc_graph_input,
            alloc_graph_output=True,
        )
        submodule.meta.update({"non_const_buffer_sizes": bufsizes})

    for cond_node in get_cond_nodes(graph_module):
        handle_submodule(typing.cast(torch.fx.Node, cond_node.args[1]))
        handle_submodule(typing.cast(torch.fx.Node, cond_node.args[2]))

    for while_node in get_while_nodes(graph_module):
        handle_submodule(typing.cast(torch.fx.Node, while_node.args[0]))
        handle_submodule(typing.cast(torch.fx.Node, while_node.args[1]))
    # TODO: Add test coverage for map operator once dynamo tracing is
    # fully supported for this. T142287208
    for map_node in get_map_nodes(graph_module):
        handle_submodule(
            typing.cast(torch.fx.Node, map_node.args[0]), alloc_graph_input=True
        )

    graph_module.meta.update({"non_const_buffer_sizes": bufsizes})

    return bufsizes
