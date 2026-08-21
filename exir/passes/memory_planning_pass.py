# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from executorch.exir._warnings import deprecated
from executorch.exir.error import internal_assert
from executorch.exir.memory import alloc
from executorch.exir.memory_planning import (
    _is_out_var_node,
    apply_algo,
    collect_specs_from_nodes,
    filter_nodes,
    get_node_tensor_specs,
    MemoryPlanningAlgorithmSuite,
    Verifier,
)
from executorch.exir.operator.convert import get_out_args_from_opoverload
from executorch.exir.pass_base import PassBase, PassResult
from executorch.exir.tensor import ALIGNMENT, TensorSpec
from torch import fx
from torch.export.exported_program import ExportGraphSignature
from torch.fx import Node
from torch.utils import _pytree as pytree


# copied from https://stackoverflow.com/questions/75582932/python-how-can-i-print-the-function-name-of-a-partial-function
def _callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)


def _is_buffer(
    node: Node, graph_signature: ExportGraphSignature
) -> Tuple[bool, Optional[str]]:
    """
    Check if the node is buffer according to the provided graph signature.
    If it is one return its fqn as well
    """
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                return (True, fqn)
    return (False, None)


def _is_mutable_buffer(
    node: Node, graph_signature: ExportGraphSignature
) -> Tuple[bool, Optional[str]]:
    """
    Check if the node is mutable buffer according to the provided graph signature.
    If it is one return its fqn as well
    """
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                # if the buffer is mutated then record that
                if fqn in graph_signature.buffers_to_mutate.values():
                    return True, fqn
    return False, None


def _get_spec_from_node(node: fx.Node) -> TensorSpec:
    specs = get_node_tensor_specs(node)
    return specs[0]


def _insert_mutable_buffer_specs(
    state: "_MemoryPlanningState", gm: torch.fx.GraphModule, gs: ExportGraphSignature
):
    for node in gm.graph.nodes:
        is_mutable, fqn = _is_mutable_buffer(node, gs)
        if is_mutable:
            assert fqn
            spec = _get_spec_from_node(node)
            if (
                getattr(spec, "mem_id", None) is not None
                or getattr(spec, "mem_offset", None) is not None
            ):
                raise ValueError(
                    "Cannot share mutable buffers if they already have a mem_id or mem_offset assigned"
                )
            if fqn not in state.mutable_buffers.keys():
                state.mutable_buffers[fqn] = set()
            state.mutable_buffers[fqn].add(spec)
            continue
        is_buffer, fqn = _is_buffer(node, gs)
        # If it is not a mutable buffer it might just appear to be a buffer in this entry point. Think model.get_state()
        # So cache it and later double check that this buffer never appears mutable
        if is_buffer:
            assert fqn
            spec = _get_spec_from_node(node)
            if (
                getattr(spec, "mem_id", None) is not None
                or getattr(spec, "mem_offset", None) is not None
            ):
                raise ValueError(
                    "Cannot share mutable buffers if they already have a mem_id or mem_offset assigned"
                )
            if fqn not in state.maybe_mutable_buffers.keys():
                state.maybe_mutable_buffers[fqn] = set()
            state.maybe_mutable_buffers[fqn].add(spec)


def _check_default_mem_ids(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        for spec in collect_specs_from_nodes(
            filter_nodes(itertools.chain([node], node.args, node.kwargs.values())),
            None,
            ignore_graph_input=False,
            ignore_const=False,
            ignore_out_var_node=False,
            dedup=False,
            do_assertion=False,
            ignore_dynamic_unbound_tensor=False,
        ):
            mem_id = getattr(spec, "mem_id", None)
            if mem_id is not None and mem_id != 1:
                raise ValueError(
                    "Cannot share mutable buffers if all other tensors are not on the default mem_id of 1"
                )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _iter_unique_specs(graph_module: torch.fx.GraphModule) -> list[TensorSpec]:
    """Every TensorSpec reachable from graph node metas, deduplicated by identity.

    A spec can be referenced by several nodes; shifting one twice would corrupt
    the layout, so dedupe on id() rather than equality.
    """
    seen: set[int] = set()
    specs: list[TensorSpec] = []
    for node in graph_module.graph.nodes:
        # meta["spec"] may be a nested pytree (SpecPropPass), so flatten to leaves
        for spec in pytree.tree_leaves(node.meta.get("spec")):
            if not isinstance(spec, TensorSpec) or id(spec) in seen:
                continue
            seen.add(id(spec))
            specs.append(spec)
    return specs


def _resolve_inplace_root(spec: TensorSpec) -> TensorSpec:
    """Follow ``spec.inplace_base`` to the terminal spec it aliases.

    In-place ops produce a distinct result spec that aliases an input's spec
    (greedy gives it the same mem_id/mem_offset). The chain can be several links
    long; a ``seen`` set guards against a pathological cycle.
    """
    seen: set[int] = set()
    cur = spec
    while getattr(cur, "inplace_base", None) is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.inplace_base
    return cur


def _collect_declared_shared_specs(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    declared: frozenset[str],
) -> dict[str, TensorSpec]:
    specs_by_fqn: dict[str, TensorSpec] = {}
    for node in graph_module.graph.nodes:
        # A declared shared buffer may be read-only in one method while mutated in
        # another; collect it whenever it is a buffer so it front-packs to the same
        # slot in every method.
        is_buffer, fqn = _is_buffer(node, graph_signature)
        if is_buffer and fqn in declared:
            assert fqn is not None
            specs_by_fqn[fqn] = _get_spec_from_node(node)
    return specs_by_fqn


def _relocate_around_front_region(
    graph_module: torch.fx.GraphModule,
    declared_front: dict[int, int],
    reserved: dict[int, int],
    vacated: dict[int, list[tuple[int, int]]],
) -> None:
    """Move each non-declared spec up by its arena's reserved front region, less
    the declared columns vacated below it (so the arena does not grow). A spec that
    aliases a declared buffer in place follows that buffer to the front -- both
    ``mem_id`` and ``mem_offset`` -- so the aliased write lands on the shared
    buffer rather than a shifted copy, even if the alias were given a different
    arena.
    """
    for spec in _iter_unique_specs(graph_module):
        if id(spec) in declared_front:
            continue
        root = _resolve_inplace_root(spec)
        root_front = declared_front.get(id(root))
        if root_front is not None:
            spec.mem_id = root.mem_id
            spec.mem_offset = root_front
            continue
        mem_id = spec.mem_id
        mem_offset = spec.mem_offset
        if mem_id is None or mem_offset is None:
            continue
        below = sum(size for off, size in vacated.get(mem_id, []) if off < mem_offset)
        spec.mem_offset = reserved.get(mem_id, 0) + mem_offset - below


def _grow_arena_sizes(
    graph_module: torch.fx.GraphModule,
    reserved: dict[int, int],
    vacated: dict[int, list[tuple[int, int]]],
) -> None:
    """Update each front-packed arena's size. The scalar ``front - reclaimed`` term
    preserves any non-spec size the algorithm folded into ``non_const_buffer_sizes``
    (submodule / XNNPACK padding); the ``max`` against the real high-water mark of
    the final placements keeps the arena from being recorded smaller than actual
    usage if an algorithm ever leaves an unaligned slot.
    """
    sizes = list(graph_module.meta.get("non_const_buffer_sizes", []))
    high_water: dict[int, int] = {}
    for spec in _iter_unique_specs(graph_module):
        if spec.mem_id is None or spec.mem_offset is None:
            continue
        end = spec.mem_offset + spec.allocated_memory
        if end > high_water.get(spec.mem_id, 0):
            high_water[spec.mem_id] = end
    for mem_id, front in reserved.items():
        while len(sizes) <= mem_id:
            sizes.append(0)
        reclaimed = sum(size for _, size in vacated.get(mem_id, []))
        sizes[mem_id] = max(
            sizes[mem_id] + front - reclaimed, high_water.get(mem_id, 0)
        )
    graph_module.meta["non_const_buffer_sizes"] = sizes


@dataclass
class _MemoryPlanningState:
    mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    maybe_mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    graph_modules: List[torch.fx.GraphModule] = field(default_factory=list)


class MemoryPlanningPass(PassBase):
    def __init__(
        self,
        memory_planning_algo: Optional[Callable[..., List[int]]] = None,
        allow_lifetime_and_storage_overlap: bool = False,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        alloc_mutable_buffers: bool = True,
        share_mutable_buffers: bool = False,
        alignment: int = ALIGNMENT,
        shared_buffer_fqns: frozenset[str] | None = None,
    ) -> None:
        r"""
        alloc_graph_input/alloc_graph_output will have 4 different combinations
        to control if the memory planning algorithm need allocate memory for
        the graph input/output. The default behavior is the algorithm will allocate
        memory for both graph input and output.

        shared_buffer_fqns opts into the arena-aware sharing path. When set (and
        share_mutable_buffers is True), the named mutable buffers keep the real
        arena each is planned onto and are front-packed to an identical offset in
        every method, instead of being forced onto the legacy dedicated mem_id 2
        arena. This lifts the single-default-arena requirement so programs with a
        device/accelerator arena can still share a mutable buffer.
        """
        if memory_planning_algo is None:
            memory_planning_algo = MemoryPlanningAlgorithmSuite()
        if share_mutable_buffers and not alloc_mutable_buffers:
            raise ValueError(
                "share_mutable_buffers is only meaningful when alloc_mutable_buffers is True"
            )
        if shared_buffer_fqns is not None and not share_mutable_buffers:
            raise ValueError("shared_buffer_fqns requires share_mutable_buffers=True")
        if shared_buffer_fqns is not None and not shared_buffer_fqns:
            raise ValueError(
                "shared_buffer_fqns is empty; pass None for legacy sharing of all "
                "mutable buffers, or a non-empty set to front-pack named buffers"
            )
        self.memory_planning_algo: Callable[..., List[int]] = memory_planning_algo
        self.allow_lifetime_and_storage_overlap = allow_lifetime_and_storage_overlap
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.alloc_mutable_buffers = alloc_mutable_buffers
        self.share_mutable_buffers = share_mutable_buffers
        self.shared_buffer_fqns: frozenset[str] | None = shared_buffer_fqns
        self.alignment = alignment
        self.state = _MemoryPlanningState()
        # Resulting (mem_id, mem_offset, allocated_memory) of each declared
        # shared buffer from the first method it appears in. Later methods must
        # agree; a mismatch would alias two different tensors in one arena.
        self._shared_placement: dict[str, tuple[int | None, int | None, int]] = {}
        # Set by EdgeProgramManager.to_executorch() from the top-level
        # ExecutorchBackendConfig. When True, apply_algo partitions specs by
        # device so non-CPU buffers get their own memory arenas.
        self.enable_non_cpu_memory_planning: bool = False

    def _set_alloc_node_spec(self, graph_module: torch.fx.GraphModule) -> None:
        """
        Pass for setting all of the alloc node's specs. These nodes are created
        in the ToOutVarPass but do not have a spec.

        TODO(shunting): we probablly should setup the spec for memory.alloc node
          in the ToOutVarPass
        """
        for subgm in graph_module.modules():
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            for node in subgm.graph.nodes:
                if _is_out_var_node(node):
                    out_arg_names = get_out_args_from_opoverload(node.target)
                    if len(out_arg_names) == 1:
                        out_alloc_node = node.kwargs[out_arg_names[0]]
                        out_alloc_node.meta["spec"] = node.meta["spec"]
                        share_idx = node.meta.get("_share_alloc_with_arg_idx")
                        if share_idx is not None and share_idx < len(node.args):
                            input_node = node.args[share_idx]
                            if isinstance(input_node, Node):
                                base_spec = input_node.meta.get("spec")
                                if isinstance(base_spec, TensorSpec):
                                    node.meta["spec"].inplace_base = base_spec
                        continue
                    specs = get_node_tensor_specs(node)
                    i = 0
                    for out_arg in out_arg_names:
                        out_alloc_node = node.kwargs[out_arg]
                        if out_alloc_node is None:
                            warnings.warn(
                                f"Function {node.target}'s {out_arg} kwarg value is None",
                                stacklevel=1,
                            )
                            continue
                            # dont increment i as we dont have a spec for this node
                        internal_assert(
                            out_alloc_node.op == "call_function"
                            and out_alloc_node.target == alloc,
                            f"Out-var's node {out_alloc_node} has op {out_alloc_node.op} and target {out_alloc_node.target}",
                        )
                        internal_assert(
                            "spec" not in out_alloc_node.meta,
                            f"Out-var's allocation node {out_alloc_node} already has a spec assigned",
                        )
                        out_alloc_node.meta["spec"] = specs[i]
                        i += 1

    @deprecated(
        "MemoryPlanningPass.call() is deprecated as it does not handle graphs \
        with mutation, please use MemoryPlanningPass.run() instead",
        category=FutureWarning,
    )
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return self.run(graph_module)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        """
        A pass for memory planning. The actual algorithm used will be picked by
        memory_planning_algo
        """
        self._set_alloc_node_spec(graph_module)
        # TODO(shunting) if people have concern of adding a field to GraphModule
        # directly, we should define a GraphModule subclass that we can add our
        # customized fields. Using the graph_module object to convey information across
        # passes/stages is quite natural and avoid yet another 'context' data structure
        # to do the job.

        # Shared mutable buffers are excluded from the main algo (and placed on
        # the dedicated mem_id 2 arena later in run_multimethod) ONLY on the legacy
        # path. The arena-aware path (shared_buffer_fqns set) keeps them in the
        # algo so each lands on its real arena, then front-packs them below.
        plan_mutable_buffers_in_algo = self.alloc_mutable_buffers and (
            not self.share_mutable_buffers or self.shared_buffer_fqns is not None
        )

        _ = apply_algo(
            self.memory_planning_algo,
            graph_module,
            self.alignment,
            graph_signature,
            self.alloc_graph_input,
            self.alloc_graph_output,
            plan_mutable_buffers_in_algo,
            self.enable_non_cpu_memory_planning,
        )

        if self.share_mutable_buffers and graph_signature is not None:
            if self.shared_buffer_fqns is None:
                self.state.graph_modules.append(graph_module)
                _check_default_mem_ids(graph_module)
                _insert_mutable_buffer_specs(self.state, graph_module, graph_signature)
            else:
                self._front_pack_shared_buffers(graph_module, graph_signature)

        # TODO: make the verifier do the work recursively to handle
        # control flow
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            plan_mutable_buffers_in_algo,
            graph_signature,
        )

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            num_reuse_pairs = verifier.verify_storage_reuse(
                self.allow_lifetime_and_storage_overlap
            )
            logging.debug(
                f"The {getattr(self.memory_planning_algo, '__name__', repr(self.memory_planning_algo))} algorithm reuses storage for {num_reuse_pairs} pair of tensors"
            )
        verifier.verify_graph_input_output()
        if (
            callable(self.memory_planning_algo)
            and _callable_name(self.memory_planning_algo) == "greedy"
        ):
            # Only verify storage reuse for greedy algorithm
            # At the moment cadence backends memory planning fails this
            # I dont know if that is a valid thing but if it is we should adjust verify_storage_reuse function
            verifier.verify_storage_reuse()
        return PassResult(graph_module, True)

    def _front_pack_shared_buffers(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
    ) -> None:
        """Pin the declared shared buffers to a deterministic front region.

        Each declared buffer keeps its own arena (``mem_id``) and is packed, in
        sorted-FQN order, into a region reserved at the front of that arena.
        Every other spec in the arena is relocated up by that region, less the
        size of the declared columns that sat below it. A shared buffer has an
        infinite lifetime, so the algorithm gave it an exclusive slot that no
        other spec reuses; reclaiming that vacated slot in the same pass leaves no
        dead space, so the arena does not grow. A buffer has the same ``mem_id``
        and size in every method, so its resulting front offset is identical
        across methods by construction.
        """
        declared = self.shared_buffer_fqns
        assert declared is not None
        specs_by_fqn = _collect_declared_shared_specs(
            graph_module, graph_signature, declared
        )
        missing = declared - specs_by_fqn.keys()
        if missing:
            raise ValueError(
                "shared_buffer_fqns declares buffer(s) not present as buffers in "
                f"this method: {sorted(missing)}"
            )

        reserved: dict[int, int] = {}
        placement: dict[str, int] = {}
        # id() of each declared buffer's placeholder spec -> its front offset, so
        # specs that alias a declared buffer in place land on the same offset.
        declared_front: dict[int, int] = {}
        # Per arena, the (algo offset, size) each declared buffer was assigned.
        # Those slots are vacated by the move to the front and reclaimed so the
        # arena grows by nothing rather than by the whole front region.
        vacated: dict[int, list[tuple[int, int]]] = {}
        for fqn in sorted(specs_by_fqn):
            spec = specs_by_fqn[fqn]
            mem_id = spec.mem_id
            if mem_id is None:
                raise ValueError(
                    f"Declared shared buffer '{fqn}' was not assigned a memory arena"
                )
            if spec.mem_offset is None:
                raise ValueError(
                    f"Declared shared buffer '{fqn}' was not assigned an offset"
                )
            vacated.setdefault(mem_id, []).append(
                (spec.mem_offset, spec.allocated_memory)
            )
            offset = reserved.get(mem_id, 0)
            placement[fqn] = offset
            declared_front[id(spec)] = offset
            reserved[mem_id] = _align_up(offset + spec.allocated_memory, self.alignment)

        # Relocate the other specs around the reserved front region (declared
        # buffers are shifted first so nothing lands back on the region), then
        # place the declared buffers at the front and grow each arena to fit.
        _relocate_around_front_region(graph_module, declared_front, reserved, vacated)
        for fqn, offset in placement.items():
            specs_by_fqn[fqn].mem_offset = offset
        _grow_arena_sizes(graph_module, reserved, vacated)

        self._validate_shared_placement(specs_by_fqn)

    def _validate_shared_placement(self, specs_by_fqn: dict[str, TensorSpec]) -> None:
        for fqn, spec in specs_by_fqn.items():
            placement = (spec.mem_id, spec.mem_offset, spec.allocated_memory)
            prior = self._shared_placement.get(fqn)
            if prior is None:
                self._shared_placement[fqn] = placement
            elif prior != placement:
                raise ValueError(
                    f"Shared buffer '{fqn}' has inconsistent placement across "
                    f"methods: {prior} (first method) != {placement} (this "
                    "method); a declared shared buffer must have identical size "
                    "and resulting placement in every method"
                )

    def run_multimethod(self):
        """Resolve any memory planning done across entry points, called after run is called on all entry points."""
        if self.share_mutable_buffers:
            arena: int = 0

            # Every spec that shares an fqn is the same tensor! So we give it the same id and offset
            # anywhere it appears.
            for fqn, specs_set in self.state.mutable_buffers.items():
                specs = list(specs_set)
                # If the same buffer appears in mutable and maybe mutable then we know it is in fact mutable.
                if fqn in self.state.maybe_mutable_buffers.keys():
                    specs.extend(self.state.maybe_mutable_buffers[fqn])
                for spec in specs:
                    # Assume a default memory planning placed all activations on 1, place shared state on 2.
                    spec.mem_id = 2
                    spec.realign(self.alignment)
                    # State is persistent, so the memory never overlaps.
                    spec.mem_offset = arena
                # They should all be the same size since they are the same tensor, so just bump off the first.
                arena += specs[0].allocated_memory

            for graph_module in self.state.graph_modules:
                if len(graph_module.meta["non_const_buffer_sizes"]) != 2:
                    raise ValueError(
                        "Cannot share mutable state if not using default memory ids"
                    )
                graph_module.meta["non_const_buffer_sizes"].append(arena)
