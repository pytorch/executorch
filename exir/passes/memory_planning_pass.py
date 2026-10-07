# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
from executorch.exir._warnings import deprecated
from executorch.exir.error import internal_assert
from executorch.exir.memory import alloc, view
from executorch.exir.memory_planning import (
    _build_non_const_buffer_device,
    _CPU_KEY,
    _device_order_key,
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
from executorch.exir.passes.replace_view_copy_with_view_pass import _ViewSpec
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import ALIGNMENT, calculate_aligned_num_bytes, TensorSpec
from torch import fx
from torch.export.exported_program import ExportGraphSignature
from torch.fx import Node
from torch.utils import _pytree as pytree

_DeviceKey = tuple[DeviceType, int]


def _format_device_key(device: _DeviceKey) -> str:
    """A device key as ``name:index``, for a diagnostic to read as prose."""
    return f"{device[0].name}:{device[1]}"


# copied from https://stackoverflow.com/questions/75582932/python-how-can-i-print-the-function-name-of-a-partial-function
def _callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)


# Marks a planning algorithm that checks its own returned bufsizes against a
# declared budget. Read as an attribute rather than matched against a list of
# class names here: banked_memory_planning imports this module, so naming its
# class would be an import cycle.
_BUDGET_ATTR = "plans_against_a_memory_budget"


def _budget_checking_algorithms(algo: Callable[..., Any]) -> list[str]:
    """The name of every algorithm ``algo`` would run that checks a budget.

    The marker is read off the object the pass was handed and off each entry of
    its ``algo_list``, because a MemoryPlanningAlgorithmSuite runs those entries,
    so a constraint that rules an algorithm out has to rule out a suite
    containing it and has to name the entry the caller must act on.

    Nothing is looked through. A ``functools.partial`` or a bound method
    forwards neither attribute lookup nor ``algo_list``, so an algorithm handed
    over behind one declares nothing here and the arena is appended without a
    word.
    """
    if getattr(algo, _BUDGET_ATTR, False):
        return [_callable_name(algo)]
    return [
        _callable_name(entry)
        for entry in getattr(algo, "algo_list", [])
        if getattr(entry, _BUDGET_ATTR, False)
    ]


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


def _move_memory_meta_to_spec(node: Node) -> None:
    """Move storage sharing metadata from node.meta to node.meta["spec"].

    Only applies if _share_alloc_with_arg_idx is set.
    """
    share_idx = node.meta.get("_share_alloc_with_arg_idx")
    shared_alloc_offset = node.meta.get("_shared_alloc_offset")

    if share_idx is None:
        if shared_alloc_offset is not None:
            raise ValueError(
                "_shared_alloc_offset meta was set but not _share_alloc_with_arg_idx."
            )
        return
    if shared_alloc_offset is None:
        shared_alloc_offset = 0

    if not isinstance(share_idx, int):
        raise TypeError("_share_alloc_with_arg_idx must be an int")
    if not isinstance(shared_alloc_offset, int):
        raise TypeError("_shared_alloc_offset must be an int")

    output_spec = node.meta.get("spec")
    if not isinstance(output_spec, TensorSpec):
        raise TypeError(
            "_share_alloc_with_arg_idx requires node.meta['spec'] to be a TensorSpec"
        )

    if share_idx < 0 or share_idx >= len(node.args):
        raise IndexError("_share_alloc_with_arg_idx must index node.args")

    input_node = node.args[share_idx]
    if not isinstance(input_node, Node):
        raise TypeError("_share_alloc_with_arg_idx must reference a Node argument")

    base_spec = input_node.meta.get("spec")
    if not isinstance(base_spec, TensorSpec):
        raise TypeError(
            "_share_alloc_with_arg_idx must reference an argument with a TensorSpec"
        )

    output_spec.storage_base = base_spec
    output_spec.storage_base_offset = shared_alloc_offset


def _iter_unique_specs(
    graph_module: torch.fx.GraphModule,
) -> list[tuple[Node, TensorSpec]]:
    """Every TensorSpec reachable from graph node metas, once each.

    A spec can be referenced by several nodes, and renumbering one twice would
    corrupt the layout.

    A _ViewSpec is skipped rather than deduplicated, because id() is not enough
    to recognize it as a repeat: ReplaceViewCopyWithViewPass gives one to every
    ``memory.view`` node, and it is a separate object that forwards mem_id --
    and every other placement field -- to the spec of its base, which is an
    argument of the view node and so is reached on a node of this graph. The
    skip is on the type as well as on ``node.target``, because
    ``_alias_inplace_result_specs`` can leave a _ViewSpec on an in-place op's
    node.

    Each spec is paired with a node carrying it, which is what names it in
    diagnostics: the first such node, except that an alloc node is given up for
    the next one found, because its name says nothing about the tensor.
    """
    index_by_spec: dict[int, int] = {}
    specs: list[tuple[Node, TensorSpec]] = []
    for node in graph_module.graph.nodes:
        if node.target == view:
            continue
        # meta["spec"] may be a nested pytree (SpecPropPass), so flatten to leaves
        for spec in pytree.tree_leaves(node.meta.get("spec")):
            if not isinstance(spec, TensorSpec) or isinstance(spec, _ViewSpec):
                continue
            index = index_by_spec.get(id(spec))
            if index is None:
                index_by_spec[id(spec)] = len(specs)
                specs.append((node, spec))
            elif specs[index][0].target == alloc:
                specs[index] = (node, spec)
    return specs


def _refuse_in_place_aliases(graph_module: torch.fx.GraphModule) -> None:
    """Refuse an in-place alias in a method planned with ``shared_buffer_fqns``.

    A spec carrying a ``storage_base`` is placed inside another spec's storage,
    at an offset the algorithm works out from where it put the base. A declared
    buffer is not placed by the algorithm at all, so an alias that reaches one
    would be left with no address to be an offset into; which aliases reach a
    declared buffer is a question about the whole chain, and this path does not
    walk one, so every in-place alias in the method is refused.
    """
    for node, spec in _iter_unique_specs(graph_module):
        if spec.storage_base is None:
            continue
        raise NotImplementedError(
            f"Tensor '{node.name}' is placed inside another tensor's storage "
            "(TensorSpec.storage_base), which a method planned with "
            "shared_buffer_fqns may not do: the declared buffers are placed by "
            "hand after the planning algorithm has returned, and nothing here "
            "follows a storage chain to tell an alias of one from an alias of "
            "anything else. Plan this method without buffer sharing -- drop "
            "shared_buffer_fqns and share_mutable_buffers -- which returns "
            "every tensor to the planning algorithm; it does place an alias "
            "inside its base. Dropping shared_buffer_fqns alone leaves the "
            "legacy sharing path, which withholds every mutable buffer from "
            "the algorithm just the same."
        )


def _collect_declared_shared_specs(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    declared_fqns: frozenset[str],
) -> dict[str, TensorSpec]:
    """The placeholder spec of each declared shared buffer this method uses.

    Collection is by ``_is_buffer`` rather than ``_is_mutable_buffer``: a
    declared buffer may be read-only in one method and mutated in another, and
    it must land in the same slot either way. A name missing from this method is
    not an error; a name missing from every method is caught later, in
    ``_shared_buffer_geometry``.

    Raises when a buffer already carries a placement. Everything declared here
    is about to be moved into the dedicated arena, so a pin to a custom pool
    could only be discarded.
    """
    specs_by_fqn: dict[str, TensorSpec] = {}
    for node in graph_module.graph.nodes:
        is_buffer, fqn = _is_buffer(node, graph_signature)
        if is_buffer and fqn in declared_fqns:
            assert fqn is not None
            spec = _get_spec_from_node(node)
            if spec.mem_id is not None or spec.mem_offset is not None:
                raise ValueError(
                    f"Cannot give '{fqn}' a dedicated shared arena if it "
                    "already has a mem_id or mem_offset assigned"
                )
            internal_assert(
                fqn not in specs_by_fqn,
                f"buffer '{fqn}' reaches this graph through more than one "
                "placeholder; only one of them can be placed in the shared "
                "arena and the other would be left unplaced",
            )
            specs_by_fqn[fqn] = spec
    return specs_by_fqn


def _mutable_buffer_fqns(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> set[str]:
    """Which buffers this method mutates, by fqn.

    The dedicated-arena path splits this set in two. Whether a declared buffer
    is mutated in *no* method is a question only run_multimethod can answer, and
    one it has to, because the emitter reads a placement on a buffer placeholder
    as proof that the buffer is mutable somewhere. The undeclared ones are what
    naming any buffer at all costs the caller, which run_multimethod warns
    about.
    """
    mutated: set[str] = set()
    for node in graph_module.graph.nodes:
        is_mutable, fqn = _is_mutable_buffer(node, graph_signature)
        if is_mutable:
            assert fqn is not None
            mutated.add(fqn)
    return mutated


def _initialized_declared_fqns(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    declared_fqns: frozenset[str],
) -> set[str]:
    """Which declared shared buffers this method carries an initializer for.

    ``et_init_buffer`` -- what InitializedMutableBufferPass sets -- makes the
    emitter serialize the buffer's stored bytes next to its placement.
    run_multimethod refuses the combination, so this only has to answer which
    methods have one.
    """
    initialized: set[str] = set()
    for node in graph_module.graph.nodes:
        is_buffer, fqn = _is_buffer(node, graph_signature)
        if is_buffer and fqn in declared_fqns and node.meta.get("et_init_buffer"):
            assert fqn is not None
            initialized.add(fqn)
    return initialized


def _used_buffer_fqns(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> set[str]:
    """Which of this method's buffers its graph actually reaches.

    A buffer registered on the module is lifted into every method's placeholders
    whether or not that method touches it, so a buffer can arrive here with no
    users at all. A method that never reaches a buffer is neither a method that
    reads it before a writer has run nor a method that can disagree with another
    about where it sits.
    """
    used: set[str] = set()
    for node in graph_module.graph.nodes:
        is_buffer, fqn = _is_buffer(node, graph_signature)
        if is_buffer and len(node.users) > 0:
            assert fqn is not None
            used.add(fqn)
    return used


def _submodule_arena_ids(graph_module: torch.fx.GraphModule) -> set[int]:
    """Arena indices carried by specs inside this graph's control-flow submodules.

    ``apply_algo`` plans a submodule by recursing into it with per-device
    planning off, so its specs land on CPU arena indices local to that
    recursion, and the parent reserves the bottom of its own CPU arena of the
    same index for them. Nothing renumbers those specs afterwards, so an index
    in this set has to go on meaning the same CPU arena.
    """
    ids: set[int] = set()
    for submodule in graph_module.modules():
        if not isinstance(submodule, torch.fx.GraphModule):
            continue
        if submodule is graph_module:
            continue
        for _, spec in _iter_unique_specs(submodule):
            if spec.mem_id is not None:
                ids.add(spec.mem_id)
    return ids


def _dense_arena_devices(graph_module: torch.fx.GraphModule) -> list[_DeviceKey]:
    """The device of every planned arena, one entry per arena.

    Inverts ``non_const_buffer_device``, which is empty for a CPU-only program
    and otherwise carries an entry only for the arenas that are not CPU index
    zero: an index it does not name is CPU:0, and ``non_const_buffer_sizes``
    gives the number of arenas to expand it to.
    """
    arena_devices: list[_DeviceKey] = [_CPU_KEY] * len(
        graph_module.meta["non_const_buffer_sizes"]
    )
    for entry in graph_module.meta["non_const_buffer_device"]:
        arena_devices[entry.buffer_idx] = (entry.device_type, entry.device_index)
    return arena_devices


def _rebuild_device_entries(
    graph_module: torch.fx.GraphModule, arena_devices: list[_DeviceKey]
) -> None:
    """Refresh the sparse arena device list from a dense one.

    The sparse encoding is ``_build_non_const_buffer_device``'s rule and is left
    there, so that which arenas get an entry is decided in one place.

    Rewritten in place: to_executorch copies this meta dict into the exported
    program by a shallow dict.update, so rebinding the slot afterwards would
    leave the emitter reading the stale value.
    """
    graph_module.meta["non_const_buffer_device"][:] = _build_non_const_buffer_device(
        arena_devices
    )


@dataclass
class _SharedArenaMethod:
    """One planned method's contribution to the cross-method arena layout.

    ``arena_devices`` is a snapshot of how this method planned its own arenas,
    positionally aligned with ``non_const_buffer_sizes`` as they stood then:
    entry *i* names the device this method put arena *i* on. The cross-method
    numbering reads this record and does not rewrite it, so it goes on
    describing the method's own numbering. ``shared_index`` names which of
    those entries are the method's own shared arenas, so they are not counted
    as regular arenas of their device.

    ``mutated``, ``read`` and ``initialized`` are subsets of
    ``declared_specs``: the buffers this method writes, the ones it reaches
    without writing, and the ones whose placeholder carries et_init_buffer.
    ``read`` is not the complement of ``mutated`` -- a buffer no method touches
    is lifted into the method's placeholders all the same.
    ``undeclared_mutable`` names the mutable buffers of this method that
    ``shared_buffer_fqns`` does not. ``used_buffers`` names every buffer this
    method reads or writes, declared or not, which is what says whether a
    second method exists to disagree with this one about where a buffer sits.
    """

    graph_module: torch.fx.GraphModule
    arena_devices: list[_DeviceKey]
    shared_index: dict[_DeviceKey, int]
    declared_specs: dict[str, TensorSpec]
    mutated: set[str]
    read: set[str]
    initialized: set[str]
    undeclared_mutable: set[str]
    used_buffers: set[str]


@dataclass
class _SharedArenaLayout:
    """The cross-method arena numbering every method is renumbered into.

    ``base[d]`` is the first arena index owned by device *d*; the regular arenas
    of *d* run from there, as many of them as the most any single method needed,
    so that a method using fewer is zero-padded rather than shifted.
    ``shared_id[d]`` is the dedicated shared arena appended to that block, and
    is present only for the devices that have a declared buffer -- a device can
    have a block without one.
    """

    base: dict[_DeviceKey, int]
    shared_id: dict[_DeviceKey, int]
    num_arenas: int


class _SharedBufferGeometry(NamedTuple):
    """Everything about a declared shared buffer that every method must agree on.

    Aligned allocation size alone does not identify a tensor: ``float32[3]`` and
    ``float32[4]`` both round to 16 bytes. ``allocated_memory`` is carried
    because the arena is laid out from it.
    """

    allocated_memory: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    layout: torch.layout
    device: _DeviceKey

    def differences_from(self, other: "_SharedBufferGeometry") -> str:
        """The fields the two disagree on, named, with both values."""
        parts: list[str] = []
        for field_name in self._fields:
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            if mine == theirs:
                continue
            if field_name == "device":
                mine = _format_device_key(mine)
                theirs = _format_device_key(theirs)
            parts.append(f"{field_name} {mine} against {theirs}")
        return ", ".join(parts)


@dataclass
class _MemoryPlanningState:
    mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    maybe_mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    graph_modules: List[torch.fx.GraphModule] = field(default_factory=list)
    shared_arena_methods: List[_SharedArenaMethod] = field(default_factory=list)


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
        shared_buffer_fqns: Optional[Iterable[str]] = None,
    ) -> None:
        r"""
        alloc_graph_input/alloc_graph_output will have 4 different combinations
        to control if the memory planning algorithm need allocate memory for
        the graph input/output. The default behavior is the algorithm will allocate
        memory for both graph input and output.

        shared_buffer_fqns opts into the dedicated-arena sharing path, and
        requires share_mutable_buffers=True. The named buffers are withheld from
        the planning algorithm and given a memory arena of their own on each
        device that owns one, at an arena index and offset that mean the same
        thing in every method. This generalizes the legacy CPU-only mem_id 2
        arena to any number of devices and coexists with custom memory pools,
        which the legacy path rejects outright.

        Naming buffers replaces the legacy path rather than adding to it: every
        mutable buffer the argument does not name goes back to the planning
        algorithm and is no longer shared across methods. run_multimethod warns
        about the ones it finds.
        """
        if memory_planning_algo is None:
            memory_planning_algo = MemoryPlanningAlgorithmSuite()
        if share_mutable_buffers and not alloc_mutable_buffers:
            raise ValueError(
                "share_mutable_buffers is only meaningful when alloc_mutable_buffers is True"
            )
        if shared_buffer_fqns is not None and not share_mutable_buffers:
            raise ValueError("shared_buffer_fqns requires share_mutable_buffers=True")
        if isinstance(shared_buffer_fqns, str):
            # A string is iterable, so it would otherwise be taken as a
            # collection of one-character names, which no fully qualified name
            # can match.
            raise TypeError(
                "shared_buffer_fqns must be a collection of fully qualified "
                f"names, not the single string {shared_buffer_fqns!r}; pass "
                f"frozenset({{{shared_buffer_fqns!r}}})"
            )
        # Materialized rather than kept as given: the names are used as a
        # membership test once per placeholder per method, so a one-shot
        # iterable would be empty from the second test on.
        declared_fqns: Optional[frozenset[str]] = (
            None if shared_buffer_fqns is None else frozenset(shared_buffer_fqns)
        )
        if declared_fqns is not None and not declared_fqns:
            raise ValueError(
                "shared_buffer_fqns is empty; pass None for legacy sharing of all "
                "mutable buffers, or a non-empty set to give named buffers a "
                "dedicated arena"
            )
        self.memory_planning_algo: Callable[..., List[int]] = memory_planning_algo
        self.allow_lifetime_and_storage_overlap = allow_lifetime_and_storage_overlap
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.alloc_mutable_buffers = alloc_mutable_buffers
        self.share_mutable_buffers = share_mutable_buffers
        self.shared_buffer_fqns: Optional[frozenset[str]] = declared_fqns
        self.alignment = alignment
        self.state = _MemoryPlanningState()
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

                        _move_memory_meta_to_spec(node)
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

        dedicated_arena_path = (
            self.share_mutable_buffers
            and graph_signature is not None
            and self.shared_buffer_fqns is not None
        )
        if self.shared_buffer_fqns is not None and graph_signature is None:
            # Which placeholders are buffers is knowable only from the graph
            # signature, so with none there is nothing to name and nothing to
            # place. The program is built exactly as it would be with
            # shared_buffer_fqns=None, and three entry points plan a caller's
            # own pass this way -- ExirExportedProgram.to_executorch,
            # LoweredBackendModule.program() and .buffer() -- so this says so
            # rather than raising.
            warnings.warn(
                "shared_buffer_fqns names "
                f"{sorted(self.shared_buffer_fqns)}, but memory planning was "
                "given no graph signature, so none of them is shared: they are "
                "planned as ordinary tensors, and the checks that a declared "
                "name is a buffer of some method and is mutated by some method "
                "do not run. Plan through EdgeProgramManager.to_executorch, "
                "which passes the signature, or drop shared_buffer_fqns.",
                UserWarning,
                stacklevel=2,
            )

        # Both sharing paths place their buffers after the algorithm has run, so
        # both withhold them from it. The legacy path withholds every mutable
        # buffer; the dedicated-arena path withholds only the declared ones.
        declared_specs: dict[str, TensorSpec] = {}
        declared_mutated: set[str] = set()
        declared_read: set[str] = set()
        undeclared_mutable: set[str] = set()
        used_buffers: set[str] = set()
        declared_initialized: set[str] = set()
        excluded_spec_ids: set[int] = set()
        if dedicated_arena_path:
            assert graph_signature is not None
            assert self.shared_buffer_fqns is not None
            declared_specs = _collect_declared_shared_specs(
                graph_module, graph_signature, self.shared_buffer_fqns
            )
            mutable_fqns = _mutable_buffer_fqns(graph_module, graph_signature)
            # A buffer this method only overwrites can reach planning with an
            # unused placeholder, so the buffers it writes are unioned in rather
            # than assumed to be among the ones its graph reads.
            used_buffers = (
                _used_buffer_fqns(graph_module, graph_signature) | mutable_fqns
            )
            declared_mutated = mutable_fqns & self.shared_buffer_fqns
            declared_read = (used_buffers & self.shared_buffer_fqns) - declared_mutated
            undeclared_mutable = mutable_fqns - self.shared_buffer_fqns
            declared_initialized = _initialized_declared_fqns(
                graph_module, graph_signature, self.shared_buffer_fqns
            )
            _refuse_in_place_aliases(graph_module)
            excluded_spec_ids = {id(spec) for spec in declared_specs.values()}

        plan_mutable_buffers_in_algo = self.alloc_mutable_buffers and (
            not self.share_mutable_buffers or dedicated_arena_path
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
            excluded_spec_ids,
        )

        if self.share_mutable_buffers and graph_signature is not None:
            if self.shared_buffer_fqns is None:
                self.state.graph_modules.append(graph_module)
                _check_default_mem_ids(graph_module)
                _insert_mutable_buffer_specs(self.state, graph_module, graph_signature)
            else:
                self._add_shared_arenas(
                    graph_module,
                    declared_specs,
                    declared_mutated,
                    declared_read,
                    declared_initialized,
                    undeclared_mutable,
                    used_buffers,
                )

        # TODO: make the verifier do the work recursively to handle
        # control flow
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            # Legacy sharing leaves its buffers with no mem_obj_id, which
            # verify_storage_reuse rejects as soon as it pairs one with a
            # planned spec, so those specs are skipped. The dedicated-arena
            # path assigns an id for every spec it places by hand.
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

    def _device_key(self, spec: TensorSpec) -> _DeviceKey:
        """The arena-owning device of a spec.

        With per-device planning off, ``apply_algo`` buckets every spec into
        CPU:0 regardless of ``spec.device``, so the shared arenas must follow.
        """
        if not self.enable_non_cpu_memory_planning:
            return _CPU_KEY
        return (spec.device, spec.device_index)

    def _refuse_a_budget_checking_algorithm(self) -> None:
        """Refuse to append a shared arena after a budget-checking algorithm.

        Asked here, where the arena is about to be appended, rather than in
        __init__: ``memory_planning_algo`` and a suite's ``algo_list`` are both
        ordinary attributes, so what actually runs can be rebound after the
        pass is built.
        """
        budget_algos = _budget_checking_algorithms(self.memory_planning_algo)
        if not budget_algos:
            return
        named = (
            budget_algos[0]
            if len(budget_algos) == 1
            else f"{', '.join(budget_algos[:-1])} and {budget_algos[-1]}"
        )
        verb = "does" if len(budget_algos) == 1 else "do"
        raise ValueError(
            "shared_buffer_fqns cannot be combined with a planning algorithm "
            f"that declares {_BUDGET_ATTR}, and {named} "
            f"{verb}. A dedicated shared arena is appended after the algorithm "
            "has returned, so it is not one of the arenas that budget "
            "describes and nothing is charged for its bytes. Plan this program "
            "with the greedy algorithm instead."
        )

    def _add_shared_arenas(
        self,
        graph_module: torch.fx.GraphModule,
        declared_specs: dict[str, TensorSpec],
        declared_mutated: set[str],
        declared_read: set[str],
        declared_initialized: set[str],
        undeclared_mutable: set[str],
        used_buffers: set[str],
    ) -> None:
        """Append this method's shared arenas and place its declared buffers.

        The arena indices and the offsets are provisional: run_multimethod
        restates every arena size and device in the numbering agreed across
        methods and overwrites each declared buffer's mem_id, mem_offset and
        mem_obj_id. What it keeps from this call is the record appended to
        state.shared_arena_methods. The provisional values are written anyway
        for a caller invoking run() directly, which never reaches
        run_multimethod.

        The realignment is not provisional: a declared buffer is withheld from
        the algorithm, which is what realigns everything else, and nothing
        realigns it again -- so the alignment set here is what
        _shared_buffer_geometry records as allocated_memory, and that is what
        sizes the final arena and spaces the buffers in it.
        """
        if self.enable_non_cpu_memory_planning:
            for node, spec in _iter_unique_specs(graph_module):
                if spec.device == DeviceType.CPU and spec.device_index != 0:
                    raise NotImplementedError(
                        f"'{node.name}' is on {_format_device_key((spec.device, spec.device_index))}, "
                        "and shared_buffer_fqns supports only CPU index zero "
                        "among the host devices. Put the host tensors on "
                        "CPU:0. A non-zero host device index is out of this "
                        "argument's scope: no arena layout here is designed "
                        "or tested for one."
                    )

        sizes = graph_module.meta["non_const_buffer_sizes"]
        arena_devices = _dense_arena_devices(graph_module)

        by_device: dict[_DeviceKey, list[str]] = {}
        for fqn, spec in declared_specs.items():
            by_device.setdefault(self._device_key(spec), []).append(fqn)

        if by_device:
            self._refuse_a_budget_checking_algorithm()

        shared_index: dict[_DeviceKey, int] = {}
        for device in sorted(by_device, key=_device_order_key):
            shared_index[device] = len(sizes)
            offset = 0
            for obj_id, fqn in enumerate(sorted(by_device[device])):
                spec = declared_specs[fqn]
                spec.realign(self.alignment)
                spec.mem_id = shared_index[device]
                spec.mem_offset = offset
                spec.mem_obj_id = obj_id
                offset = calculate_aligned_num_bytes(
                    offset + spec.allocated_memory, self.alignment
                )
            # apply_algo has already returned by now, so an algorithm that
            # checks its own bufsizes against a budget ran without this arena
            # in them. One carrying _BUDGET_ATTR is refused above; one that
            # makes the check without saying so is not.
            sizes.append(offset)
            arena_devices.append(device)

        _rebuild_device_entries(graph_module, arena_devices)
        self.state.shared_arena_methods.append(
            _SharedArenaMethod(
                graph_module=graph_module,
                arena_devices=list(arena_devices),
                shared_index=shared_index,
                declared_specs=declared_specs,
                mutated=declared_mutated,
                read=declared_read,
                initialized=declared_initialized,
                undeclared_mutable=undeclared_mutable,
                used_buffers=used_buffers,
            )
        )

    def _shared_buffer_geometry(self) -> dict[str, _SharedBufferGeometry]:
        """The geometry of each declared buffer, agreed across methods.

        A buffer is one tensor no matter how many methods reference it, so a
        method that disagrees about it would make the shared arena mean two
        different things.

        This is also where the whole-program checks on the declared names live
        -- that each names a buffer of some method, that each is mutated by some
        method, that none has zero elements and that none carries an
        initializer -- because none of them can be answered from one method
        alone.

        Every spec read here was realigned by _add_shared_arenas before its
        method was recorded, so the sizes compared are the pass's own.
        """
        geometry: dict[str, _SharedBufferGeometry] = {}
        # A method is identified by when it was planned, because run() is not
        # told the name of the method it is planning.
        first_seen: dict[str, int] = {}
        for order, method in enumerate(self.state.shared_arena_methods):
            for fqn, spec in method.declared_specs.items():
                seen = _SharedBufferGeometry(
                    allocated_memory=spec.allocated_memory,
                    dtype=spec.dtype,
                    shape=tuple(spec.shape),
                    stride=tuple(spec.stride),
                    layout=spec.layout,
                    device=self._device_key(spec),
                )
                if fqn not in geometry:
                    geometry[fqn] = seen
                    first_seen[fqn] = order
                elif geometry[fqn] != seen:
                    raise ValueError(
                        f"Shared buffer '{fqn}' is described differently by the "
                        f"method planned {first_seen[fqn]} and the method "
                        f"planned {order} (numbered in the order memory "
                        "planning ran on them): "
                        f"{geometry[fqn].differences_from(seen)}, the first "
                        "value in each pair being the earlier method's; a "
                        "declared shared buffer must be the same tensor on the "
                        "same device in every method"
                    )
        assert self.shared_buffer_fqns is not None
        missing = self.shared_buffer_fqns - geometry.keys()
        if missing:
            raise ValueError(
                "shared_buffer_fqns declares buffer(s) that are not a buffer of "
                f"any method: {sorted(missing)}"
            )
        empty = sorted(fqn for fqn, g in geometry.items() if g.allocated_memory == 0)
        if empty:
            raise ValueError(
                f"shared_buffer_fqns declares buffer(s) with no elements: {empty}. "
                "A tensor with no elements holds no state, so there is nothing "
                "for two methods to share through it, and where it is the only "
                "declared buffer of its device it leaves that device an arena "
                "of zero bytes. Leave these out of shared_buffer_fqns."
            )
        never_mutated = geometry.keys() - {
            fqn for method in self.state.shared_arena_methods for fqn in method.mutated
        }
        if never_mutated:
            raise ValueError(
                "shared_buffer_fqns declares buffer(s) that no method mutates: "
                f"{sorted(never_mutated)}. The emitter reads any buffer "
                "placeholder carrying a mem_id and a mem_offset as a mutable "
                "buffer, and emits a mutable buffer without its state_dict "
                "data unless a pass has set et_init_buffer on it, so giving one "
                "of these a place in the dedicated arena would serve it from "
                "uninitialized planned memory instead of its stored values. "
                "Leave it out of shared_buffer_fqns and it stays a constant."
            )
        initialized = {
            fqn
            for method in self.state.shared_arena_methods
            for fqn in method.initialized
        }
        if initialized:
            raise ValueError(
                "shared_buffer_fqns declares buffer(s) that carry an "
                f"initializer: {sorted(initialized)}. A buffer whose "
                "placeholder carries et_init_buffer -- what "
                "InitializedMutableBufferPass sets -- is emitted with its "
                "stored bytes, and the runtime copies those bytes into the "
                "buffer's planned allocation every time it loads a method that "
                "has it. The dedicated arena makes that one allocation for the "
                "whole program, so any second load overwrites the live value "
                "with the initial one, and loading is not once per method -- "
                "PyProgram::load_method builds a fresh Method every call. "
                "Either leave the buffer out of shared_buffer_fqns, or stop "
                "initializing it and have a method write it before any method "
                "reads it."
            )
        return geometry

    def _warn_about_read_only_methods(self) -> None:
        """Report each declared buffer some method reads without writing.

        Mutation by *some* method is all _shared_buffer_geometry asks, which
        leaves the reader that runs before the writer holding uninitialized
        arena bytes where an unshared build would have served the stored
        constant. Refusing that shape is not an option -- it is the point of the
        feature for a prefill/decode pair -- so it is warned about, and it is
        warned about because nothing at runtime will: the method reads whatever
        the arena holds and reports success.

        Methods are named by when they were planned, which is all that is known
        here: run() is not told the name of the method it plans.
        """
        readers: dict[str, list[int]] = {}
        for order, method in enumerate(self.state.shared_arena_methods):
            for fqn in sorted(method.read):
                readers.setdefault(fqn, []).append(order)
        if not readers:
            return
        named = "; ".join(
            f"'{fqn}' in method(s) {', '.join(str(order) for order in orders)}"
            for fqn, orders in sorted(readers.items())
        )
        warnings.warn(
            "shared_buffer_fqns declares buffer(s) that some method only reads: "
            f"{named} (methods numbered in the order memory planning ran on "
            "them). A declared buffer is emitted without its state_dict data, "
            "so the dedicated arena holds whatever the runtime left there until "
            "a method writes it, and one of these methods called before any "
            "writer has run reads that rather than the registered value. Call "
            "a method that writes the whole buffer first: a writer that reads "
            "before it writes, as state.add_(x) does, accumulates onto the "
            "arena's contents instead of onto the registered value, so calling "
            "it first leaves the buffer wrong for every later reader too.",
            UserWarning,
            stacklevel=2,
        )

    def _warn_about_undeclared_mutable_buffers(self) -> None:
        """Report the mutable buffers naming any buffer stopped sharing.

        ``share_mutable_buffers`` on its own gives every mutable buffer of the
        program one arena index and one offset, the same in every method.
        Adding ``shared_buffer_fqns`` hands every buffer it does not name back to
        the planning algorithm, which plans each method on its own and has no
        reason to put it at the same address twice. Each method still keeps the
        bytes for the whole of its own run -- update_tensor_lifetime never frees
        a mutable buffer -- so what is lost is the agreement between methods, and
        nothing at runtime reports that.

        Only the buffers more than one method reaches are named: a buffer one
        method has to itself has no second placement to disagree with.
        """
        methods = self.state.shared_arena_methods
        undeclared = {
            fqn
            for method in methods
            for fqn in method.undeclared_mutable
            if sum(fqn in other.used_buffers for other in methods) > 1
        }
        if not undeclared:
            return
        warnings.warn(
            "share_mutable_buffers shares every mutable buffer, but "
            "shared_buffer_fqns names only some of them, so "
            f"{sorted(undeclared)} is not shared across methods. A method that "
            "writes one of these places it wherever its own plan puts it, and "
            "nothing holds two such methods to one address, so a write through "
            "one is not what the other reads; a method that only reads one gets "
            "no placement for it at all and returns the registered value. Name "
            "these buffers in shared_buffer_fqns too, or drop "
            "shared_buffer_fqns for the legacy path, which shares every mutable "
            "buffer but requires every other tensor on arena 1.",
            UserWarning,
            stacklevel=2,
        )

    def _build_arena_layout(
        self, shared_devices: set[_DeviceKey]
    ) -> _SharedArenaLayout:
        """Lay out a common arena numbering across every planned method.

        Each device gets a contiguous block wide enough for the most arenas any
        one method put on it, followed by that device's dedicated shared arena
        if it has one. Indices are handed out in order, so a custom memory pool
        is one more arena on its device: it widens that device's block, and the
        shared arena follows the widened block.
        """
        width: dict[_DeviceKey, int] = {}
        for method in self.state.shared_arena_methods:
            shared_indices = set(method.shared_index.values())
            per_method: dict[_DeviceKey, int] = {}
            for index, device in enumerate(method.arena_devices):
                if index == 0 or index in shared_indices:
                    continue
                per_method[device] = per_method.get(device, 0) + 1
            for device, count in per_method.items():
                width[device] = max(width.get(device, 0), count)

        order = sorted(width.keys() | shared_devices, key=_device_order_key)
        base: dict[_DeviceKey, int] = {}
        shared_id: dict[_DeviceKey, int] = {}
        # Index 0 is the constants placeholder and is never planned into.
        index = 1
        for device in order:
            base[device] = index
            index += width.get(device, 0)
            if device in shared_devices:
                shared_id[device] = index
                index += 1
        return _SharedArenaLayout(
            base=base,
            shared_id=shared_id,
            num_arenas=index,
        )

    def _arena_mapping(
        self, method: _SharedArenaMethod, layout: _SharedArenaLayout
    ) -> list[int]:
        """The map from one method's arena index to the common one.

        Reads the method and refuses it; writes nothing, so that every method
        can be asked before any method is rewritten.
        """
        if _submodule_arena_ids(method.graph_module):
            raise NotImplementedError(
                "shared_buffer_fqns does not support a method with a "
                "control-flow submodule. apply_algo plans a cond, map or "
                "while_loop submodule by recursing into it, and nothing "
                "renumbers the arena indices that recursion leaves on its "
                "specs, so the cross-method numbering would have to hand every "
                "one of them back unchanged, which it has no way to promise. "
                "Plan such a method without shared_buffer_fqns."
            )
        device_by_shared_index = {
            index: device for device, index in method.shared_index.items()
        }
        first_index: dict[_DeviceKey, int] = {}
        for index, device in enumerate(method.arena_devices):
            if index > 0 and index not in device_by_shared_index:
                first_index.setdefault(device, index)
        mapping = [0] * len(method.arena_devices)
        for index, device in enumerate(method.arena_devices):
            if index == 0:
                continue
            if index in device_by_shared_index:
                mapping[index] = layout.shared_id[device_by_shared_index[index]]
            else:
                mapping[index] = layout.base[device] + (index - first_index[device])

        for node, spec in self._specs_to_renumber(method):
            internal_assert(
                0 < spec.mem_id < len(mapping),
                f"the spec on '{node.name}' sits on arena {spec.mem_id}, which "
                f"is not one of the {len(mapping) - 1} arenas this method "
                "planned (arena 0 is the constants pool and is not planned "
                "into); there is no entry in the common numbering to move it to",
            )
        return mapping

    def _specs_to_renumber(
        self, method: _SharedArenaMethod
    ) -> list[tuple[Node, TensorSpec]]:
        """The specs of one method that the common numbering has to move.

        Everything the algorithm placed, which is every planned spec except the
        declared buffers: those are given their arena directly, from the layout
        rather than from the mapping.
        """
        declared_spec_ids = {id(spec) for spec in method.declared_specs.values()}
        return [
            (node, spec)
            for node, spec in _iter_unique_specs(method.graph_module)
            if spec.mem_id is not None and id(spec) not in declared_spec_ids
        ]

    def _apply_arena_mapping(
        self, method: _SharedArenaMethod, mapping: list[int]
    ) -> None:
        """Move one method's specs into the common numbering.

        Separate from _arena_mapping so that every method's numbering can be
        checked before any method's specs are rewritten: a refusal on the last
        method must not leave the ones before it half-moved.
        """
        for _, spec in self._specs_to_renumber(method):
            spec.mem_id = mapping[spec.mem_id]

    def _rewrite_arena_meta(
        self,
        method: _SharedArenaMethod,
        layout: _SharedArenaLayout,
        mapping: list[int],
        shared_arena_sizes: dict[_DeviceKey, int],
    ) -> None:
        """Restate one method's arena sizes and devices in the common numbering.

        ``mapping`` is what _arena_mapping built for this method and
        _apply_arena_mapping has already moved its specs by; this brings the
        metadata the emitter reads onto the same numbering.
        """
        sizes = method.graph_module.meta["non_const_buffer_sizes"]
        new_sizes = [0] * layout.num_arenas
        new_sizes[0] = sizes[0]
        for index in range(1, len(sizes)):
            new_sizes[mapping[index]] = sizes[index]
        # Only the shared arenas this method declared a buffer on carry their
        # size; the rest of the numbering is held open at zero.
        for device in method.shared_index:
            new_sizes[layout.shared_id[device]] = shared_arena_sizes[device]
        # In place: to_executorch already shallow-copied this meta dict into the
        # exported program, so rebinding the slot would leave the emitter reading
        # the pre-renumbering sizes while the specs carry the new indices.
        sizes[:] = new_sizes

        # Only the arenas this method actually planned carry their device; the
        # padding that holds the numbering open stays CPU. Tagging an empty
        # padded slot for a device would make the method demand that device's
        # allocator to load: DeviceMemoryBuffer::create looks the allocator up
        # before it looks at the size.
        arena_devices = [_CPU_KEY] * layout.num_arenas
        for index, device in enumerate(method.arena_devices):
            if index > 0:
                arena_devices[mapping[index]] = device
        _rebuild_device_entries(method.graph_module, arena_devices)

    def _assign_dedicated_shared_arenas(self) -> None:
        """Give the declared shared buffers an arena of their own per device.

        Runs once, after every method has been planned, because index *i* must
        denote the same memory in every method even when the methods differ in
        which devices, custom pools, or shared buffers they use.
        """
        methods = self.state.shared_arena_methods
        if not methods:
            return

        geometry = self._shared_buffer_geometry()
        # Both warnings after _shared_buffer_geometry, so that a program
        # refused over the buffers it declares is not advised about those
        # buffers first. The numbering below can still refuse after a warning.
        self._warn_about_read_only_methods()
        self._warn_about_undeclared_mutable_buffers()
        layout = self._build_arena_layout({g.device for g in geometry.values()})

        offset_by_fqn: dict[str, int] = {}
        obj_id_by_fqn: dict[str, int] = {}
        shared_arena_sizes: dict[_DeviceKey, int] = {}
        for device in layout.shared_id:
            offset = 0
            for obj_id, fqn in enumerate(
                sorted(f for f, g in geometry.items() if g.device == device)
            ):
                offset_by_fqn[fqn] = offset
                obj_id_by_fqn[fqn] = obj_id
                offset = calculate_aligned_num_bytes(
                    offset + geometry[fqn].allocated_memory, self.alignment
                )
            shared_arena_sizes[device] = offset

        # Every refusal in the numbering is a function of the records alone, so
        # all of them are asked before any method is rewritten: a method refused
        # last must not leave the methods before it in the common numbering and
        # the rest in their own.
        mappings = [self._arena_mapping(method, layout) for method in methods]

        for method, mapping in zip(methods, mappings):
            self._apply_arena_mapping(method, mapping)
            for fqn, spec in method.declared_specs.items():
                spec.mem_id = layout.shared_id[geometry[fqn].device]
                spec.mem_offset = offset_by_fqn[fqn]
                # Usually a restatement of the id the per-method pass wrote:
                # both number the declared buffers from 0 in fqn order, so the
                # two differ only for a method that declares a subset of one
                # device's buffers. Written anyway because the id has to mean
                # the same object in every method.
                spec.mem_obj_id = obj_id_by_fqn[fqn]
            self._rewrite_arena_meta(method, layout, mapping, shared_arena_sizes)

    def reset_multimethod_state(self) -> None:
        """Drop everything the pass recorded while planning one program.

        Everything in ``_MemoryPlanningState`` describes one program, and both
        sharing paths fill it from ``run``. ``run_multimethod`` consumes them,
        and only the dedicated-arena path clears them afterwards, so records
        outlive their program whenever a per-method refusal raises from ``run``
        itself and, on the legacy path, even when nothing raised at all. A next
        program planned alongside those records is held to their geometry and
        to their arena widths, either of which can move its shared arena, so
        to_executorch calls this before it plans anything.

        The whole state object is rebound, so a field added to
        ``_MemoryPlanningState`` is covered here without a second edit.
        """
        self.state = _MemoryPlanningState()

    def run_multimethod(self):
        """Resolve any memory planning done across entry points, called after run is called on all entry points."""
        if self.shared_buffer_fqns is not None:
            try:
                self._assign_dedicated_shared_arenas()
            finally:
                # A pass instance may be handed to to_executorch more than
                # once, and the records are per program. In a finally because
                # the cross-method checks all raise from inside this call; a
                # refusal raised earlier, from run(), never reaches here, which
                # is what to_executorch's own reset call covers.
                self.reset_multimethod_state()
            return

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
