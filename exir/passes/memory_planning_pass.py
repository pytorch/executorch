# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import operator
import warnings
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.utils._pytree as pytree
from executorch.exir._warnings import deprecated
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.error import internal_assert
from executorch.exir.inplace_aliasing import verify_inplace_result_aliases
from executorch.exir.memory import alloc, view
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
from executorch.exir.passes.propagate_device_config import (
    _get_target_device_from_compile_specs,
    _METADATA_READERS,
)
from executorch.exir.passes.replace_view_copy_with_view_pass import _ViewSpec
from executorch.exir.schema import DeviceType, NonConstBufferDevice
from executorch.exir.tensor import ALIGNMENT, TensorSpec
from torch import fx
from torch.export.exported_program import ExportGraphSignature, OutputKind
from torch.fx import Node


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


def _refuse_planned_buffer(fqn: str) -> None:
    raise ValueError(
        f"Cannot share mutable buffers: buffer {fqn} is already planned in memory "
        "of its own, for example in device memory by skip_h2d_for_constants"
    )


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
            if fqn in state.planned_buffers:
                _refuse_planned_buffer(fqn)
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
                if fqn in state.mutable_buffers:
                    _refuse_planned_buffer(fqn)
                state.planned_buffers.add(fqn)
                continue
            if fqn not in state.maybe_mutable_buffers.keys():
                state.maybe_mutable_buffers[fqn] = set()
            state.maybe_mutable_buffers[fqn].add(spec)


def _check_shared_layouts(fqn: str, specs: List[TensorSpec]) -> None:
    layouts = {(tuple(spec.shape), spec.dtype, tuple(spec.dim_order)) for spec in specs}
    if len(layouts) > 1:
        raise ValueError(
            f"Cannot share mutable buffer {fqn}: methods give it different "
            "shapes, dtypes or dim orders: "
            + ", ".join(
                sorted(
                    f"{dtype} {list(shape)} dim order {list(dim_order)}"
                    for shape, dtype, dim_order in layouts
                )
            )
        )


def _planned_specs(gm: torch.fx.GraphModule) -> List[TensorSpec]:
    """Every distinct TensorSpec read or written by a node of gm or of its
    control flow submodules."""
    specs: Dict[int, TensorSpec] = {}
    for subgm in gm.modules():
        if not isinstance(subgm, torch.fx.GraphModule):
            continue
        for node in subgm.graph.nodes:
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
                specs[id(spec)] = spec
    return list(specs.values())


def _check_default_mem_ids(gm: torch.fx.GraphModule, device_planning: bool) -> None:
    # Device-aware planning gives each device its own arena, which
    # run_multimethod moves past the shared one, and leaves arena 1 on the host
    # unless it names a device for it.
    arena_devices = {
        entry.buffer_idx: (entry.device_type, entry.device_index)
        for entry in gm.meta.get("non_const_buffer_device") or []
    }
    if device_planning:
        arena_devices.setdefault(1, (DeviceType.CPU, 0))
    for spec in _planned_specs(gm):
        mem_id = getattr(spec, "mem_id", None)
        if mem_id is None or (mem_id == 1 and mem_id not in arena_devices):
            continue
        arena_device = arena_devices.get(mem_id)
        if arena_device != (spec.device, spec.device_index):
            raise ValueError(
                "Cannot share mutable buffers if all other tensors are not on the "
                "default mem_id of 1 or in the arena of their device: a "
                f"{_format_devices({(spec.device, spec.device_index)})} tensor "
                f"of shape {list(spec.shape)} is in arena {mem_id}"
                + (
                    f", which is on {_format_devices({arena_device})}"
                    if arena_device
                    else ""
                )
            )


def _format_devices(devices: Set[Tuple[DeviceType, int]]) -> str:
    return ", ".join(f"{t.name.lower()}:{i}" for t, i in sorted(devices))


def _copy_source(copy: Node) -> Optional[Node]:
    return copy.args[0] if copy.args else copy.kwargs.get("self")


def _storage_root(spec: Any) -> Any:
    # A _ViewSpec cannot view another one: it reads the fields of its base directly.
    return spec._base if isinstance(spec, _ViewSpec) else spec


def _misuse(
    state: "_MemoryPlanningState",
    device: Tuple[DeviceType, int],
    graph_module: torch.fx.GraphModule,
    holder: Node,
    user: Node,
) -> Optional[str]:
    """Why user may not use holder, which holds a shared buffer planned in
    device memory, or None if it may."""
    if user.op == "output":
        # Returning it would hand the caller device memory.
        if all(
            index in state.mutation_outputs[id(graph_module)]
            for index, output in enumerate(user.args[0])
            if output is holder
        ):
            return None
        return "returns it"
    elif user.target in _METADATA_READERS:
        return None
    elif user.target == torch.ops.et_copy._d2h_copy.out:
        if _copy_source(user) is holder and user.kwargs.get("out") is not holder:
            return None
    elif user.target == torch.ops.et_copy._h2d_copy.out:
        if user.kwargs.get("out") is holder and _copy_source(user) is not holder:
            return None
    elif user.target == executorch_call_delegate:
        # Its outputs can be the buffer itself, so they do not tell its device.
        target = _get_target_device_from_compile_specs(
            getattr(graph_module, user.args[0].target)
        )
        if target == device:
            return None
        if target is None:
            return "is a delegate without a target_device compile spec"
        return f"is a delegate on {_format_devices({target})}"
    return "uses it on the host"


def _host_use(
    state: "_MemoryPlanningState",
    device: Tuple[DeviceType, int],
    graph_module: torch.fx.GraphModule,
    holder: Node,
    root: TensorSpec,
) -> Optional[Tuple[Node, str]]:
    """The first user of holder that may not use the shared buffer root, and
    why. Follows views of the buffer and the delegate result that is it."""
    for user in holder.users:
        if user.target == view:
            # Emitted at the place of what it views, so it must view the buffer.
            found = (
                _host_use(state, device, graph_module, user, root)
                if _storage_root(user.meta["spec"]) is root
                else (user, "is a view planned apart from it")
            )
        elif user.target == operator.getitem:
            # One result of a delegate. Only the one that is the buffer holds it.
            found = (
                _host_use(state, device, graph_module, user, root)
                if _storage_root(holder.meta["spec"][user.args[1]]) is root
                else None
            )
        else:
            reason = _misuse(state, device, graph_module, holder, user)
            found = None if reason is None else (user, reason)
        if found is not None:
            return found
    return None


def _refuse_host_access(
    state: "_MemoryPlanningState",
    spec_fqns: Dict[int, str],
    device: Tuple[DeviceType, int],
) -> None:
    """Refuses a host use of a shared buffer planned in device memory, since
    planning a buffer in device memory inserts no copies."""
    for graph_module in state.graph_modules:
        for node in graph_module.graph.nodes:
            # Besides the placeholder, a copy that writes the buffer and a
            # delegate that updates it in place hold its memory. A view or one
            # result of a delegate is checked with what it comes from.
            if node.target in (view, operator.getitem):
                continue
            for leaf in pytree.tree_leaves(node.meta.get("spec")):
                root = _storage_root(leaf)
                fqn = spec_fqns.get(id(root))
                found = (
                    None
                    if fqn is None
                    else _host_use(state, device, graph_module, node, root)
                )
                if found is not None:
                    user, reason = found
                    target = (
                        user.target
                        if isinstance(user.target, (str, torch._ops.OpOverload))
                        else getattr(user.target, "__name__", user.target)
                    )
                    raise ValueError(
                        f"Cannot share mutable buffer {fqn} in "
                        f"{_format_devices({device})} memory: {user.name} "
                        f"({target}) {reason}. Only a delegate on that device, "
                        "et_copy._d2h_copy reading it, et_copy._h2d_copy writing "
                        "it, an op that reads only its sizes, strides, storage "
                        "offset or element count, and the write-back of its new "
                        "value can use it."
                    )


def _refuse_buffer_copy_specs(
    gm: torch.fx.GraphModule, gs: ExportGraphSignature
) -> None:
    """Refuses a copy into a buffer whose spec differs from the buffer's.

    _set_alloc_node_spec gives the buffer the spec of the copy, so views made
    before planning would still point at the old one and be planned elsewhere.
    It does that only for an op with one out argument.
    """
    for node in gm.graph.nodes:
        if not _is_out_var_node(node):
            continue
        out_args = get_out_args_from_opoverload(node.target)
        out = node.kwargs.get(out_args[0]) if len(out_args) == 1 else None
        fqn = _is_buffer(out, gs)[1] if isinstance(out, Node) else None
        if fqn is not None and node.meta.get("spec") is not out.meta.get("spec"):
            raise ValueError(
                f"Cannot share mutable buffer {fqn}: {node.name} writes it with "
                "a TensorSpec of its own. Give the copy the spec of the buffer."
            )


def _insert_shared_arena(
    gm: torch.fx.GraphModule,
    size: int,
    device: Tuple[DeviceType, int],
    marked: bool,
) -> None:
    """Make mem_id 2 of gm the shared arena, on device, and mark it if gm uses it.

    Every method that uses a shared buffer must find the shared arena at the
    same mem_id, so the device arenas that device-aware planning put at mem_id
    2 and above move up by one. Loading a method that marks the shared arena
    initializes the shared state.
    """
    specs = _planned_specs(gm)
    # Read every id first, since a view spec forwards mem_id to its base.
    mem_ids = [spec.mem_id for spec in specs]
    for spec, mem_id in zip(specs, mem_ids):
        if mem_id is not None and mem_id >= 2:
            spec.mem_id = mem_id + 1
    gm.meta["non_const_buffer_sizes"].insert(2, size)
    if marked:
        gm.meta["shared_non_const_buffers"] = [2]
    entries = [
        (
            replace(entry, buffer_idx=entry.buffer_idx + 1)
            if entry.buffer_idx >= 2
            else entry
        )
        for entry in gm.meta.get("non_const_buffer_device") or []
    ]
    if device[0] != DeviceType.CPU:
        entries.append(
            NonConstBufferDevice(
                buffer_idx=2, device_type=device[0], device_index=device[1]
            )
        )
    if entries:
        gm.meta["non_const_buffer_device"] = sorted(
            entries, key=lambda entry: entry.buffer_idx
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


@dataclass
class _MemoryPlanningState:
    mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    maybe_mutable_buffers: Dict[str, Set[TensorSpec]] = field(default_factory=dict)
    # Buffers some method already planned, such as a constant kept on a device.
    planned_buffers: Set[str] = field(default_factory=set)
    graph_modules: List[torch.fx.GraphModule] = field(default_factory=list)
    # Output positions of each graph module that write back a mutated buffer.
    mutation_outputs: Dict[int, Set[int]] = field(default_factory=dict)


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
    ) -> None:
        r"""
        alloc_graph_input/alloc_graph_output will have 4 different combinations
        to control if the memory planning algorithm need allocate memory for
        the graph input/output. The default behavior is the algorithm will allocate
        memory for both graph input and output.
        """
        if memory_planning_algo is None:
            memory_planning_algo = MemoryPlanningAlgorithmSuite()
        if share_mutable_buffers and not alloc_mutable_buffers:
            raise ValueError(
                "share_mutable_buffers is only meaningful when alloc_mutable_buffers is True"
            )
        self.memory_planning_algo: Callable[..., List[int]] = memory_planning_algo
        self.allow_lifetime_and_storage_overlap = allow_lifetime_and_storage_overlap
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.alloc_mutable_buffers = alloc_mutable_buffers
        self.share_mutable_buffers = share_mutable_buffers
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
        verify_inplace_result_aliases(graph_module)
        if self.share_mutable_buffers and graph_signature is not None:
            _refuse_buffer_copy_specs(graph_module, graph_signature)
        self._set_alloc_node_spec(graph_module)
        # TODO(shunting) if people have concern of adding a field to GraphModule
        # directly, we should define a GraphModule subclass that we can add our
        # customized fields. Using the graph_module object to convey information across
        # passes/stages is quite natural and avoid yet another 'context' data structure
        # to do the job.

        _ = apply_algo(
            self.memory_planning_algo,
            graph_module,
            self.alignment,
            graph_signature,
            self.alloc_graph_input,
            self.alloc_graph_output,
            # If mutable buffers are shared, then do not allocate them in the
            # main memory planning algo; they are allocated in run_multimethod.
            self.alloc_mutable_buffers and not self.share_mutable_buffers,
            self.enable_non_cpu_memory_planning,
        )

        if self.share_mutable_buffers and graph_signature is not None:
            self.state.graph_modules.append(graph_module)
            self.state.mutation_outputs[id(graph_module)] = {
                index
                for index, spec in enumerate(graph_signature.output_specs)
                if spec.kind == OutputKind.BUFFER_MUTATION
            }
            _check_default_mem_ids(graph_module, self.enable_non_cpu_memory_planning)
            _insert_mutable_buffer_specs(self.state, graph_module, graph_signature)

        # TODO: make the verifier do the work recursively to handle
        # control flow
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            # If mutable buffers are shared, they are allocated after the
            # main memory planning algo in run_multimethod, and should be
            # skipped in the Verifier.
            self.alloc_mutable_buffers and not self.share_mutable_buffers,
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

    def run_multimethod(self):
        """Resolve any memory planning done across entry points, called after run is called on all entry points."""
        try:
            self._share_mutable_buffers()
        finally:
            # The pass may plan another program next.
            self.reset()

    def reset(self) -> None:
        """Forgets the graphs this pass has planned, such as those of an
        export that failed part way, so the next program starts empty."""
        self.state = _MemoryPlanningState()

    def _share_mutable_buffers(self) -> None:
        if self.share_mutable_buffers:
            arena: int = 0
            shared_specs: List[TensorSpec] = []
            devices: Set[Tuple[DeviceType, int]] = set()
            spec_fqns: Dict[int, str] = {}

            # Every spec that shares an fqn is the same tensor! So we give it the same offset
            # anywhere it appears.
            for fqn, specs_set in self.state.mutable_buffers.items():
                specs = list(specs_set)
                # If the same buffer appears in mutable and maybe mutable then we know it is in fact mutable.
                if fqn in self.state.maybe_mutable_buffers.keys():
                    specs.extend(self.state.maybe_mutable_buffers[fqn])
                _check_shared_layouts(fqn, specs)
                fqn_devices = {(spec.device, spec.device_index) for spec in specs}
                if self.enable_non_cpu_memory_planning and len(fqn_devices) > 1:
                    raise ValueError(
                        f"Cannot share mutable buffer {fqn}: methods place it on "
                        f"{_format_devices(fqn_devices)}"
                    )
                devices |= fqn_devices
                for spec in specs:
                    spec.realign(self.alignment)
                    # State is persistent, so the memory never overlaps.
                    spec.mem_offset = arena
                # They are the same tensor, so just bump off the first.
                arena += specs[0].allocated_memory
                shared_specs.extend(specs)
                spec_fqns.update((id(spec), fqn) for spec in specs)

            device = (DeviceType.CPU, 0)
            if self.enable_non_cpu_memory_planning and devices:
                if len(devices) > 1:
                    raise ValueError(
                        "Cannot share mutable buffers on more than one device, "
                        f"found {_format_devices(devices)}"
                    )
                (device,) = devices
            if device[0] != DeviceType.CPU:
                _refuse_host_access(self.state, spec_fqns, device)

            for graph_module in self.state.graph_modules:
                uses_shared_buffers = any(
                    id(node.meta.get("spec")) in spec_fqns
                    for node in graph_module.graph.nodes
                    if node.op == "placeholder"
                )
                # On the host every method keeps arena 2, the layout older
                # runtimes know, but one without shared state needs no memory
                # there: a runtime that gives every method one arena 2 sizes it
                # for the largest method. A method that uses no shared buffer
                # needs no device arena.
                if uses_shared_buffers or device[0] == DeviceType.CPU:
                    _insert_shared_arena(
                        graph_module,
                        arena if uses_shared_buffers else 0,
                        device,
                        uses_shared_buffers,
                    )
            for spec in shared_specs:
                spec.mem_id = 2
