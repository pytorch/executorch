# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

        _ = apply_algo(
            self.memory_planning_algo,
            graph_module,
            self.alignment,
            graph_signature,
            self.alloc_graph_input,
            self.alloc_graph_output,
            # If we are sharing the mutable buffers then do not allocate them in
            # memory planning algo, instead collect all of the specs over all the entry
            # points and then allocate them directly in the run_multimethod name call
            self.alloc_mutable_buffers and not self.share_mutable_buffers,
        )

        if self.share_mutable_buffers and graph_signature is not None:
            self.state.graph_modules.append(graph_module)
            _check_default_mem_ids(graph_module)
            _insert_mutable_buffer_specs(self.state, graph_module, graph_signature)

        # TODO: make the verifier do the work recursively to handle
        # control flow
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            self.alloc_mutable_buffers,
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
