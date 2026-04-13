#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Program Builder - converts an ExportedProgram to an MLXGraph.

This module is responsible for:
1. Walking the FX graph from an ExportedProgram
2. Converting each node to the corresponding MLX op
3. Managing tensor and value slots
4. Building the final MLXGraph dataclass for serialization

Op handlers are registered in ops.py.
Pattern handlers are registered in patterns.py.
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.backends.mlx._logging import logger
from executorch.backends.mlx.builder.op_helpers import torch_dtype_to_scalar_type
from executorch.backends.mlx.builder.op_registry import (
    Handler,
    PatternHandler,
    REGISTRY,
)
from executorch.backends.mlx.builder.pattern_matcher import PatternMatcher
from executorch.backends.mlx.builder.slot_manager import (
    IdSpace,
    IdType,
    Slot,
    SlotManager,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    FloatOrVid,
    IdCopyNode,
    Instruction,
    InstructionChain,
    IntOrVid,
    IntOrVidOrTid,
    MLXGraph,
    NamedSlot,
    OpNodeUnion,
    ShapeDim,
    SlotType,
    SlotVariant,
    TensorMeta,
    Tid,
    Vid,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node
from torch.utils import _pytree as pytree


def _check_dtype(node: Node) -> Optional[str]:
    """
    Check if a node has a supported dtype.

    Args:
        node: The FX node to check

    Returns:
        None if the node's dtype is supported, otherwise an error message string
    """
    fake_val = node.meta.get("val", None)
    if fake_val is not None and hasattr(fake_val, "dtype"):
        try:
            torch_dtype_to_scalar_type(fake_val.dtype)
        except ValueError:
            return f"has unsupported dtype: {fake_val.dtype}"
    return None


def _check_input_dtypes(node: Node) -> Optional[str]:
    """
    Check if all input tensors to a node have supported dtypes.

    Args:
        node: The FX node to check

    Returns:
        None if all input dtypes are supported, otherwise an error message string
        describing which input (arg position or kwarg name) has an unsupported dtype
    """
    # Check positional args
    for i, arg in enumerate(node.args):
        if isinstance(arg, Node):
            dtype_error = _check_dtype(arg)
            if dtype_error is not None:
                return f"arg[{i}] ({arg.name}) {dtype_error}"

    # Check kwargs
    for kwarg_name, kwarg_val in node.kwargs.items():
        if isinstance(kwarg_val, Node):
            dtype_error = _check_dtype(kwarg_val)
            if dtype_error is not None:
                return f"kwarg '{kwarg_name}' ({kwarg_val.name}) {dtype_error}"

    return None


@dataclass
class NodeInfo:
    handled: bool = False
    handler: Optional[Union[Handler, PatternHandler]] = None
    supported: bool = False
    unsupported_reason: Optional[str] = None
    name: Optional[str] = None
    remaining_reads: int = 0


class MLXProgramBuilder:
    """
    Builds an MLXGraph from an ExportedProgram.

    Args:
        ep: The ExportedProgram to build from
    """

    def __init__(self, ep: ExportedProgram, named_data_key_prefix: str = ""):
        self.ep: ExportedProgram = ep
        self._chains: List[List[Instruction]] = [[]]  # chain 0 = main
        self._current_chain: int = 0
        self.init_chain_idx: int = -1
        self.extra_constants: Dict[str, torch.Tensor] = {}
        self.slot_manager = SlotManager()
        self.node_info: DefaultDict[Node, NodeInfo] = defaultdict(NodeInfo)
        self._mlx_graph: Optional[MLXGraph] = None
        # Map from SymInt symbol names (e.g., "s77") to the FX Node that produces them.
        # This is used to resolve symbolic tensor dimensions to Vid references.
        self._symint_to_node: Dict[str, Node] = {}
        # Maps for remapping local slot indices to global Tid/Vid indices during build
        self._tid_slot_map: List[Tuple[Tid, Slot]] = []
        self._vid_slot_map: List[Tuple[Vid, Slot]] = []
        # Prefix for named_data_store keys and named_slots to avoid collisions
        # in multi-method programs where different methods may have lifted tensor
        # constants with the same auto-generated name.
        self._named_data_key_prefix: str = named_data_key_prefix
        # Unprefixed canonical-name → Slot for constants, populated by _build_io_maps().
        # Used by get_named_data_store() to look up tensors without prefix interference.
        self._constant_name_to_slot: Dict[str, Slot] = {}

    def _prefix_key(self, name: str) -> str:
        """Apply the named-data key prefix for the .pte namespace.

        This is the single point where canonical (unprefixed) names are
        transformed into the external keys used in the .pte's ``named_data``
        section and the FlatBuffer ``named_slots`` field.
        """
        if self._named_data_key_prefix:
            return f"{self._named_data_key_prefix}/{name}"
        return name

    def emit(self, op: OpNodeUnion) -> None:
        self._chains[self._current_chain].append(Instruction(op=op))

    def emit_init(self, op: OpNodeUnion) -> None:
        if self.init_chain_idx == -1:
            self.init_chain_idx = len(self._chains)
            self._chains.append([])
        self._chains[self.init_chain_idx].append(Instruction(op=op))

    @contextmanager
    def new_chain(self):
        """Context manager that creates a new instruction chain and redirects emit() to it.

        Usage:
            with P.new_chain() as chain_idx:
                P.emit(MulNode(...))   # goes to the new chain
            # P.emit() goes back to the previous chain
        """
        chain_idx = len(self._chains)
        self._chains.append([])
        prev_chain = self._current_chain
        self._current_chain = chain_idx
        try:
            yield chain_idx
        finally:
            self._current_chain = prev_chain

    def args(self, node: Node) -> Tuple[Any, ...]:
        return self.slot_map(node.args)

    def kwargs(self, node: Node) -> Dict[str, Any]:
        return self.slot_map(node.kwargs)

    def slot_map(self, tree):
        leaves, spec = pytree.tree_flatten(tree)
        new_leaves = []
        for a in leaves:
            if isinstance(a, Node):
                # Use make_or_get_slots which handles both single and multi-output nodes.
                # For single-output nodes, returns a 1-tuple; for multi-output, returns n-tuple.
                # We unwrap single-element tuples for convenience.
                slots = self.make_or_get_slots(a)
                if len(slots) == 1:
                    new_leaves.append(slots[0])
                else:
                    new_leaves.append(slots)
            else:
                new_leaves.append(a)

        for a in new_leaves:
            if isinstance(a, Slot):
                assert self.slot_manager.is_alive(
                    a
                ), f"Slot {a} is not alive; it was either already freed or never created"

        return pytree.tree_unflatten(new_leaves, spec)

    def make_or_get_slots(
        self, node: Node, id_space: IdSpace = IdSpace.Temp
    ) -> Tuple[Slot, ...]:
        """Get or create slots for a multi-output node. Always returns a tuple."""
        return self.slot_manager.make_or_get_slots(node, id_space)

    def make_or_get_slot(self, node: Node, id_space: IdSpace = IdSpace.Temp) -> Slot:
        """Get or create a slot for a single-output node. Returns a single Slot."""
        return self.slot_manager.make_or_get_slot(node, id_space)

    def set_slot(self, node: Node, slot: Slot):
        self.slot_manager.set_slot(node, slot)

    def make_tmp_slot(self) -> Tuple[str, Slot]:
        """Create a temporary tensor slot."""
        return self.slot_manager.make_tmp_slot()

    def make_tmp_value_slot(self) -> Tuple[str, Slot]:
        """Create a temporary value (SymInt) slot."""
        return self.slot_manager.make_tmp_value_slot()

    def make_or_get_constant(self, name: str, tensor: torch.Tensor) -> Slot:
        """
        Creates an extra constant outside of the ExportedProgram state_dict.
        Ops can use this to add constants during build that do not exist in the
        ExportedProgram state_dict, e.g., doing naive packing of quantized ops.
        """
        assert name not in self.ep.state_dict
        assert name not in self.ep.constants

        if name in self.extra_constants:
            # During fake tensor tracing, we can't use torch.equal
            # Just assume tensors with same name are the same
            slot = self.slot_manager.get_slot(name)
            assert slot is not None
            return slot

        slot = self.slot_manager.make_constant_slot(name)
        self.extra_constants[name] = tensor
        return slot

    def get_placeholder_target_and_tensor(self, node: Node) -> Tuple[str, torch.Tensor]:
        assert node.op == "placeholder"
        placeholder_name = node.name

        sig = self.ep.graph_signature
        sd = self.ep.state_dict
        consts = self.ep.constants

        for ispec in sig.input_specs:
            if ispec.arg.name != placeholder_name:
                continue
            target = ispec.target
            if target is None:
                continue
            if target in sd:
                return (target, sd[target])
            if target in consts:
                return (target, consts[target])

        raise KeyError(f"Unable to resolve placeholder {placeholder_name}")

    def slot_to_tid(self, slot: Slot) -> Tid:
        """Convert a tensor Slot to a Tid, recording it for later remapping."""
        assert slot.id_type == IdType.Tensor
        # Use local slot.idx as placeholder - will be remapped to global idx in build()
        tid = Tid(idx=slot.idx)
        self._tid_slot_map.append((tid, slot))
        return tid

    def slot_to_vid(self, slot: Slot) -> Vid:
        """Convert a value Slot to a Vid, recording it for later remapping."""
        assert slot.id_type != IdType.Tensor
        vid = Vid(idx=slot.idx)
        self._vid_slot_map.append((vid, slot))
        return vid

    def to_int_or_vid(self, v: Union[int, Slot]) -> IntOrVid:
        if isinstance(v, Slot):
            return IntOrVid.from_vid(self.slot_to_vid(v))
        return IntOrVid.from_literal(int(v))

    def to_float_or_vid(self, v: Union[float, int, Slot]) -> FloatOrVid:
        if isinstance(v, Slot):
            return FloatOrVid.from_vid(self.slot_to_vid(v))
        return FloatOrVid.from_literal(float(v))

    def to_int_or_vid_or_tid(self, v: Union[int, Slot]) -> IntOrVidOrTid:
        if isinstance(v, Slot):
            if v.id_type == IdType.Tensor:
                return IntOrVidOrTid.from_tid(self.slot_to_tid(v))
            return IntOrVidOrTid.from_vid(self.slot_to_vid(v))
        return IntOrVidOrTid.from_literal(int(v))

    def _mark_read(self, node: Node):
        assert self.node_info[node].handled, f"Node {node} is not handled"
        assert (
            self.node_info[node].remaining_reads > 0
        ), f"Reading node {node}, but it has no remaining reads"
        self.node_info[node].remaining_reads -= 1

        if self.node_info[node].remaining_reads == 0:
            slot = self.slot_manager.get_slot(node)
            if slot is None:
                return
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                if s.id_space != IdSpace.Temp:
                    continue
                if s.id_type == IdType.Tensor:
                    self.slot_manager.tid_managers[IdSpace.Temp].return_id(s.idx)
                else:
                    self.slot_manager.vid_managers[IdSpace.Temp].return_id(s.idx)

    def _mark_node_handled(self, node: Node, *, handler: Optional[Handler] = None):
        if self.node_info[node].handled:
            return
        self.node_info[node].handled = True
        self.node_info[node].remaining_reads = len(node.users)
        self.node_info[node].handler = handler

        if handler == PatternHandler.deferred_handler:
            return

        def mark_read(n: Node):
            flat_args, spec = pytree.tree_flatten((n.args, n.kwargs))
            seen = set()
            for a in flat_args:
                if isinstance(a, Node):
                    if a not in seen:
                        self._mark_read(a)
                        seen.add(a)

        if isinstance(handler, PatternHandler):
            for n in handler.body:
                mark_read(n)
        mark_read(node)

    def _mark_node_supported(self, node: Node, *, handler: Optional[Handler] = None):
        self.node_info[node].supported = True
        self._mark_node_handled(node, handler=handler)

    def _mark_node_unsupported(self, node: Node, reason: str):
        self.node_info[node].supported = False
        self.node_info[node].unsupported_reason = reason
        self._mark_node_handled(node)

    def _is_handled(self, node: Node) -> bool:
        return self.node_info[node].handled

    def _mark_supported(
        self, nodes: Union[List[Node], Node], *, handler: Optional[Handler] = None
    ) -> None:
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            self._mark_node_supported(node, handler=handler)

    def _mark_unsupported(self, nodes: Union[List[Node], Node], reason: str) -> None:
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            self._mark_node_unsupported(node, reason)

    def _make_io_slots(self):  # noqa: C901
        from torch.export.graph_signature import (
            InputKind,
            OutputKind,
            SymIntArgument,
            TensorArgument,
        )

        output_kind_targets = defaultdict(set)
        constant_tensors = []
        user_inputs = []
        user_outputs = []
        mutable_buffers = []

        for ospec in self.ep.graph_signature.output_specs:
            kind = ospec.kind
            arg = ospec.arg
            name = arg.name
            target = ospec.target
            if target is not None:
                output_kind_targets[kind].add(target)
            if kind in (OutputKind.USER_OUTPUT, OutputKind.USER_INPUT_MUTATION):
                user_outputs.append(name)

        for ispec in self.ep.graph_signature.input_specs:
            kind = ispec.kind
            arg = ispec.arg
            name = arg.name
            target = ispec.target

            if isinstance(arg, TensorArgument):
                if kind == InputKind.PARAMETER:
                    # Parameters are treated as constants (not mutated)
                    constant_tensors.append(name)
                elif kind == InputKind.BUFFER:
                    if target in output_kind_targets[OutputKind.BUFFER_MUTATION]:
                        mutable_buffers.append(name)
                    else:
                        # Non-mutated buffers (like lifted tensor constants) are constants
                        constant_tensors.append(name)
                elif kind == InputKind.USER_INPUT:
                    user_inputs.append(name)
                elif kind == InputKind.CONSTANT_TENSOR:
                    constant_tensors.append(name)
                else:
                    raise NotImplementedError(
                        f"Support for input {arg} is not implemented"
                    )
            elif isinstance(arg, SymIntArgument):
                if kind == InputKind.USER_INPUT:
                    user_inputs.append(name)
                else:
                    raise NotImplementedError(
                        f"Support for input {arg} is not implemented"
                    )
            else:
                raise NotImplementedError(f"Support for input {arg} is not implemented")

        for node in self.ep.graph.nodes:
            if node.op == "placeholder":
                if node.users == {}:
                    continue
                if node.name in constant_tensors:
                    self.make_or_get_slot(node, id_space=IdSpace.Constant)
                elif node.name in user_inputs:
                    val = node.meta.get("val", None)
                    if isinstance(val, torch.Tensor) and not val.is_contiguous():
                        raise ValueError(
                            f"MLX backend requires contiguous input tensors, "
                            f"but input '{node.name}' has non-contiguous strides. "
                            f"shape={list(val.shape)}, stride={list(val.stride())}. "
                            f"Ensure example inputs passed to torch.export.export() "
                            f"are contiguous (call .contiguous() on them)."
                        )
                    self.make_or_get_slot(node, id_space=IdSpace.Input)
                elif node.name in mutable_buffers:
                    self.make_or_get_slot(node, id_space=IdSpace.MutableBuffer)
                else:
                    raise NotImplementedError(
                        f"Support for placeholder {node.name} is not implemented"
                    )
            elif node.op == "output":
                outs, _ = pytree.tree_flatten(node.args)
                for o in outs:
                    if isinstance(o, Node) and o.name in user_outputs:
                        self.make_or_get_slot(o, id_space=IdSpace.Output)

    def _mark_noop(self):
        """Mark noops and dead nodes."""
        dead = set()
        noop_handler = REGISTRY.get_noop_handler()
        if noop_handler is None:
            return

        for n in reversed(self.ep.graph.nodes):
            handler = REGISTRY.get_handler(n)
            if handler == noop_handler:
                dead.add(n)

            if n.op != "output" and all(user in dead for user in n.users):
                self.node_info[n].handler = noop_handler
                dead.add(n)

    def _apply_patterns(self) -> None:
        """
        Find and apply pattern handlers to the graph.

        Uses PatternMatcher to discover multi-node patterns and assigns
        handlers to matched nodes. This must run BEFORE _mark_noop so
        pattern body nodes don't get incorrectly marked as dead.
        """
        matcher = PatternMatcher(self.ep, REGISTRY)
        for handler in matcher.find_patterns():
            handler.set_handlers(self)

    def _process_nodes(self) -> None:  # noqa C901
        """
        Common logic for processing all nodes: create slots, match patterns, run handlers.

        This method:
        1. Creates I/O slots for placeholders and outputs
        2. Matches patterns FIRST (so body nodes get handlers and aren't marked dead)
        3. Marks dead/noop nodes
        4. Runs handlers for remaining nodes, marking them supported/unsupported

        The ordering is important: patterns must be matched before noops because
        some pattern body nodes (e.g., update_cache) have no users since they
        mutate in-place, but they're not dead - they're handled by the pattern.
        """
        self._make_io_slots()

        # Apply patterns BEFORE _mark_noop so pattern body nodes don't get
        # incorrectly marked as dead (e.g., update_cache nodes have no users
        # since they mutate in-place, but they're not dead)
        self._apply_patterns()
        self._mark_noop()

        for n in self.ep.graph.nodes:
            if self._is_handled(n):
                continue

            if self.node_info[n].handler is not None:
                handler = self.node_info[n].handler
                handler(self, n)
                self._mark_supported(n, handler=handler)
                continue

            # Check input dtypes before processing node
            unsupported_dtype_msg = _check_input_dtypes(n)
            if unsupported_dtype_msg is not None:
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slots(n)
                self._mark_unsupported(n, unsupported_dtype_msg)
                continue

            if n.op in ("placeholder", "output"):
                dtype_error = _check_dtype(n)
                if dtype_error is not None:
                    self._mark_unsupported(n, f"{n.op} {dtype_error}")
                    continue
                self._mark_supported(n)
                continue

            handler = REGISTRY.get_handler(n)
            if handler is None:
                msg = f"no handler for target={n.target}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slots(n)
                self._mark_unsupported(n, msg)
                continue

            try:
                handler(self, n)
                self._mark_supported(n, handler=handler)
            except Exception as e:
                trace_str = traceback.format_exc()
                msg = f"{handler} failed for {n.target}: {e}.\n{trace_str}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slots(n)
                self._mark_unsupported(n, msg)

    def check_support_only(self) -> None:
        """
        Check which nodes are supported without building the full MLXGraph.

        This method populates node_info with supported/unsupported status for each
        node, but avoids calling _build_mlx_graph() which can corrupt the shape_env
        by evaluating symbolic shapes.

        Use this method for ops_to_not_decompose() and similar queries where you
        only need to know support status, not the full compiled graph.
        """
        self._process_nodes()
        # NOTE: We intentionally skip _verify_build() and _build_mlx_graph() here
        # because _build_mlx_graph() calls int() on tensor shapes which evaluates
        # SymInts and corrupts the shape_env. This method is used for
        # ops_to_not_decompose() where we only need support status.

    def _emit_buffer_mutation_writebacks(self):
        """Emit copy-back instructions for BUFFER_MUTATION outputs.

        When a model mutates a buffer (e.g., via .copy_() or .mul_()),
        torch.export functionalizes it: the new value is a computation result,
        and the output spec marks it as BUFFER_MUTATION with a target buffer.

        This method emits an IdCopyNode for each BUFFER_MUTATION output,
        copying the computation result back to the mutable buffer slot so
        the updated value persists across execution calls.
        """
        from torch.export.graph_signature import InputKind, OutputKind

        # Map buffer target name -> input placeholder name
        target_to_placeholder = {}
        for ispec in self.ep.graph_signature.input_specs:
            if ispec.kind == InputKind.BUFFER and ispec.target is not None:
                target_to_placeholder[ispec.target] = ispec.arg.name

        for ospec in self.ep.graph_signature.output_specs:
            if ospec.kind != OutputKind.BUFFER_MUTATION:
                continue

            result_slot = self.slot_manager.get_slot(ospec.arg.name)
            placeholder_name = target_to_placeholder.get(ospec.target)
            if result_slot is None or placeholder_name is None:
                continue

            buffer_slot = self.slot_manager.get_slot(placeholder_name)
            if buffer_slot is None or buffer_slot.id_space != IdSpace.MutableBuffer:
                continue

            self.emit(
                IdCopyNode(
                    x=self.slot_to_tid(result_slot),
                    out=self.slot_to_tid(buffer_slot),
                )
            )

    def build(self) -> MLXGraph:
        if self._mlx_graph is not None:
            return self._mlx_graph

        self._process_nodes()
        self._emit_buffer_mutation_writebacks()
        self._verify_build()
        self._mlx_graph = self._build_mlx_graph()
        return self._mlx_graph

    def _verify_build(self):
        noop_handler = REGISTRY.get_noop_handler()

        for n, info in self.node_info.items():
            assert info.handled
            assert (
                info.remaining_reads == 0
            ), f"Expected {n} to have no remaining reads, but it has {info.remaining_reads}"
            if n.op == "output":
                assert self.slot_manager.get_slot(n) is None
                continue
            if (
                info.handler in (noop_handler, PatternHandler.deferred_handler)
                or n.users == {}
            ):
                # Deferred body nodes may or may not have slots — this is fine.
                # Pattern handlers absorb nodes into their body and may set
                # slots on them (e.g., GatedDeltaRuleHandler sets getitem[0]'s
                # slot to the ScanNode output). Dead nodes (no users) also
                # skip the slot check.
                pass
            else:
                assert (
                    self.slot_manager.get_slot(n) is not None
                ), f"Expected slot for node {n}"

    def _collect_used_slots(
        self,
    ) -> Tuple[Set[Slot], Dict[IdSpace, int], Dict[IdSpace, int]]:
        """
        Collect all used slots and count tensors/values per IdSpace.

        For constants and temps, only includes those actually referenced by
        instructions. This ensures unused slots are not serialized or counted.

        Returns:
            (used_slots, num_tensors, num_values)
        """
        # Get slots actually referenced by instructions
        instruction_referenced: Set[Slot] = {slot for _, slot in self._tid_slot_map}
        instruction_referenced.update({slot for _, slot in self._vid_slot_map})

        used_slots: Set[Slot] = set()
        for _n, slot in self.slot_manager.name_to_slot.items():
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                # For constants and temps, only include if referenced by instructions
                if s.id_space in (IdSpace.Constant, IdSpace.Temp):
                    if s in instruction_referenced:
                        used_slots.add(s)
                else:
                    # Inputs, outputs, mutable buffers - always include
                    used_slots.add(s)

        num_tensors: Dict[IdSpace, int] = defaultdict(int)
        num_values: Dict[IdSpace, int] = defaultdict(int)
        seen: Set[Slot] = set()
        for s in used_slots:
            if s in seen:
                continue
            seen.add(s)
            if s.id_type == IdType.Tensor:
                num_tensors[s.id_space] += 1
            else:
                num_values[s.id_space] += 1

        return used_slots, num_tensors, num_values

    def _create_slot_mappings(
        self, used_slots: Set[Slot]
    ) -> Tuple[Dict[Slot, int], Dict[Slot, int]]:
        """
        Create slot→Tid and slot→Vid mappings, and remap existing references.

        Returns:
            (slot_to_tid, slot_to_vid)
        """
        id_space_order = {
            IdSpace.Constant: 0,
            IdSpace.Input: 1,
            IdSpace.Output: 2,
            IdSpace.MutableBuffer: 3,
            IdSpace.Temp: 4,
        }

        # Create Tid mapping
        slot_to_tid = sorted(
            [s for s in used_slots if s.id_type == IdType.Tensor],
            key=lambda s: (id_space_order[s.id_space], s.idx),
        )
        slot_to_tid = {s: idx for idx, s in enumerate(slot_to_tid)}

        # Create Vid mapping
        slot_to_vid = sorted(
            [s for s in used_slots if s.id_type != IdType.Tensor],
            key=lambda s: (id_space_order[s.id_space], s.idx),
        )
        slot_to_vid = {s: idx for idx, s in enumerate(slot_to_vid)}

        # Remap all Tid/Vid values in instructions to use global indices
        if hasattr(self, "_tid_slot_map"):
            for tid, slot in self._tid_slot_map:
                if slot in slot_to_tid:
                    tid.idx = slot_to_tid[slot]
                else:
                    logger.warning(f"Slot {slot} not found in slot_to_tid mapping")

        if hasattr(self, "_vid_slot_map"):
            for vid, slot in self._vid_slot_map:
                if slot in slot_to_vid:
                    vid.idx = slot_to_vid[slot]
                else:
                    logger.warning(f"Slot {slot} not found in slot_to_vid mapping")

        return slot_to_tid, slot_to_vid

    def _to_slot_variant(
        self,
        slot: Slot,
        slot_to_tid: Dict[Slot, int],
        slot_to_vid: Dict[Slot, int],
    ) -> SlotVariant:
        """Convert a Slot to a SlotVariant using the provided mappings."""
        if slot.id_type == IdType.Tensor:
            idx = slot_to_tid[slot]
            slot_type = SlotType.TensorSlot
        elif slot.id_type == IdType.SymInt:
            idx = slot_to_vid[slot]
            slot_type = SlotType.IntValueSlot
        elif slot.id_type == IdType.SymBool:
            idx = slot_to_vid[slot]
            slot_type = SlotType.BoolValueSlot
        else:
            raise NotImplementedError(f"Unsupported slot type {slot.id_type}")
        return SlotVariant(idx=idx, slot_type=slot_type)

    def _build_io_maps(
        self,
        used_slots: Set[Slot],
        slot_to_tid: Dict[Slot, int],
        slot_to_vid: Dict[Slot, int],
    ) -> Tuple[
        List[SlotVariant], List[SlotVariant], List[SlotVariant], List[NamedSlot]
    ]:
        """
        Build input/output/mutable_buffer maps and named slots.

        Returns:
            (input_map, output_map, mutable_buffer_map, named_slots)
        """
        input_map: List[SlotVariant] = []
        output_map: List[SlotVariant] = []
        mutable_buffer_map: List[SlotVariant] = []
        # Canonical (unprefixed) name → Slot.  The prefix is applied only at
        # the exit boundaries: NamedSlot construction and NamedDataStore keys.
        name_to_slot: Dict[str, Slot] = {}

        for ispec in self.ep.graph_signature.input_specs:
            slot = self.slot_manager.get_slot(ispec.arg.name)
            if slot is None:
                continue
            assert isinstance(slot, Slot)
            name = ispec.target if ispec.target is not None else ispec.arg.name
            if slot.id_space == IdSpace.Input:
                input_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                name_to_slot[name] = slot
            elif slot.id_space == IdSpace.MutableBuffer:
                mutable_buffer_map.append(
                    self._to_slot_variant(slot, slot_to_tid, slot_to_vid)
                )
                name_to_slot[name] = slot
            else:
                if slot in used_slots:
                    name_to_slot[name] = slot

        for ospec in self.ep.graph_signature.output_specs:
            name = ospec.arg.name
            slot = self.slot_manager.get_slot(name)
            if slot is None:
                continue
            assert isinstance(slot, Slot)
            if slot.id_space == IdSpace.Output:
                output_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                name = ospec.target if ospec.target is not None else ospec.arg.name
                name_to_slot[name] = slot
            elif slot.id_space == IdSpace.MutableBuffer:
                name = ospec.target if ospec.target is not None else ospec.arg.name
                name_to_slot[name] = slot

        for name in self.extra_constants:
            slot = self.slot_manager.get_slot(name)
            assert slot is not None and isinstance(slot, Slot)
            if slot in used_slots:
                name_to_slot[name] = slot

        # Store unprefixed constant mapping for get_named_data_store()
        self._constant_name_to_slot = {
            n: s for n, s in name_to_slot.items() if s.id_space == IdSpace.Constant
        }

        # Apply prefix at the exit boundary — the FlatBuffer named_slots
        named_slots = [
            NamedSlot(
                name=self._prefix_key(n),
                slot=self._to_slot_variant(s, slot_to_tid, slot_to_vid),
            )
            for n, s in name_to_slot.items()
        ]

        return input_map, output_map, mutable_buffer_map, named_slots

    def _build_tensor_meta(  # noqa: C901
        self,
        used_slots: Set[Slot],
        slot_to_tid: Dict[Slot, int],
        slot_to_vid: Dict[Slot, int],
        num_tensors: Dict[IdSpace, int],
    ) -> List[TensorMeta]:
        """
        Build tensor metadata list with shape/dtype information.

        Static dimensions are stored as ShapeDim(value=N).
        Dynamic dimensions (SymInt) are stored as ShapeDim(value=-1)
        with min/max bounds from the shape_env.

        Note: tensor_meta shapes are only consumed by the runtime for
        constant and mutable buffer allocation (which are always static).
        Dynamic dim metadata is informational — the runtime resolves
        dynamic shapes via SymSizeNode at execution time.
        """

        def _get_dim_bounds(dim: torch.SymInt) -> tuple:
            """Get (min, max) bounds for a symbolic dimension."""
            try:
                node = dim.node
                shape_env = node.shape_env
                if shape_env is not None:
                    expr = node.expr
                    lower = int(shape_env.bound_sympy(expr).lower)
                    upper = int(shape_env.bound_sympy(expr).upper)
                    if upper > 2**30:
                        return (lower, -1)  # treat as unbounded
                    return (lower, upper)
            except Exception:
                pass
            return (0, -1)  # unbounded fallback

        def to_tensor_meta(t: torch.Tensor) -> TensorMeta:
            shape: List[ShapeDim] = []
            for dim in t.shape:
                if isinstance(dim, torch.SymInt):
                    lo, hi = _get_dim_bounds(dim)
                    shape.append(ShapeDim(value=-1, min_value=lo, max_value=hi))
                else:
                    shape.append(ShapeDim(value=int(dim)))

            dim_order = list(range(len(t.shape))) if len(t.shape) > 0 else None

            return TensorMeta(
                shape=shape,
                scalar_type=torch_dtype_to_scalar_type(t.dtype),
                dim_order=dim_order,
            )

        tensor_meta: Dict[int, TensorMeta] = {}
        for n in self.node_info:
            slot = self.slot_manager.get_slot(n)
            if not isinstance(slot, tuple):
                slot = (slot,)
            fake_val = n.meta.get("val", None)
            if not isinstance(fake_val, tuple):
                fake_val = (fake_val,)
            for s, fv in zip(slot, fake_val):
                if s not in used_slots:
                    continue
                if s.id_type != IdType.Tensor:
                    continue
                if s.id_space == IdSpace.Temp:
                    continue
                idx = slot_to_tid[s]
                if fv is not None and hasattr(fv, "shape"):
                    tensor_meta[idx] = to_tensor_meta(fv)

        for name, t in self.extra_constants.items():
            slot = self.slot_manager.get_slot(name)
            assert slot is not None and isinstance(slot, Slot)
            if slot in used_slots:
                idx = slot_to_tid[slot]
                tensor_meta[idx] = to_tensor_meta(t)

        num_non_temp_tensors = sum(num_tensors.values()) - num_tensors[IdSpace.Temp]
        return [tensor_meta.get(i) for i in range(num_non_temp_tensors)]

    def _build_mlx_graph(self) -> MLXGraph:
        # Check support
        for node, info in self.node_info.items():
            if not info.supported:
                raise ValueError(
                    f"Found unsupported node: {node}\nReason: {info.unsupported_reason}"
                )

        # Collect slots and create mappings
        used_slots, num_tensors, num_values = self._collect_used_slots()
        slot_to_tid, slot_to_vid = self._create_slot_mappings(used_slots)

        # Store for use in get_constant_data() - needed to serialize in Tid order
        self._slot_to_final_tid = slot_to_tid

        # Build I/O maps and metadata
        input_map, output_map, mutable_buffer_map, named_slots = self._build_io_maps(
            used_slots, slot_to_tid, slot_to_vid
        )
        tensor_meta_list = self._build_tensor_meta(
            used_slots, slot_to_tid, slot_to_vid, num_tensors
        )

        # Compute final counts
        num_constant_tensors = num_tensors[IdSpace.Constant]
        num_temp_tensors = num_tensors[IdSpace.Temp]
        num_values_count = sum(num_values.values())

        return MLXGraph(
            version="1",
            num_constant_tensors=num_constant_tensors,
            num_input_tensors=num_tensors[IdSpace.Input],
            num_output_tensors=num_tensors[IdSpace.Output],
            num_mutable_buffer_tensors=num_tensors[IdSpace.MutableBuffer],
            num_temp_tensors=num_temp_tensors,
            num_values=num_values_count,
            instruction_chains=[
                InstructionChain(instructions=chain) for chain in self._chains
            ],
            main_chain_idx=0,
            init_chain_idx=self.init_chain_idx,
            input_map=input_map,
            output_map=output_map,
            mutable_buffer_map=mutable_buffer_map,
            named_slots=named_slots,
            tensor_meta=tensor_meta_list,
        )

    def get_named_data_store(self) -> NamedDataStore:
        """
        Get a NamedDataStore containing all constant tensors.

        Uses the unprefixed canonical-name → Slot mapping built by
        ``_build_io_maps()`` so that tensor lookups hit ``ep.state_dict`` /
        ``ep.constants`` / ``extra_constants`` (which all use unprefixed
        keys).  The prefix is applied at the exit boundary — the
        ``NamedDataStore`` key — so it matches the FlatBuffer ``named_slots``.

        To reduce peak memory, each constant is deleted from the EP
        immediately after its bytes are added to the NamedDataStore.
        This avoids holding two full copies of all constants simultaneously
        (important for large models where constants can be 20+ GB).
        """
        named_data_store = NamedDataStore()

        # Sort by final TID for deterministic ordering
        entries = sorted(
            self._constant_name_to_slot.items(),
            key=lambda x: self._slot_to_final_tid.get(x[1], 0),
        )

        # Free EP constants not used by the MLX graph to reduce peak memory.
        used = set(self._constant_name_to_slot.keys())
        for ispec in self.ep.graph_signature.input_specs:
            if ispec.arg.name in used and ispec.target is not None:
                used.add(ispec.target)

        for d in (self.ep._state_dict, self.ep._constants):
            for name in list(d.keys()):
                if name not in used and isinstance(d[name], torch.Tensor):
                    del d[name]

        logger.debug(f"Adding {len(entries)} constants to NamedDataStore...")
        for canonical_name, _slot in entries:
            tensor = self._find_constant_tensor(canonical_name)
            if tensor is None:
                continue

            t = tensor.detach().cpu().contiguous()
            named_data_store.add_named_data(
                key=self._prefix_key(canonical_name),
                data=t,
                alignment=16,
            )

            # Free the original tensor from the EP immediately.
            # The contiguous copy is now serialized as bytes in the
            # NamedDataStore — the EP reference is no longer needed.
            # (It would be deleted by lowered_backend_module.py after
            # preprocess() returns anyway.)
            self._delete_constant_tensor(canonical_name)
            del tensor, t

        logger.debug("Done adding constants to NamedDataStore")

        return named_data_store

    def get_mutable_buffer_names(self) -> List[str]:
        """
        Get the names of all mutable buffers in Tid order.

        Returns:
            List of mutable buffer names in the order they appear in mutable_buffer_map.
        """
        assert self._mlx_graph is not None, "Must call build() first"

        names = []
        for name, slot in self.slot_manager.name_to_slot.items():
            if isinstance(slot, tuple):
                continue
            if slot.id_space != IdSpace.MutableBuffer:
                continue
            if slot in self._slot_to_final_tid:
                names.append((name, self._slot_to_final_tid[slot]))

        # Sort by Tid and return just the names
        names.sort(key=lambda x: x[1])
        return [n for n, _ in names]

    def _find_constant_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Find a constant tensor by name from various sources."""
        result = self._resolve_constant(name)
        if result is None:
            return None

        d, k = result
        return d[k]

    def _delete_constant_tensor(self, name: str) -> None:
        """Delete a constant from the EP to free memory during serialization."""

        result = self._resolve_constant(name)
        if result:
            d, k = result
            del d[k]

    def _resolve_constant(self, name):
        """Returns (dict, key) or None."""
        if name in self.ep._state_dict:
            return self.ep._state_dict, name
        if name in self.ep._constants:
            return self.ep._constants, name
        if name in self.extra_constants:
            return self.extra_constants, name
        for ispec in self.ep.graph_signature.input_specs:
            if ispec.arg.name == name and ispec.target is not None:
                if ispec.target in self.ep._state_dict:
                    return self.ep._state_dict, ispec.target
                if ispec.target in self.ep._constants:
                    return self.ep._constants, ispec.target
        return None
