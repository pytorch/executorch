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
Pattern handlers are registered in mlx_patterns.py.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
    DataSegment,
    DTypeId,
    FloatOrVid,
    Instruction,
    IntOrVid,
    MLXGraph,
    NamedSlot,
    OpNodeUnion,
    SlotType,
    SlotVariant,
    TensorMeta,
    Tid,
    Vid,
)
from executorch.exir.sym_util import eval_shape_upper_bound
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node
from torch.utils import _pytree as pytree

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


# =============================================================================
# Utility functions
# =============================================================================


def get_aten_target(target):
    """
    Unwrap EdgeOpOverload to get the underlying ATen op.

    In Edge IR, ops are wrapped in EdgeOpOverload. This extracts the
    underlying ATen op for consistent comparison.
    """
    if hasattr(target, "_op") and "EdgeOpOverload" in type(target).__name__:
        return target._op
    return target


# Mapping from _copy variants to their non-copy equivalents.
# Edge IR uses _copy variants for certain ops, but for pattern matching
# we want to compare against the semantic operation.
_COPY_TO_NON_COPY = {
    torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    torch.ops.aten.transpose_copy.int: torch.ops.aten.transpose.int,
    torch.ops.aten.view_copy.default: torch.ops.aten.view.default,
    torch.ops.aten.permute_copy.default: torch.ops.aten.permute.default,
}


def get_aten_target_normalized(target):
    """
    Get ATen target, mapping _copy variants to their non-copy equivalents.

    Use this for pattern matching where Edge IR uses _copy variants but
    we want to match the semantic operation.

    E.g., aten.transpose_copy.int -> aten.transpose.int
    """
    target = get_aten_target(target)
    return _COPY_TO_NON_COPY.get(target, target)


def emit_stop_position(
    P: "MLXProgramBuilder",
    start: "Union[int, Slot]",
    length_tensor: "Slot",
    length_dim: int,
    length_meta: "Optional[torch.Tensor]" = None,
) -> "Union[int, Slot]":
    """
    Emit nodes to compute stop = start + length for slice operations.

    May emit SymSizeNode and/or AddScalarNode depending on whether
    start and length are static or dynamic.

    Args:
        P: The program builder
        start: Start position (int or Slot)
        length_tensor: The tensor slot whose dimension gives the length
        length_dim: Which dimension of length_tensor contains the length
        length_meta: Optional tensor metadata for static length extraction

    Returns:
        stop position as int (if fully static) or Slot (if any dynamic)
    """
    from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
        AddScalarNode,
        IntOrVid,
        SymSizeNode,
    )

    # Check if seq_len is symbolic (dynamic)
    seq_len_is_symbolic = False
    seq_len_concrete = None

    if length_meta is not None:
        seq_len_dim = length_meta.shape[length_dim]
        if hasattr(seq_len_dim, "node"):
            seq_len_is_symbolic = True
        else:
            seq_len_concrete = int(seq_len_dim)

    if seq_len_is_symbolic or length_meta is None:
        # Dynamic seq_len: emit SymSizeNode to get length at runtime
        _, seq_len_slot = P.slot_manager.make_tmp_value_slot()
        P._emit(
            SymSizeNode(
                a=P._slot_to_tid(length_tensor),
                dim=length_dim,
                out=P._slot_to_vid(seq_len_slot),
            )
        )
        _, stop_slot = P.slot_manager.make_tmp_value_slot()
        if isinstance(start, Slot):
            start_iov = P._to_int_or_vid(start)
        else:
            start_iov = IntOrVid.from_literal(int(start))
        P._emit(
            AddScalarNode(
                a=start_iov,
                b=IntOrVid.from_vid(P._slot_to_vid(seq_len_slot)),
                out=P._slot_to_vid(stop_slot),
            )
        )
        return stop_slot
    else:
        # Static seq_len
        if isinstance(start, Slot):
            # Dynamic start + static length
            _, stop_slot = P.slot_manager.make_tmp_value_slot()
            P._emit(
                AddScalarNode(
                    a=P._to_int_or_vid(start),
                    b=IntOrVid.from_literal(seq_len_concrete),
                    out=P._slot_to_vid(stop_slot),
                )
            )
            return stop_slot
        else:
            # Both static - just return the sum
            return start + seq_len_concrete


# =============================================================================
# Type conversions
# =============================================================================

_TORCH_DTYPE_TO_DTYPEID: Dict[torch.dtype, DTypeId] = {
    torch.float16: DTypeId.f16,
    torch.float32: DTypeId.f32,
    torch.bfloat16: DTypeId.bf16,
    torch.int32: DTypeId.i32,
    torch.int64: DTypeId.i64,
    torch.uint32: DTypeId.u32,
    torch.uint8: DTypeId.u8,
    torch.bool: DTypeId.boolean,
    torch.int8: DTypeId.i8,
}


def _torch_dtype_to_dtypeid(dtype: torch.dtype) -> DTypeId:
    if dtype not in _TORCH_DTYPE_TO_DTYPEID:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _TORCH_DTYPE_TO_DTYPEID[dtype]


# =============================================================================
# Slot management
# =============================================================================


class IdType(Enum):
    Tensor = auto()
    SymInt = auto()
    SymBool = auto()


class IdSpace(Enum):
    Constant = auto()
    Input = auto()
    Output = auto()
    MutableBuffer = auto()
    Temp = auto()


@dataclass(frozen=True)
class Slot:
    id_type: IdType
    id_space: IdSpace
    idx: Optional[int] = None


class IdManager:
    def __init__(self):
        self.free: list[int] = []
        self.next_new_id = 0

    def get_id(self):
        return self.free.pop() if self.free else self._bump()

    def _bump(self):
        idx = self.next_new_id
        self.next_new_id += 1
        return idx

    def return_id(self, idx):
        if self.free and self.free[-1] == idx:
            return
        self.free.append(idx)


class SlotManager:
    def __init__(self):
        self.tid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.vid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.name_to_slot: Dict[str, Slot] = {}

    def set_slot(self, node_or_name: Union[Node, str], slot: Slot):
        if isinstance(node_or_name, Node):
            node_or_name = node_or_name.name
        # Allow setting a slot to the same value (e.g., for in-place ops like SLICE_UPDATE)
        existing = self.name_to_slot.get(node_or_name)
        if existing is not None:
            # If already set to the same slot, it's fine
            if existing == slot:
                return
            raise AssertionError(
                f"Slot for {node_or_name} already set to {existing}, trying to set to {slot}"
            )
        self.name_to_slot[node_or_name] = slot

    def get_slot(
        self, node_or_name: Union[Node, str]
    ) -> Optional[Union[Tuple[Slot], Slot]]:
        if isinstance(node_or_name, Node):
            node_or_name = node_or_name.name
        return self.name_to_slot.get(node_or_name, None)

    def _val_to_idtype(self, v) -> IdType:
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(v, FakeTensor):
            return IdType.Tensor
        elif isinstance(v, torch.SymInt):
            return IdType.SymInt
        elif isinstance(v, torch.SymBool):
            return IdType.SymBool
        else:
            raise NotImplementedError(f"val_to_idtype: {v}")

    def is_alive(self, slot: Slot) -> bool:
        if slot.id_type == IdType.Tensor:
            manager = self.tid_managers[slot.id_space]
        else:
            manager = self.vid_managers[slot.id_space]
        idx = slot.idx
        if idx >= manager.next_new_id:
            return False
        if idx in manager.free:
            return False
        return True

    def make_constant_slot(self, name: str) -> Slot:
        assert name not in self.name_to_slot
        id_space = IdSpace.Constant
        manager = self.tid_managers[id_space]
        idx = manager.get_id()
        slot = Slot(id_type=IdType.Tensor, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        return slot

    def make_tmp_slot(self) -> Tuple[str, Slot]:
        name = f"tmp_{uuid.uuid4().hex}"
        id_space = IdSpace.Temp
        manager = self.tid_managers[id_space]
        idx = manager.get_id()
        slot = Slot(id_type=IdType.Tensor, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        return name, slot

    def make_tmp_value_slot(self) -> Tuple[str, Slot]:
        """Create a temporary SymInt slot and register it."""
        name = f"tmp_val_{uuid.uuid4().hex}"
        id_space = IdSpace.Temp
        manager = self.vid_managers[id_space]
        idx = manager.get_id()
        slot = Slot(id_type=IdType.SymInt, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        return name, slot

    def make_or_get_slots(
        self, node: Node, id_space: IdSpace = IdSpace.Temp
    ) -> Tuple[Slot, ...]:
        """
        Get or create slots for a node. Always returns a tuple of slots.

        Use this for multi-output ops (e.g., rope returns (q_out, k_out)).
        For single-output ops, prefer make_or_get_slot() which returns a single Slot.
        """
        if node.name in self.name_to_slot:
            slot = self.name_to_slot[node.name]
            # Normalize to tuple for consistent return type
            if not isinstance(slot, tuple):
                return (slot,)
            return slot

        val = node.meta.get("val", None)
        assert val is not None, f"Node {node} has no val"
        if not isinstance(val, (list, tuple)):
            val = (val,)

        slots = []
        for v in val:
            id_type = self._val_to_idtype(v)
            if id_type == IdType.Tensor:
                manager = self.tid_managers[id_space]
            else:
                manager = self.vid_managers[id_space]
            idx = manager.get_id()
            slots.append(Slot(id_type=id_type, id_space=id_space, idx=idx))
        slots = tuple(slots)

        # Store in the format that matches the node's output structure
        if len(slots) == 1:
            self.set_slot(node, slots[0])
        else:
            self.set_slot(node, slots)
        return slots

    def make_or_get_slot(self, node: Node, id_space: IdSpace = IdSpace.Temp) -> Slot:
        """
        Get or create a slot for a single-output node. Returns a single Slot.

        Use this for single-output ops (the common case).
        For multi-output ops, use make_or_get_slots() instead.
        """
        slots = self.make_or_get_slots(node, id_space)
        assert len(slots) == 1, (
            f"Expected single output for node {node.name}, got {len(slots)}. "
            f"Use make_or_get_slots() for multi-output ops."
        )
        return slots[0]


# =============================================================================
# Pattern handlers for fused ops
# =============================================================================

# Handler type: takes (builder, node) and returns optional slot(s)
# Returns None for no-ops, Slot for single outputs, Tuple[Slot, ...] for multiple outputs
Handler = Callable[
    ["MLXProgramBuilder", Node], Optional[Union["Slot", Tuple["Slot", ...]]]
]


class PatternHandler:
    def __init__(self, head: Node, body: List[Node]) -> None:
        self.head: Node = head
        self.body: List[Node] = body

    @classmethod
    def deferred_handler(cls, P: "MLXProgramBuilder", n: Node) -> None:
        pass

    @classmethod
    def maybe_create(
        cls, ep: ExportedProgram, head: Node
    ) -> Optional["PatternHandler"]:
        raise NotImplementedError

    def __call__(self, P: "MLXProgramBuilder", n: Node) -> None:
        raise NotImplementedError

    def set_handlers(self, P: "MLXProgramBuilder"):
        if P.node_info[self.head].handler is not None:
            raise AssertionError(
                f"Head node {self.head.name} already has handler {P.node_info[self.head].handler}, "
                f"cannot set pattern {self.__class__.__name__}"
            )
        for n in self.body:
            if P.node_info[n].handler is not None:
                raise AssertionError(
                    f"Body node {n.name} already has handler {P.node_info[n].handler}, "
                    f"cannot set pattern {self.__class__.__name__}"
                )

        # Set handlers
        logging.debug(
            f"Pattern {self.__class__.__name__} assigning handlers: "
            f"HEAD={self.head.name}, BODY={[n.name for n in self.body]}"
        )
        P.node_info[self.head].handler = self
        for n in self.body:
            P.node_info[n].handler = PatternHandler.deferred_handler


# =============================================================================
# Node info tracking
# =============================================================================


@dataclass
class NodeInfo:
    handled: bool = False
    handler: Optional[Union[Handler, PatternHandler]] = None
    supported: bool = False
    unsupported_reason: Optional[str] = None
    name: Optional[str] = None
    remaining_reads: int = 0


# =============================================================================
# Pattern matching
# =============================================================================


class PatternMatcher:
    """
    Discovers and applies pattern handlers to an FX graph.

    Pattern handlers match multi-node subgraphs and lower them to optimized
    MLX operations. This class orchestrates the pattern discovery process:

    1. Iterates through all registered pattern types
    2. For each pattern, tries to match it against every node in the graph
    3. When a match is found, assigns handlers to the head and body nodes

    The ordering matters: patterns are matched before dead code elimination
    because some pattern body nodes (e.g., update_cache) have no users
    since they mutate in-place, but they're not dead.
    """

    def __init__(self, ep: ExportedProgram, registry: "MLXOpRegistry"):
        self.ep = ep
        self.registry = registry
        self._matches: List[PatternHandler] = []

    def find_patterns(self) -> List[PatternHandler]:
        """
        Find all pattern matches in the graph.

        Returns a list of PatternHandler instances, one for each match found.
        Patterns are tried in registration order.
        """
        self._matches = []
        for name in self.registry.patterns():
            self._find_pattern(name)
        return self._matches

    def _find_pattern(self, name: str) -> None:
        """Try to match a single pattern type against all nodes."""
        pattern_cls = self.registry.get_pattern_cls(name)
        if pattern_cls is None:
            return

        for n in self.ep.graph.nodes:
            handler = pattern_cls.maybe_create(self.ep, n)
            if handler is not None:
                logging.debug(f"Pattern {name} matched at node {n.name}")
                self._matches.append(handler)


# =============================================================================
# Op registry
# =============================================================================


class MLXOpRegistry:
    """Registry for op handlers and pattern handlers."""

    def __init__(self):
        self._handlers: Dict[Union[str, Callable], Handler] = {}
        self._patterns: Dict[str, Type[PatternHandler]] = {}

    def reset(self) -> None:
        """Reset the registry to empty state. Useful for testing."""
        self._handlers.clear()
        self._patterns.clear()

    def register(self, target: Union[str, Callable, list, tuple]):
        """Decorator to register a handler for one or more op targets."""

        def deco(fn: Handler):
            targets = target if isinstance(target, (list, tuple)) else [target]
            for t in targets:
                if t in self._handlers:
                    raise ValueError(f"Target {t} already registered")
                self._handlers[t] = fn
            return fn

        return deco

    def get_handler(self, node: Node) -> Optional[Handler]:
        """Get the handler for a node, or None if not registered."""
        t = node.target
        if t in self._handlers:
            return self._handlers[t]
        # Handle EdgeOpOverload by extracting the underlying ATen op
        if hasattr(t, "_op") and t._op in self._handlers:
            return self._handlers[t._op]
        # Check for string-based targets (e.g., higher_order ops)
        target_str = str(t)
        if target_str in self._handlers:
            return self._handlers[target_str]
        return None

    def registered_ops(self) -> set:
        """Return all registered op targets."""
        return set(self._handlers.keys())

    def register_pattern(self, name: str):
        """Decorator to register a pattern handler class."""

        def deco(cls: Type[PatternHandler]):
            if not issubclass(cls, PatternHandler):
                raise TypeError(
                    "register_pattern must decorate a PatternHandler subclass"
                )
            if name in self._patterns:
                raise ValueError(f"Pattern '{name}' already registered")
            self._patterns[name] = cls
            return cls

        return deco

    def get_pattern_cls(self, name: str) -> Optional[Type[PatternHandler]]:
        """Get a pattern handler class by name."""
        return self._patterns.get(name)

    def get_noop_handler(self) -> Optional[Handler]:
        """Get the NOOP handler, if registered."""
        return self._handlers.get("NOOP")

    def patterns(self):
        """Return all registered pattern names."""
        return self._patterns.keys()


# Global registry
REGISTRY = MLXOpRegistry()


# =============================================================================
# MLXProgramBuilder - main class
# =============================================================================


class MLXProgramBuilder:
    """
    Builds an MLXGraph from an ExportedProgram.

    Args:
        ep: The ExportedProgram to build from
    """

    def __init__(self, ep: ExportedProgram):
        self.ep: ExportedProgram = ep
        self._instrs: List[Instruction] = []
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

    # -------------------------------------------------------------------------
    # Op emission helpers
    # -------------------------------------------------------------------------

    def _emit(self, op: OpNodeUnion) -> None:
        self._instrs.append(Instruction(op=op))

    # -------------------------------------------------------------------------
    # Slot and arg helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Slot to Tid/Vid conversion
    # -------------------------------------------------------------------------

    def _slot_to_tid(self, slot: Slot) -> Tid:
        """Convert a tensor Slot to a Tid, recording it for later remapping."""
        assert slot.id_type == IdType.Tensor
        # Use local slot.idx as placeholder - will be remapped to global idx in build()
        tid = Tid(idx=slot.idx)
        self._tid_slot_map.append((tid, slot))
        return tid

    def _slot_to_vid(self, slot: Slot) -> Vid:
        """Convert a value Slot to a Vid, recording it for later remapping."""
        assert slot.id_type != IdType.Tensor
        vid = Vid(idx=slot.idx)
        self._vid_slot_map.append((vid, slot))
        return vid

    def make_tmp_vid(self) -> Vid:
        """
        Create a temporary Vid for intermediate value computations.

        This is useful for op handlers that need to emit intermediate scalar
        computations (e.g., computing end = start + length in narrow).

        Returns:
            A new Vid that can be used as an output for scalar operations.
        """
        idx = self.slot_manager.vid_managers[IdSpace.Temp].get_id()
        return Vid(idx=idx)

    def _to_int_or_vid(self, v: Union[int, Slot]) -> IntOrVid:
        if isinstance(v, Slot):
            return IntOrVid.from_vid(self._slot_to_vid(v))
        return IntOrVid.from_literal(int(v))

    def _to_float_or_vid(self, v: Union[float, int, Slot]) -> FloatOrVid:
        if isinstance(v, Slot):
            return FloatOrVid.from_vid(self._slot_to_vid(v))
        return FloatOrVid.from_literal(float(v))

    # -------------------------------------------------------------------------
    # Node lifecycle management
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # I/O slot creation
    # -------------------------------------------------------------------------

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
            if kind == OutputKind.USER_OUTPUT:
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

    # -------------------------------------------------------------------------
    # Build process
    # -------------------------------------------------------------------------

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

    def _process_nodes(self) -> None:
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

            if n.op in ("placeholder", "output"):
                self._mark_supported(n)
                continue

            if self.node_info[n].handler is not None:
                handler = self.node_info[n].handler
                handler(self, n)
                self._mark_supported(n, handler=handler)
                continue

            handler = REGISTRY.get_handler(n)
            if handler is None:
                msg = f"no handler for target={n.target}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slot(n)
                self._mark_unsupported(n, msg)
                continue

            try:
                handler(self, n)
                self._mark_supported(n, handler=handler)
            except Exception as e:
                trace_str = traceback.format_exc()
                msg = f"{handler} failed for {n.target}: {e}.\n{trace_str}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slot(n)
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

    def build(self) -> MLXGraph:
        if self._mlx_graph is not None:
            return self._mlx_graph

        self._process_nodes()
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
                assert (
                    self.slot_manager.get_slot(n) is None
                ), f"Did not expect node {n} handled by {info.handler} to have a slot"
            else:
                assert (
                    self.slot_manager.get_slot(n) is not None
                ), f"Expected slot for node {n}"

    def _collect_used_slots(
        self,
    ) -> Tuple[Set[Slot], Dict[IdSpace, int], Dict[IdSpace, int]]:
        """
        Collect all used slots and count tensors/values per IdSpace.

        Returns:
            (used_slots, num_tensors, num_values)
        """
        used_slots: Set[Slot] = set()
        for _n, slot in self.slot_manager.name_to_slot.items():
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                used_slots.add(s)

        num_tensors: Dict[IdSpace, int] = defaultdict(int)
        num_values: Dict[IdSpace, int] = defaultdict(int)
        seen: Set[Slot] = set()
        for _n, slot in self.slot_manager.name_to_slot.items():
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
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
                    logging.warning(f"Slot {slot} not found in slot_to_tid mapping")

        if hasattr(self, "_vid_slot_map"):
            for vid, slot in self._vid_slot_map:
                if slot in slot_to_vid:
                    vid.idx = slot_to_vid[slot]
                else:
                    logging.warning(f"Slot {slot} not found in slot_to_vid mapping")

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
        name_to_slot_dict: Dict[str, Slot] = {}

        for ispec in self.ep.graph_signature.input_specs:
            slot = self.slot_manager.get_slot(ispec.arg.name)
            if slot is None:
                continue
            assert isinstance(slot, Slot)
            name = ispec.target if ispec.target is not None else ispec.arg.name
            if slot.id_space == IdSpace.Input:
                input_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                name_to_slot_dict[name] = slot
            elif slot.id_space == IdSpace.MutableBuffer:
                # Mutable buffers are also inputs to the delegate
                input_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                mutable_buffer_map.append(
                    self._to_slot_variant(slot, slot_to_tid, slot_to_vid)
                )
                name_to_slot_dict[name] = slot
            else:
                if slot in used_slots:
                    name_to_slot_dict[name] = slot

        for ospec in self.ep.graph_signature.output_specs:
            name = ospec.arg.name
            slot = self.slot_manager.get_slot(name)
            if slot is None:
                continue
            assert isinstance(slot, Slot)
            if slot.id_space == IdSpace.Output:
                output_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                name = ospec.target if ospec.target is not None else ospec.arg.name
                name_to_slot_dict[name] = slot
            elif slot.id_space == IdSpace.MutableBuffer:
                # Mutable buffer mutations are also outputs from the delegate
                output_map.append(self._to_slot_variant(slot, slot_to_tid, slot_to_vid))
                name = ospec.target if ospec.target is not None else ospec.arg.name
                name_to_slot_dict[name] = slot

        for name in self.extra_constants:
            slot = self.slot_manager.get_slot(name)
            assert slot is not None and isinstance(slot, Slot)
            if slot in used_slots:
                name_to_slot_dict[name] = slot

        named_slots = [
            NamedSlot(name=n, slot=self._to_slot_variant(s, slot_to_tid, slot_to_vid))
            for n, s in name_to_slot_dict.items()
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

        For dynamic shapes, symbolic dimensions are tracked as IntOrVid references
        so the runtime can resolve actual sizes dynamically.
        """
        # Build a mapping from SymInt symbol names to their Slots
        symint_symbol_to_slot: Dict[str, Slot] = {}
        for n in self.node_info:
            val = n.meta.get("val", None)
            if isinstance(val, torch.SymInt):
                symbol_name = str(val.node) if hasattr(val, "node") else str(val)
                slot = self.slot_manager.get_slot(n)
                if slot is not None and not isinstance(slot, tuple):
                    symint_symbol_to_slot[symbol_name] = slot

        def to_tensor_meta(t: torch.Tensor) -> TensorMeta:
            shape: List[IntOrVid] = []
            for _i, dim in enumerate(t.shape):
                if isinstance(dim, torch.SymInt):
                    symbol_name = str(dim.node) if hasattr(dim, "node") else str(dim)
                    if symbol_name in symint_symbol_to_slot:
                        slot = symint_symbol_to_slot[symbol_name]
                        vid = Vid(idx=slot_to_vid.get(slot, slot.idx))
                        shape.append(IntOrVid.from_vid(vid))
                    else:
                        # Fall back to upper bound if we can't find the Slot
                        try:
                            from torch.utils._sympy.numbers import int_oo
                        except ImportError:
                            int_oo = None
                        upper = eval_shape_upper_bound([dim])[0]
                        if int_oo is not None and upper is int_oo:
                            shape.append(IntOrVid.from_literal(int(dim)))
                        else:
                            shape.append(IntOrVid.from_literal(upper))
                else:
                    shape.append(IntOrVid.from_literal(int(dim)))

            return TensorMeta(
                shape=shape,
                dtype=_torch_dtype_to_dtypeid(t.dtype),
                strides=None,
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

        # Build I/O maps and metadata
        input_map, output_map, mutable_buffer_map, named_slots = self._build_io_maps(
            used_slots, slot_to_tid, slot_to_vid
        )
        tensor_meta_list = self._build_tensor_meta(
            used_slots, slot_to_tid, slot_to_vid, num_tensors
        )

        # Compute final counts
        num_constant_tensors = num_tensors[IdSpace.Constant]
        num_non_constant_tensors = sum(num_tensors.values()) - num_constant_tensors
        num_non_constant_values = sum(num_values.values())

        return MLXGraph(
            version="1",
            num_constant_tensors=num_constant_tensors,
            num_non_constant_tensors=num_non_constant_tensors,
            num_non_constant_values=num_non_constant_values,
            instructions=self._instrs,
            input_map=input_map,
            output_map=output_map,
            mutable_buffer_map=mutable_buffer_map,
            named_slots=named_slots,
            tensor_meta=tensor_meta_list,
            constant_segment=DataSegment(offset=0, size=0),
        )

    def get_constant_data(self) -> Tuple[bytes, Dict[str, int]]:  # noqa: C901
        """
        Extract constant tensor data.

        Returns:
            (constant_bytes, name_to_offset) mapping constant names to byte offsets.
        """
        assert self._mlx_graph is not None, "Must call build() first"

        from io import BytesIO

        buffer = BytesIO()
        name_to_offset = {}

        for name, slot in self.slot_manager.name_to_slot.items():
            if isinstance(slot, tuple):
                continue
            if slot.id_space != IdSpace.Constant:
                continue

            # Find tensor
            tensor = None
            if name in self.ep.state_dict:
                tensor = self.ep.state_dict[name]
            elif name in self.ep.constants:
                tensor = self.ep.constants[name]
            elif name in self.extra_constants:
                tensor = self.extra_constants[name]
            else:
                # Look up by target
                for ispec in self.ep.graph_signature.input_specs:
                    if ispec.arg.name == name and ispec.target is not None:
                        if ispec.target in self.ep.state_dict:
                            tensor = self.ep.state_dict[ispec.target]
                            break
                        elif ispec.target in self.ep.constants:
                            tensor = self.ep.constants[ispec.target]
                            break

            if tensor is None:
                continue

            # Align to 16 bytes
            current_pos = buffer.tell()
            padding = (16 - (current_pos % 16)) % 16
            if padding > 0:
                buffer.write(b"\x00" * padding)

            name_to_offset[name] = buffer.tell()
            t = tensor.detach().cpu().contiguous()
            # BFloat16 is not supported by numpy, so we extract raw bytes via view
            if t.dtype == torch.bfloat16:
                # View as uint16 to access raw bf16 bytes
                tensor_bytes = t.view(torch.uint16).numpy().tobytes()
            else:
                tensor_bytes = t.numpy().tobytes()
            buffer.write(tensor_bytes)

        return buffer.getvalue(), name_to_offset


# =============================================================================
# Import op and pattern handlers to register them
# =============================================================================

# These imports register the handlers with the REGISTRY
# They must come after REGISTRY is defined
from executorch.backends.apple.mlx import ops, patterns  # noqa: F401, E402
