#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.fx.node import Node


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


@dataclass(eq=False, frozen=True)
class Slot:
    """Represents an allocated tensor or symbolic int slot.

    Uses identity-based equality and hashing (not field-based) so that
    two Slots with the same (id_type, id_space, idx) — which can happen
    when the delete-as-you-go allocator recycles an idx — remain distinct
    in sets and dicts during build().
    """

    id_type: IdType
    id_space: IdSpace
    idx: Optional[int] = None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class IdManager:
    def __init__(self):
        self.free: set[int] = set()
        self.next_new_id = 0

    def get_id(self):
        return self.free.pop() if self.free else self._bump()

    def _bump(self):
        idx = self.next_new_id
        self.next_new_id += 1
        return idx

    def return_id(self, idx):
        self.free.add(idx)


class SlotManager:
    def __init__(self):
        self.tid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.vid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.name_to_slot: Dict[str, Slot] = {}
        # Stack of active temp-slot scopes (see ``tmp_scope``). Temp tids/vids
        # allocated via make_tmp_slot()/make_tmp_value_slot() are registered on
        # the innermost scope and their ids returned for reuse on scope exit.
        self._tmp_scopes: List[List[Slot]] = []

    @contextmanager
    def tmp_scope(self) -> Iterator[None]:
        """Scope temporary slot allocations so their ids can be reused.

        Temp tids/vids allocated via :meth:`make_tmp_slot` /
        :meth:`make_tmp_value_slot` inside this context are returned to their
        id pools when the context exits, so later allocations (temp or node)
        can reuse them. Allocating a temp slot outside any ``tmp_scope`` raises
        ``RuntimeError``.

        Scopes may be nested; each allocation is tied to the innermost scope.
        The Slot objects stay in ``name_to_slot`` (mirroring node-slot reclaim
        via ``return_id``) so serialization still sees every distinct slot.

        Invariant: the slots yielded here are dead after the context exits.
        Never ``set_slot`` a temp slot as a node's persistent output because its id is
        reclaimed on scope exit and would be reused (and coalesced) by a later
        node while still live. Node outputs must come from ``make_or_get_slot``.
        """
        self._tmp_scopes.append([])
        try:
            yield
        finally:
            scope = self._tmp_scopes.pop()
            for slot in scope:
                if slot.id_type == IdType.Tensor:
                    self.tid_managers[slot.id_space].return_id(slot.idx)
                else:
                    self.vid_managers[slot.id_space].return_id(slot.idx)

    def _new_tmp_slot(self, id_type: IdType, prefix: str) -> Tuple[str, Slot]:
        if not self._tmp_scopes:
            raise RuntimeError(
                f"{prefix}() must be called within a SlotManager.tmp_scope() "
                "context so temporary ids can be reclaimed and reused."
            )
        name = f"{prefix}_{uuid.uuid4().hex}"
        id_space = IdSpace.Temp
        manager = (
            self.tid_managers[id_space]
            if id_type == IdType.Tensor
            else self.vid_managers[id_space]
        )
        idx = manager.get_id()
        slot = Slot(id_type=id_type, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        self._tmp_scopes[-1].append(slot)
        return name, slot

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
        return self._new_tmp_slot(IdType.Tensor, "tmp")

    def make_tmp_value_slot(self) -> Tuple[str, Slot]:
        """Create a temporary SymInt slot and register it."""
        return self._new_tmp_slot(IdType.SymInt, "tmp_val")

    def make_or_get_slots(
        self, node: Node, id_space: IdSpace = IdSpace.Temp
    ) -> Tuple[Slot, ...]:
        """
        Get or create slots for a node. Always returns a tuple of slots.

        Use this for multi-output ops (e.g., topk returns (values, indices)).
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
