#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Regression tests for temp-slot recycling in the MLX program builder.

Two invariants are guarded here:

1. ``SlotManager.tmp_scope`` reclaims temp tids/vids on exit (and creating a
   temp slot outside a scope raises), so local ids are reused.
2. The serialized graph coalesces slots that share ``(id_space, idx)`` to a
   single global Tid/Vid, so ``num_temp_tensors`` / ``num_values`` reflect that
   reuse. Without this, recycled slots each get their own runtime slot (which is
   never freed until end-of-execution ``reset()``), inflating peak memory. This
   is easy to silently reintroduce (e.g. enumerating distinct Slot objects), so
   it is asserted directly.

Run::

    python -m unittest executorch.backends.mlx.builder.test_slot_recycling
"""

import unittest

import torch
import torch.nn as nn

from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import (
    IdSpace,
    IdType,
    Slot,
    SlotManager,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import Tid, Vid


def _trivial_ep():
    """Minimal ExportedProgram just to satisfy ``MLXProgramBuilder.__init__``.

    The graph is never processed; the coalescing tests drive the builder's slot
    bookkeeping directly.
    """

    class _Identity(nn.Module):
        def forward(self, x):
            return x + 1

    return torch.export.export(_Identity(), (torch.zeros(2),))


class TmpScopeTest(unittest.TestCase):
    def test_make_tmp_requires_scope(self):
        sm = SlotManager()
        with self.assertRaises(RuntimeError):
            sm.make_tmp_slot()
        with self.assertRaises(RuntimeError):
            sm.make_tmp_value_slot()

    def test_tmp_ids_reclaimed_and_reused(self):
        sm = SlotManager()
        with sm.tmp_scope():
            _, a = sm.make_tmp_slot()
            _, b = sm.make_tmp_slot()
            self.assertNotEqual(a.idx, b.idx)  # live simultaneously
            self.assertTrue(sm.is_alive(a))
        # Reclaimed on exit.
        self.assertFalse(sm.is_alive(a))
        self.assertFalse(sm.is_alive(b))
        # Next scope reuses a freed idx.
        with sm.tmp_scope():
            _, c = sm.make_tmp_slot()
            self.assertIn(c.idx, (a.idx, b.idx))

    def test_value_slots_reclaimed(self):
        sm = SlotManager()
        with sm.tmp_scope():
            _, v = sm.make_tmp_value_slot()
            self.assertTrue(sm.is_alive(v))
        self.assertFalse(sm.is_alive(v))

    def test_nested_scopes(self):
        sm = SlotManager()
        with sm.tmp_scope():
            _, outer = sm.make_tmp_slot()
            with sm.tmp_scope():
                _, inner = sm.make_tmp_slot()
            # Inner scope reclaimed its slot; outer slot is still live.
            self.assertFalse(sm.is_alive(inner))
            self.assertTrue(sm.is_alive(outer))


class SlotCoalescingTest(unittest.TestCase):
    """Slots sharing ``(id_space, idx)`` must map to one global Tid/Vid."""

    def _builder_with_slots(self, tensor_slots, value_slots):
        P = MLXProgramBuilder(_trivial_ep())
        # Start from a clean slot table so the trivial graph's own slots don't
        # interfere, then register synthetic slots as if emitted by handlers.
        P.slot_manager = SlotManager()
        P._tid_slot_map = []
        P._vid_slot_map = []
        for i, s in enumerate(tensor_slots):
            P.slot_manager.name_to_slot[f"t{i}"] = s
            P._tid_slot_map.append((Tid(idx=None), s))
        for i, s in enumerate(value_slots):
            P.slot_manager.name_to_slot[f"v{i}"] = s
            P._vid_slot_map.append((Vid(idx=None), s))
        return P

    def test_reused_tids_coalesce(self):
        a = Slot(IdType.Tensor, IdSpace.Temp, 0)
        b = Slot(IdType.Tensor, IdSpace.Temp, 0)  # reused idx 0 (disjoint life)
        c = Slot(IdType.Tensor, IdSpace.Temp, 1)
        k = Slot(IdType.Tensor, IdSpace.Constant, 0)
        P = self._builder_with_slots([a, b, c, k], [])

        used, num_tensors, _ = P._collect_used_slots()
        slot_to_tid, _ = P._create_slot_mappings(used)

        self.assertEqual(slot_to_tid[a], slot_to_tid[b], "reused idx must coalesce")
        self.assertNotEqual(slot_to_tid[a], slot_to_tid[c], "distinct idx stays distinct")
        # Counts reflect distinct (id_space, idx), not distinct Slot objects.
        self.assertEqual(num_tensors[IdSpace.Temp], 2)
        self.assertEqual(sum(num_tensors.values()), len(set(slot_to_tid.values())))
        # Emitted Tid references collapse in the serialized graph too.
        ref = {id(s): t for t, s in P._tid_slot_map}
        self.assertEqual(ref[id(a)].idx, ref[id(b)].idx)
        self.assertNotEqual(ref[id(a)].idx, ref[id(c)].idx)

    def test_reused_vids_coalesce(self):
        # SymInt and SymBool share the vid pool, so equal idx must coalesce.
        v0 = Slot(IdType.SymInt, IdSpace.Temp, 0)
        v0b = Slot(IdType.SymBool, IdSpace.Temp, 0)
        v1 = Slot(IdType.SymInt, IdSpace.Temp, 1)
        P = self._builder_with_slots([], [v0, v0b, v1])

        used, _, num_values = P._collect_used_slots()
        _, slot_to_vid = P._create_slot_mappings(used)

        self.assertEqual(slot_to_vid[v0], slot_to_vid[v0b], "shared vid idx must coalesce")
        self.assertNotEqual(slot_to_vid[v0], slot_to_vid[v1])
        self.assertEqual(num_values[IdSpace.Temp], 2)
        self.assertEqual(sum(num_values.values()), len(set(slot_to_vid.values())))


if __name__ == "__main__":
    unittest.main()
