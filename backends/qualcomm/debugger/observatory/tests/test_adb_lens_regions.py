# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the AdbLens region structure (RFC §4.5).

The Qualcomm ADB lens contributes a top-level ``device`` Region opened
lazily on the first patched SimpleADB operation. All ``adb.execute #N``
inference records (and any other ADB-related collects) live directly
under it.

These tests don't run an actual on-device inference. They invoke the
public AdbLens entry points (``note_simple_adb``, ``begin_inference``,
``record_event``, ``end_inference``) directly to exercise the region
state machine.
"""

from __future__ import annotations

import pytest

from executorch.backends.qualcomm.debugger.observatory.lenses.adb import (
    AdbLens,
    AdbExecuteEvent,
)
from executorch.devtools.observatory import Observatory


class _DummyAdb:
    """Minimal stand-in for SimpleADB used by AdbLens.note_simple_adb."""

    host_id = "test-host"
    device_id = "test-device"


@pytest.fixture(autouse=True)
def _reset_observatory():
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = True
    Observatory.register_lens(AdbLens)
    yield
    # AdbLens has class-level state; reset it so tests are independent.
    AdbLens._device_info = []
    AdbLens._device_info_seen = set()
    AdbLens._raw_events = []
    AdbLens._push_groups = []
    AdbLens._pull_groups = []
    AdbLens._open_groups = []
    AdbLens._execute_seq = 0
    AdbLens._active_inference = None
    AdbLens._device_stack = None
    AdbLens._enter_context_fn = None
    Observatory.clear()
    Observatory._lens_registry = []
    Observatory._lenses_initialized = False


def test_no_device_region_until_first_adb_call():
    """The 'device' region is lazy — nothing happens until ADB activity."""

    with Observatory.enter_context("session_no_adb"):
        # No SimpleADB call yet — collect a record from a hypothetical
        # other lens. Its region_stack should not include "device".
        Observatory.collect("non_adb_record", object())

    rec = list(Observatory._records.values())[0]
    assert "device" not in rec.region_stack
    assert AdbLens._device_stack is None


def test_first_simple_adb_call_opens_device_region():
    """note_simple_adb fires _ensure_device_region the first time."""

    with Observatory.enter_context("session_with_adb"):
        AdbLens.note_simple_adb(_DummyAdb())
        # Now the device region is open; subsequent collects pick it up.
        Observatory.collect("inside_device", object())

    rec = list(Observatory._records.values())[0]
    assert rec.region_stack == ["session_with_adb", "device"]


def test_multiple_inference_events_share_one_device_region():
    """Several adb.execute records should land directly under 'device' (no
    per-call sub-region — every region holds >=2 records)."""

    with Observatory.enter_context("session_multi_exec"):
        AdbLens.note_simple_adb(_DummyAdb())
        # Collect two records from inside the lazy device region. We
        # bypass the full inference event flow here because that path
        # has many internal dependencies (event publishing, active
        # inference accumulator) that aren't worth wiring up for a
        # region-structure test.
        Observatory.collect("adb.execute #1", object())
        Observatory.collect("adb.execute #2", object())

    by_region: dict = {}
    for rec in Observatory._records.values():
        by_region.setdefault(tuple(rec.region_stack), []).append(rec.name)

    assert ("session_multi_exec", "device") in by_region
    assert len(by_region[("session_multi_exec", "device")]) == 2


def test_disabled_adb_lens_does_not_open_region():
    """When AdbLens is disabled via config, no device region is opened."""

    with Observatory.enter_context(
        "session_adb_disabled", config={"adb": {"enabled": False}}
    ):
        AdbLens.note_simple_adb(_DummyAdb())
        Observatory.collect("non_adb", object())

    rec = list(Observatory._records.values())[0]
    assert "device" not in rec.region_stack
    assert AdbLens._device_stack is None


def test_device_region_closes_on_session_end():
    """on_session_end closes the lazy device ExitStack cleanly."""

    with Observatory.enter_context("session_close"):
        AdbLens.note_simple_adb(_DummyAdb())
        assert AdbLens._device_stack is not None

    assert AdbLens._device_stack is None


def test_ensure_device_region_idempotent():
    """Multiple ADB calls do not re-open the region."""

    with Observatory.enter_context("session_idempotent"):
        AdbLens.note_simple_adb(_DummyAdb())
        first = AdbLens._device_stack
        AdbLens.note_simple_adb(_DummyAdb())
        second = AdbLens._device_stack

    # Both reference the same ExitStack (was open already by second call).
    assert first is second
